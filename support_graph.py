"""
Customer Support Agent — LangGraph core (no Streamlit in this file)
--------------------------------------------------------------------
Graph:

    START -> triage --(sensitive/complex)--------------------> escalate -> END
                |
                +--(normal)--> faq_search --(match)-----------> answer -> END
                                   |
                                   +--(no match)--> web_search --(results)--> answer
                                                        |
                                                        +--(nothing)--------> escalate

Tools / skills:
  - RAG tool        : faq_search   (vector search over faq.json)
  - Search tool     : web_search   (DuckDuckGo)
  - Escalation tool : escalate     (writes a ticket to tickets.json)
  - Conversation history is kept by LangGraph's checkpointer (thread_id)
"""

import json
import os
import uuid
from datetime import datetime
from typing import Annotated, Callable, Dict, List, Optional, TypedDict

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "faq.json")
TICKETS_PATH = os.path.join(BASE_DIR, "tickets.json")
COMPANY = "ShopEasy"


# ----------------------------
# 1. Graph state — the shared data every node reads/writes
# ----------------------------
class SupportState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # full conversation history
    route: str        # "normal" | "escalate"   (set by triage)
    context: str      # text handed to the answer node (FAQ or web)
    source: str       # "faq" | "web" | "human" | ""
    best_score: float  # best FAQ distance this turn (lower = closer match)
    ticket_id: str
    search_error: str  # why web search returned nothing (if it errored)


# ----------------------------
# 2. FAQ loading + ticket storage helpers
# ----------------------------
def load_faq_documents() -> List[Document]:
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    return [
        Document(
            page_content=f"Q: {item['question']}\nA: {item['answer']}",
            metadata={"category": item.get("category", "general"), "question": item["question"],
                      "alt_questions": item.get("alt_questions", [])},
        )
        for item in items
    ]


class FAQStore:
    """
    Tiny in-memory vector store for the FAQ (14 entries doesn't need a database).
    Embeds every FAQ entry once, then ranks by cosine distance (0 = identical,
    lower = closer). Same interface as a Chroma store's
    similarity_search_with_score(), so build_graph() doesn't care which it gets.

    Pure numpy, no threads or files involved -> safe under Streamlit's reruns.
    """

    def __init__(self, documents: List[Document], embeddings):
        self.documents = documents
        self.embeddings = embeddings
        # One vector per phrasing: the main question plus any alternate phrasings.
        # A document's score is its CLOSEST phrasing, so "when are you open?" can
        # match the entry "What are your support hours?".
        texts, owners = [], []
        for i, d in enumerate(documents):
            variants = [d.metadata.get("question") or d.page_content]
            variants += list(d.metadata.get("alt_questions", []))
            texts += variants
            owners += [i] * len(variants)
        self.owner = np.array(owners)
        self.matrix = self._normalize(np.array(embeddings.embed_documents(texts), dtype=float))

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    def similarity_search_with_score(self, query: str, k: int = 3):
        q = self._normalize(np.array(self.embeddings.embed_query(query), dtype=float))
        variant_dist = 1.0 - self.matrix @ q
        best = np.full(len(self.documents), 2.0)
        np.minimum.at(best, self.owner, variant_dist)  # closest phrasing per document
        order = np.argsort(best)[:k]
        return [(self.documents[i], float(best[i])) for i in order]


def load_tickets() -> List[dict]:
    if not os.path.exists(TICKETS_PATH):
        return []
    try:
        with open(TICKETS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_ticket(ticket: dict) -> None:
    tickets = load_tickets()
    tickets.append(ticket)
    with open(TICKETS_PATH, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2, ensure_ascii=False)


# ----------------------------
# 3. Search tool (DuckDuckGo) — imported lazily so tests don't need it
# ----------------------------
def duckduckgo_search(query: str, max_results: int = 4) -> List[Dict]:
    """Raises on failure (rate limit, offline...) so the caller can report why."""
    try:
        from ddgs import DDGS  # maintained successor of duckduckgo-search
    except ImportError:
        from duckduckgo_search import DDGS  # old package name, fallback
    return list(DDGS().text(query, max_results=max_results))


# ----------------------------
# 4. Prompts
# ----------------------------
TRIAGE_PROMPT = (
    "You are a triage system for a customer support team. Decide whether the "
    "customer's LATEST message must be handled by a human agent.\n\n"
    "Reply ESCALATE if any of these apply:\n"
    "- The customer asks for a human, agent, or manager\n"
    "- Legal threats, chargebacks, fraud, or account-security problems\n"
    "- Damaged or dangerous products, injuries, or a very angry/abusive tone\n"
    "- A dispute about a specific order, payment, or refund that needs account access\n\n"
    "General how-to or policy questions (shipping times, return policy, tracking, "
    "payment methods) are NORMAL even if they mention an order.\n\n"
    "Reply with exactly one word: ESCALATE or NORMAL."
)

ANSWER_PROMPT = """You are a friendly, concise customer support agent for {company}.

Answer the customer's latest question using ONLY the context below. Use the conversation so far to understand follow-up questions.

Rules:
- If SOURCE is "faq", this is official {company} policy: answer confidently.
- If SOURCE is "web", this is general information from the internet and may not match {company}'s own policy: say so briefly.
- If the context only partly answers the question, share what you can and offer to connect the customer with a human agent.
- Never invent order details, prices, or policies that are not in the context.

SOURCE: {source}

CONTEXT:
{context}"""


# ----------------------------
# 5. Helpers
# ----------------------------
def _last_user_text(state: SupportState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _format_conversation(messages: List[BaseMessage]) -> str:
    lines = []
    for m in messages:
        who = "Customer" if isinstance(m, HumanMessage) else "Agent"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


# ----------------------------
# 6. Build the graph
# ----------------------------
def build_graph(
    llm,
    vectorstore,
    checkpointer,
    threshold: float = 0.5,
    site: str = "",
    top_k: int = 3,
    search_fn: Optional[Callable[[str], List[Dict]]] = None,
):
    """
    llm          : any LangChain chat model (ChatGroq / ChatOpenAI)
    vectorstore  : Chroma store built with cosine distance (lower = closer)
    checkpointer : e.g. MemorySaver() — this is what remembers the conversation
    threshold    : max cosine distance to count an FAQ hit as a real match
    site         : optional domain to restrict web search to, e.g. "help.example.com"
    """
    search = search_fn or duckduckgo_search

    # ---- Node: triage — should a human handle this right away? ----
    def triage(state: SupportState):
        recent = state["messages"][-4:]
        resp = llm.invoke([
            SystemMessage(content=TRIAGE_PROMPT),
            HumanMessage(content=_format_conversation(recent)),
        ])
        verdict = "escalate" if "ESCALATE" in str(resp.content).upper() else "normal"
        # Reset per-turn fields so nothing stale leaks in from the previous turn
        return {"route": verdict, "context": "", "source": "", "best_score": 1.0, "ticket_id": "", "search_error": ""}

    # ---- Node: faq_search — the RAG tool ----
    def faq_search(state: SupportState):
        query = _last_user_text(state)
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        best = min((score for _, score in results), default=1.0)
        matches = [doc for doc, score in results if score <= threshold]
        if matches:
            return {
                "context": "\n\n".join(doc.page_content for doc in matches),
                "source": "faq",
                "best_score": best,
            }
        return {"context": "", "source": "", "best_score": best}

    # ---- Node: web_search — the search tool ----
    def web_search(state: SupportState):
        query = _last_user_text(state)
        if site:
            query = f"site:{site} {query}"
        try:
            hits = search(query)
            error = "" if hits else "search returned no results"
        except Exception as e:  # rate limit, no internet, package problem...
            hits, error = [], f"{type(e).__name__}: {e}"[:200]
        if hits:
            context = "\n\n".join(
                f"{h.get('title', '')}: {h.get('body', '')} (URL: {h.get('href', '')})"
                for h in hits
            )
            return {"context": context, "source": "web", "search_error": ""}
        return {"context": "", "source": "", "search_error": error}

    # ---- Node: answer — the LLM writes the reply from the context ----
    def answer(state: SupportState):
        system = ANSWER_PROMPT.format(
            company=COMPANY, source=state["source"], context=state["context"]
        )
        # Full history (capped) so follow-up questions make sense
        resp = llm.invoke([SystemMessage(content=system)] + state["messages"][-12:])
        return {"messages": [AIMessage(content=resp.content)]}

    # ---- Node: escalate — the escalation tool ----
    def escalate(state: SupportState):
        ticket_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
        by_triage = state.get("route") == "escalate"
        save_ticket({
            "ticket_id": ticket_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "reason": "sensitive_or_complex" if by_triage else "no_answer_found",
            "last_customer_message": _last_user_text(state),
            "search_error": state.get("search_error", ""),
            "transcript": [
                {"role": "customer" if isinstance(m, HumanMessage) else "agent",
                 "text": str(m.content)}
                for m in state["messages"]
            ],
        })
        why = (
            "this needs a closer look than I can give"
            if by_triage
            else "I couldn't find a reliable answer to this"
        )
        reply = (
            f"I'm connecting you with a human support specialist because {why}. "
            f"Your ticket number is **{ticket_id}** — a team member will follow up "
            f"by email within 24 hours."
        )
        return {"messages": [AIMessage(content=reply)], "source": "human", "ticket_id": ticket_id}

    # ---- Routing functions (the conditional edges) ----
    def after_triage(state: SupportState) -> str:
        return "escalate" if state["route"] == "escalate" else "faq_search"

    def after_faq(state: SupportState) -> str:
        return "answer" if state["context"] else "web_search"

    def after_web(state: SupportState) -> str:
        return "answer" if state["context"] else "escalate"

    # ---- Wire it together ----
    graph = StateGraph(SupportState)
    graph.add_node("triage", triage)
    graph.add_node("faq_search", faq_search)
    graph.add_node("web_search", web_search)
    graph.add_node("answer", answer)
    graph.add_node("escalate", escalate)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", after_triage,
                                {"escalate": "escalate", "faq_search": "faq_search"})
    graph.add_conditional_edges("faq_search", after_faq,
                                {"answer": "answer", "web_search": "web_search"})
    graph.add_conditional_edges("web_search", after_web,
                                {"answer": "answer", "escalate": "escalate"})
    graph.add_edge("answer", END)
    graph.add_edge("escalate", END)

    return graph.compile(checkpointer=checkpointer)
