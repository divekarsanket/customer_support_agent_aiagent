"""Offline routing test: fake LLM + fake vectorstore + fake web search. No API keys needed."""
import os
import tempfile

import support_graph as sg
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

sg.TICKETS_PATH = os.path.join(tempfile.mkdtemp(), "tickets.json")


class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        first = str(messages[0].content)
        if first.startswith("You are a triage system"):
            last = str(messages[1].content).lower()
            return AIMessage(content="ESCALATE" if ("lawyer" in last or "human" in last) else "NORMAL")
        return AIMessage(content=f"ANSWER[{'faq' if 'SOURCE: faq' in first else 'web'}]")


class FakeStore:
    def similarity_search_with_score(self, query, k=3):
        if "return" in query.lower():
            return [(Document(page_content="Q: return policy\nA: 30 days"), 0.2)]
        return [(Document(page_content="Q: unrelated\nA: x"), 0.9)]


def run(msg, graph, tid):
    return graph.invoke({"messages": [HumanMessage(content=msg)]}, {"configurable": {"thread_id": tid}})


def make(search_fn):
    return sg.build_graph(FakeLLM(), FakeStore(), MemorySaver(), threshold=0.5, search_fn=search_fn)


# 1. FAQ hit -> answer from faq, no web call
web_calls = []
g = make(lambda q: web_calls.append(q) or [{"title": "t", "body": "b", "href": "u"}])
r = run("What is your return policy?", g, "t1")
assert r["source"] == "faq" and not web_calls, r
print("1 FAQ hit      ->", r["messages"][-1].content)

# 2. FAQ miss -> web search -> answer from web
r = run("Do you sell gift cards?", g, "t2")
assert r["source"] == "web" and len(web_calls) == 1, r
print("2 FAQ miss     ->", r["messages"][-1].content)

# 3. FAQ miss + empty web -> escalate (no answer found)
g3 = make(lambda q: [])
r = run("Do you sell gift cards?", g3, "t3")
assert r["source"] == "human" and r["ticket_id"].startswith("TCK-"), r
print("3 No answer    ->", r["messages"][-1].content)

# 4. Triage escalation -> straight to human, FAQ never searched
r = run("I want to talk to a human, I'm calling my lawyer", g, "t4")
assert r["source"] == "human", r
print("4 Triage       ->", r["messages"][-1].content)

# 5. Conversation history persists per thread and is isolated between threads
r = run("What is your return policy?", g, "t5")
r = run("And what about refunds?", g, "t5")
assert len(r["messages"]) == 4, len(r["messages"])
r_other = run("Hello", g, "t6")
assert len(r_other["messages"]) <= 2
print("5 History      -> thread t5 holds", len(r["messages"]), "messages; thread t6 is separate")

# 6. Per-turn reset: stale context from turn 1 must not leak into a turn-2 escalation
r = run("What is your return policy?", g3, "t7")
assert r["source"] == "faq"
r = run("Do you sell gift cards?", g3, "t7")
assert r["source"] == "human", "stale FAQ context leaked into next turn"
print("6 Turn reset   -> OK")

# 7. Real FAQStore ranking with deterministic fake embeddings (no model download)
import numpy as np

class FakeEmb:
    VOCAB = ["return", "refund", "shipping", "password", "warranty", "pay", "order"]
    def _v(self, t):
        t = t.lower()
        return [float(t.count(w)) + 0.01 for w in self.VOCAB]
    def embed_documents(self, texts): return [self._v(t) for t in texts]
    def embed_query(self, t): return self._v(t)

docs = [Document(page_content="Q: return policy\nA: 30 days return refund"),
        Document(page_content="Q: shipping time\nA: shipping 5 days"),
        Document(page_content="Q: password reset\nA: reset password link")]
store = sg.FAQStore(docs, FakeEmb())
hits = store.similarity_search_with_score("how do I reset my password", k=2)
assert "password" in hits[0][0].page_content, hits
assert hits[0][1] < hits[1][1], "closest hit must have the smallest distance"
assert 0.0 <= hits[0][1] <= 2.0
# same store works end-to-end inside the graph
g7 = sg.build_graph(FakeLLM(), store, MemorySaver(), threshold=0.3, search_fn=lambda q: [])
r = run("shipping shipping please", g7, "t8")
assert r["source"] == "faq", r
print("7 FAQStore     -> ranks by cosine distance, works in graph")

# 8. Web search that RAISES -> escalates, and the real error is reported, not hidden
def broken_search(q):
    raise RuntimeError("202 Ratelimit")

g8 = make(broken_search)
r = run("Do you sell gift cards?", g8, "t9")
assert r["source"] == "human" and "Ratelimit" in r["search_error"], r
print("8 Search error ->", r["search_error"])

# 9. FAQStore embeds the bare question, not the noisy answer text
noisy = [Document(page_content="Q: hours\nA: " + "return refund shipping pay order warranty " * 5,
                  metadata={"question": "password reset"}),
         Document(page_content="Q: other\nA: x", metadata={"question": "shipping"})]
hit = sg.FAQStore(noisy, FakeEmb()).similarity_search_with_score("password", k=1)[0]
assert hit[0].metadata["question"] == "password reset", hit
print("9 Question-only embedding -> OK")

# 10. Alternate phrasings pull a paraphrased question much closer
class HoursEmb:
    VOCAB = ["hours", "open", "password", "shipping"]
    def _v(self, t):
        t = t.lower(); return [float(t.count(w)) + 0.01 for w in self.VOCAB]
    def embed_documents(self, texts): return [self._v(t) for t in texts]
    def embed_query(self, t): return self._v(t)

with_alt = [Document(page_content="Q: support hours\nA: 9-6",
                     metadata={"question": "support hours", "alt_questions": ["when are you open"]}),
            Document(page_content="Q: shipping\nA: 5 days", metadata={"question": "shipping"})]
no_alt = [Document(page_content="Q: support hours\nA: 9-6", metadata={"question": "support hours"}),
          Document(page_content="Q: shipping\nA: 5 days", metadata={"question": "shipping"})]
d_alt = sg.FAQStore(with_alt, HoursEmb()).similarity_search_with_score("are you open", k=1)[0]
d_no = sg.FAQStore(no_alt, HoursEmb()).similarity_search_with_score("are you open", k=1)[0]
assert d_alt[0].metadata["question"] == "support hours" and d_alt[1] < d_no[1] and d_alt[1] < 0.3, (d_alt, d_no)
print(f"10 Alt phrasings -> distance {d_no[1]:.2f} without, {d_alt[1]:.2f} with")

import json
faq = json.load(open(sg.FAQ_PATH, encoding="utf-8"))
assert all(len(f["alt_questions"]) >= 3 for f in faq) and len(sg.load_faq_documents()) == len(faq)

tickets = json.load(open(sg.TICKETS_PATH))
assert len(tickets) == 4 and tickets[0]["transcript"], tickets
print(f"\nAll routing tests passed. {len(tickets)} tickets written.")
