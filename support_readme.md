# Customer Support Agent (LangGraph)

A multi-tool support agent: checks an FAQ first (RAG), falls back to a web search,
and escalates to a human when it is out of its depth. It keeps conversation history per session.

## Flow

```
START -> triage --(sensitive / asks for human)--------------> escalate -> END
            |
            +--(normal)--> faq_search --(match)--------------> answer -> END
                               |
                               +--(no match)--> web_search --(results)--> answer
                                                    |
                                                    +--(nothing found)--> escalate
```

| Node | Role | Tool |
|---|---|---|
| `triage` | LLM decides ESCALATE / NORMAL (legal threats, fraud, anger, asks for a human, account-specific disputes) | LLM |
| `faq_search` | Vector search over `faq.json`; a hit only counts if cosine distance <= threshold | RAG tool |
| `web_search` | DuckDuckGo search, optionally restricted to one site | Search tool |
| `answer` | LLM writes the reply from the context + conversation history | LLM |
| `escalate` | Writes a ticket (with full transcript) to `tickets.json`, tells the customer the ticket ID | Escalation tool |

## Files
- `support_graph.py` - the LangGraph (no Streamlit, testable on its own)
- `support_app.py` - Streamlit UI
- `faq.json` - your FAQ database (edit it, restart the app)
- `test_support_graph.py` - offline routing tests (no API key needed)

## Run (Windows)

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r support_requirements.txt
python test_support_graph.py
streamlit run support_app.py
```

Paste your Groq key in the sidebar. Try these in order:
1. "How long does shipping take?" -> answered from FAQ
2. "Do you ship to Nepal by drone?" (or any off-topic question) -> web search
3. "I want to speak to a manager, this is fraud" -> escalated, ticket created
4. After an FAQ answer, ask "and how much does that cost?" -> follow-up uses history

## Things to know
- **Memory** is `MemorySaver` + a `thread_id`: it lives in RAM, so it resets when the app restarts. Swap in `SqliteSaver` for persistence.
- **Threshold** (sidebar) is the key tuning knob. Too high = irrelevant FAQ answers; too low = everything goes to web search. Each answer shows the match distance so you can tune it.
- **Web search** uses `duckduckgo-search`, which can be rate-limited. If it returns nothing the agent escalates instead of guessing.
- The agent never has access to real order data, so order-specific disputes are escalated by design.

## Extending
- Replace `faq.json` with PDFs using the chunking code from your Clinical QA app
- Add a "check order status" tool that calls a real API
- Use `interrupt()` in LangGraph for a true human-in-the-loop approval step
