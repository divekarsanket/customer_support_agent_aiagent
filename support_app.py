"""
Customer Support Agent — Streamlit UI
--------------------------------------
Skills: LangGraph, multi-tool agent (RAG + web search + escalation), memory

Run with:
    streamlit run support_app.py
"""

import json
import uuid

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from support_graph import COMPANY, FAQStore, build_graph, load_faq_documents, load_tickets

st.set_page_config(page_title="Customer Support Agent", page_icon="🎧", layout="centered")
st.title(f"🎧 {COMPANY} Support Agent")
st.caption("LangGraph · FAQ search (RAG) → web search → human escalation")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", ["Groq (free tier)", "OpenAI"])
    if provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        model_options = ["gpt-4o-mini", "gpt-4o"]
    else:
        api_key = st.text_input("Groq API Key", type="password")
        model_options = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"]
    model_name = st.selectbox("Model", model_options)

    threshold = st.slider("FAQ match threshold", 0.2, 1.0, 0.5, 0.05,
                          help="Max cosine distance for an FAQ entry to count as a match. "
                               "Lower = stricter.")
    site = st.text_input("Restrict web search to site (optional)", placeholder="help.example.com")

    if st.button("🗑️ New conversation"):
        st.session_state.pop("thread_id", None)
        st.session_state.pop("display", None)
        st.rerun()

# ----------------------------
# Cached resources
# ----------------------------
@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90MB)...")
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Plain in-memory store: cosine distance, lower = closer. No database involved.
    return FAQStore(load_faq_documents(), embeddings)


@st.cache_resource
def get_checkpointer():
    return MemorySaver()  # remembers every thread's conversation


def build_llm():
    if provider == "OpenAI":
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=0.2)
    return ChatGroq(api_key=api_key, model=model_name, temperature=0.2)


# ----------------------------
# Session
# ----------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "display" not in st.session_state:
    st.session_state.display = []  # (role, text, meta)

SOURCE_BADGE = {
    "faq": "📚 Answered from our FAQ",
    "web": "🌐 Answered from a web search",
    "human": "🧑‍💼 Escalated to a human agent",
}

for role, text, meta in st.session_state.display:
    with st.chat_message(role):
        st.markdown(text)
        if meta:
            st.caption(meta)

question = st.chat_input("How can we help you today?")

if question:
    if not api_key:
        st.error(f"Enter your {provider} API key in the sidebar first.")
        st.stop()

    st.session_state.display.append(("user", question, ""))
    with st.chat_message("user"):
        st.markdown(question)

    graph = build_graph(
        llm=build_llm(),
        vectorstore=get_vectorstore(),
        checkpointer=get_checkpointer(),
        threshold=threshold,
        site=site.strip(),
    )
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Looking into it..."):
            try:
                result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        reply = result["messages"][-1].content
        meta = SOURCE_BADGE.get(result.get("source", ""), "")
        if result.get("route") == "normal":  # FAQ was actually searched
            meta += f" · closest FAQ distance {result['best_score']:.2f} (threshold {threshold:.2f})"
        if result.get("search_error"):
            meta += f" · web search problem: {result['search_error']}"
        st.markdown(reply)
        if meta:
            st.caption(meta)

    st.session_state.display.append(("assistant", reply, meta))

# ----------------------------
# Escalated tickets (what a human team would see)
# ----------------------------
with st.expander("🎫 Escalation tickets (human queue)"):
    tickets = load_tickets()
    if not tickets:
        st.write("No tickets yet.")
    for t in reversed(tickets):
        st.markdown(f"**{t['ticket_id']}** · {t['created_at']} · _{t['reason']}_")
        st.markdown(f"> {t['last_customer_message']}")
        with st.popover("Transcript"):
            st.code(json.dumps(t["transcript"], indent=2, ensure_ascii=False), language="json")
