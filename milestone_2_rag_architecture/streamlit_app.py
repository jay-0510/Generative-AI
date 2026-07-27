"""Simple Streamlit user interface for the FastAPI RAG endpoint."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Milestone 2 RAG", page_icon="?", layout="centered")
st.title("Document Q/A")
st.caption("S3 + PyMuPDF + LangChain + Bedrock + OpenSearch")

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        for source in item.get("sources", []):
            st.caption(f"{source['s3_uri']} | chunk {source['chunk_index']} | score {source['score']:.3f}")

question = st.chat_input("Ask a question about the indexed documents")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=90)
            response.raise_for_status()
            payload = response.json()
            st.markdown(payload["answer"])
            for source in payload["sources"]:
                st.caption(f"{source['s3_uri']} | chunk {source['chunk_index']} | score {source['score']:.3f}")
            st.session_state.history.append(
                {"role": "assistant", "content": payload["answer"], "sources": payload["sources"]}
            )
        except requests.RequestException as error:
            st.error(f"The API is unavailable: {error}")
