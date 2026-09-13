import streamlit as st
import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Enterprise Financial 10-K RAG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Enterprise Financial 10-K RAG Engine")
st.caption("Docling Layout Parsing • PostgreSQL pgvector (HNSW + GIN) • FlashRank Re-ranking • Gemini 2.5 Flash")

# Sidebar: Document Ingestion
with st.sidebar:
    st.header("📥 Ingest Document")
    uploaded_file = st.file_uploader("Upload SEC 10-K Filing (PDF)", type=["pdf"])
    company_name = st.text_input("Company Name", value="Alphabet Inc.")
    fiscal_year = st.text_input("Fiscal Year", value="2024")

    if st.button("Ingest & Index", type="primary", disabled=uploaded_file is None):
        with st.spinner("Parsing layout with Docling and indexing in PostgreSQL..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"company_name": company_name, "fiscal_year": fiscal_year}
            try:
                res = requests.post(f"{API_BASE_URL}/api/ingest", files=files, data=data, timeout=300)
                if res.status_code == 200:
                    resp_json = res.json()
                    st.success(f"Indexed {resp_json['child_chunks_indexed']} child chunks across {resp_json['parent_documents_created']} parent documents.")
                else:
                    st.error(f"Ingestion failed: {res.text}")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")

    st.markdown("---")
    st.markdown("### ⚙️ Engine Settings")
    top_k = st.slider("Parent Contexts (Top K)", min_value=1, max_value=5, value=3)

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Query Input
user_query = st.chat_input("Ask a financial question (e.g., 'What were Alphabet's cloud revenues and operating margins?')...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        start_time = time.time()

        try:
            req_payload = {"query": user_query, "top_k": top_k}
            res = requests.post(
                f"{API_BASE_URL}/api/chat/stream",
                json=req_payload,
                stream=True,
                timeout=60
            )

            if res.status_code == 200:
                for line in res.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                payload = json.loads(data_str)
                                if "token" in payload:
                                    full_response += payload["token"]
                                    response_container.markdown(full_response + "▌")
                            except json.JSONDecodeError:
                                pass

                elapsed = time.time() - start_time
                response_container.markdown(full_response)
                st.caption(f"⏱️ Generated in {elapsed:.2f}s")
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"API Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")