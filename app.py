import streamlit as st
from loader import load_data
from inverted_index import inverted_index
from search import search
from idf import compute_idf
from rank_candidates import bm25_rank
from tokenizer import tokenizer

def get_preview(text, max_chars=300):
    text = text.replace("\n", " ")
    return text[:max_chars] + "..." if len(text) > max_chars else text

st.set_page_config(page_title="Mini Search Engine", layout="wide")
st.title("🔍 Mini Search Engine")

@st.cache_data
def prepare_engine():
    documents = load_data("data")
    index, doc_lengths = inverted_index(documents)
    avg_dl = sum(doc_lengths.values()) / len(doc_lengths)
    idf = compute_idf(index, len(documents))
    return documents, index, doc_lengths, avg_dl, idf

documents, index, doc_lengths, avg_dl, idf = prepare_engine()

query = st.text_input("Enter your search query")
top_k = st.number_input("Top results", min_value=1, max_value=20, value=5)
search_button = st.button("Search")

if search_button:
    if not query.strip():
        st.warning("Please enter a query")
        st.stop()

    candidates = search(query, index, mode="or")
    query_tokens = tokenizer(query)

    scores = bm25_rank(query_tokens, candidates, index, idf, doc_lengths, avg_dl)

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    st.subheader("Search Results")
    if not ranked_results:
        st.info("No matching documents found.")
    else:
        for rank, (doc_id, score) in enumerate(ranked_results, start=1):
            st.markdown(f"### {rank}. {doc_id}")
            st.write(f"**Score:** {score:.4f}")
            st.write(get_preview(documents[doc_id]))
            st.markdown("---")