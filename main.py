from loader import load_data
from inverted_index import inverted_index
from search import search
from idf import compute_idf
from rank_candidates import bm25_rank
from tokenizer import tokenizer

def main():
    # Load documents
    documents = load_data('data')

    # Build inverted index
    index, doc_lengths= inverted_index(documents)
    avg_dl = sum(doc_lengths.values()) / len(doc_lengths)

    # Compute IDF values
    total_docs = len(documents)
    idf = compute_idf(index, total_docs)

    # Example query
    query = input("Enter your search query: ").strip()
    if not query:
        print("Empty query provided.")
        return

    # Search documents
    candidates = search(query, index, mode='or')

    # Rank candidates
    query_tokens = tokenizer(query)
    scores= bm25_rank(query_tokens, candidates, index, idf, doc_lengths, avg_dl)

    # Sort results by score
    ranked_results = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


    # Print ranked results
    for doc_id, score in ranked_results.items():
        print(f"Document ID: {doc_id}, Score: {score:.4f}") 

if __name__ == "__main__":
    main()