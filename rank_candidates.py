from tokenizer import tokenizer
from collections import defaultdict
import math

#cosine similarity based ranking
def build_query_vector(query, idf, tokenizer):
    tokens = tokenizer(query)

    tf = defaultdict(int)
    for word in tokens:
        tf[word] += 1

    query_vector = {}

    for word, freq in tf.items():
        if word in idf:
            query_vector[word] = freq * idf[word]

    return query_vector

def build_document_vector(doc_id, query_vector, index, idf):
    doc_vector = {}

    for word in query_vector:
        if word in index and doc_id in index[word]:
            tf = index[word][doc_id]
            doc_vector[word] = tf * idf[word]
        else:
            doc_vector[word] = 0.0

    return doc_vector

def cosine_similarity(query_vector, doc_vector):
    dot_product = 0.0
    query_mag = 0.0
    doc_mag = 0.0

    for word in query_vector:
        q_weight = query_vector[word]
        d_weight = doc_vector[word]

        dot_product += q_weight * d_weight
        query_mag += q_weight ** 2
        doc_mag += d_weight ** 2

    if query_mag == 0 or doc_mag == 0:
        return 0.0

    return dot_product / (math.sqrt(query_mag) * math.sqrt(doc_mag))

# Simple TF-IDF based ranking
def rank_candidates(query, candidates, index,idf):
    results = {}
    for candidate in candidates:
        score = 0.0
        for word in tokenizer(query):
            if word in index and candidate in index[word]:
                tf = index[word][candidate]
                idf_value = idf.get(word, 0.0)
                score += tf * idf_value
        if score > 0.0:
            results[candidate] = score
    return results

# BM25 based ranking- currently used in main.py
def bm25_rank(query_tokens, candidates, index, idf, doc_lengths, avg_dl, k1=1.5, b=0.75):
    scores = {}

    for doc_id in candidates:
        score = 0.0
        dl = doc_lengths[doc_id]

        for word in query_tokens:
            if word not in index or doc_id not in index[word]:
                continue

            tf = index[word][doc_id]
            idf_value = idf[word]

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avg_dl))

            score += idf_value * (numerator / denominator)

        scores[doc_id] = score

    return scores