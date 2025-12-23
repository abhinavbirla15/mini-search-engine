import math

def compute_idf(inverted_index, total_docs):
    idf = {}

    for word, doc_dict in inverted_index.items():
        df = len(doc_dict)
        idf[word] = math.log((total_docs + 1) / (df + 1)) + 1.0

    return idf
