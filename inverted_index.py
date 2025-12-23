from collections import defaultdict
from tokenizer import tokenizer


def inverted_index(documents):
    inverted_index=defaultdict(dict)
    doc_length={}

    for doc_id, content in documents.items():
        words=tokenizer(content)
        doc_length[doc_id]=len(words)

        for word in words:
            inverted_index[word][doc_id]=inverted_index[word].get(doc_id,0)+1 
              
    return inverted_index, doc_length