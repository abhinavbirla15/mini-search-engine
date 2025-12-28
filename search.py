from tokenizer import tokenizer

def search_and(query, index):
    query_words = tokenizer(query)

    if not query_words:
        return set()
    
    result_sets = None

    for word in query_words:
        if word in index:
            return set()

        if result_sets is None:
            result_sets = index[word].copy()
        else:
            result_sets = result_sets.intersection(index[word])
    
    return result_sets

def search_or(query, index):
    query_words = tokenizer(query)

    if not query_words:
        return set()

    result_set = set()

    for word in query_words:
        if word in index:
            result_set = result_set.union(index[word])

    return result_set

def search(query, index, mode = 'or'):
    if mode == 'and':
        return search_and(query, index)
    elif mode == 'or':
        return search_or(query, index)
    else:
        raise ValueError("Invalid search mode. Use 'and' or 'or'.")