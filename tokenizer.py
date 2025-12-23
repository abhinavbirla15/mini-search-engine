import re

STOPWORDS = {
    "the", "is", "a", "an", "of", "to", "in", "on", "for",
    "and", "or", "with", "as", "by", "at", "from"
}

def tokenizer(text):
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    return [t for t in tokens if t not in STOPWORDS]