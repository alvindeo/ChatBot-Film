class boolean_ir:
    def __init__(self):
        self.inverted_index = defaultdict(set)

    def build_index(self, documents):
        for doc_id, doc_tokens in enumerate(documents):
            for token in doc_tokens:
                self.inverted_index[token].add(doc_id)

    def query(self, q_tokens):
        result = None
        for token in q_tokens:
            docs = self.inverted_index.get(token, set())
            if result is None:
                result = docs
            else:
                result = result.intersection(docs)
        return result if result else set()
