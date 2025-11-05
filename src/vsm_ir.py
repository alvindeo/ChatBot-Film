class vsm_ir:
    def __init__(self, documents):
        self.documents = documents
        self.vocab = self.build_vocab(documents)
        self.doc_vectors = self.build_doc_vectors(documents, self.vocab)
        self.idf = self.compute_idf(documents, self.vocab)

    def build_vocab(self, docs):
        vocab = set()
        for d in docs:
            vocab.update(d)
        return list(vocab)

    def compute_idf(self, docs, vocab):
        N = len(docs)
        idf = {}
        for term in vocab:
            df = sum(1 for d in docs if term in d)
            idf[term] = np.log((N + 1) / (df + 1)) + 1
        return idf

    def tfidf(self, doc):
        tf = Counter(doc)
        vec = np.zeros(len(self.vocab))
        for i, term in enumerate(self.vocab):
            vec[i] = tf[term] * self.idf.get(term, 0)
        return vec

    def build_doc_vectors(self, docs, vocab):
        return np.array([self.tfidf(d) for d in docs])

    def query_vector(self, query):
        return self.tfidf(query)

    def cosine_similarity(self, vec1, vec2):
        num = np.dot(vec1, vec2)
        den = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return num / den if den != 0 else 0

    def rank(self, query, top_k=3):
        q_vec = self.query_vector(query)
        scores = np.array([self.cosine_similarity(q_vec, d_vec) for d_vec in self.doc_vectors])
        top_indices = scores.argsort()[-top_k:][::-1]
        return [(i, scores[i]) for i in top_indices if scores[i] > 0]
