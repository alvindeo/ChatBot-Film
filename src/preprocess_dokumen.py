documents = []
for filepath in glob.glob('data/*.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '->' in line:
                q, a = line.split('->')
                # Add both question and answer to documents for chatbot corpus
                documents.append(q.strip())
                documents.append(a.strip())

preprocessed_docs = [preprocess(doc) for doc in documents]
print(preprocessed_docs[:3])
