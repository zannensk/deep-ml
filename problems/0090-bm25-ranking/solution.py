import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
	# Your code here
	N = len(corpus)

	if N == 0:
		return np.array([])
	doc_lengths = [len(doc) for doc in corpus]
	avgdl = np.mean(doc_lengths)
	doc_freq = Counter()
	for doc in corpus:
		doc_freq.update(set(doc))
	
	scores = np.zeros(N)
	for i, doc in enumerate(corpus):
		tf = Counter(doc)
		dl = doc_lengths[i]

		for term in query:
			if tf[term] == 0:
				continue
			
			idf = np.log((N + 1) / (doc_freq[term] + 1))
			denominator = (
				tf[term] + k1 * (1 - b + b * dl / avgdl)
			)
			scores[i] += idf * (
				tf[term] * (k1 + 1) / denominator
			)
		
	return np.round(scores, 3)