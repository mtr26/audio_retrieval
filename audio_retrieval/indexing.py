import faiss

class Indexer:
    def __init__(self, dim: int = 64):
        """FAISS indexer for efficient similarity search."""
        self.dim = dim
        self.index = faiss.IndexFlatL2(self.dim)
        self.id_to_class = {}

    def _get_classes_(self):
        return set(self.id_to_class.values())

    def add(self, embeddings, classes, start_id=0):
        """
        Add embeddings and their corresponding classes to the index.
        """
        self.index.add(embeddings)
        for i in range(len(embeddings)):
            self.id_to_class[start_id + i] = classes[i]
        return start_id + len(embeddings)
    
    def search(self, query_embeddings, top_k=5):
        """
        Search the index for the top_k most similar embeddings.
        """
        distances, indices = self.index.search(query_embeddings, top_k)
        results = []
        for dist, idx in zip(distances, indices):
            result = [(self.id_to_class[i], d) for i, d in zip(idx, dist)]
            results.append(result)
        return results
    
    def query(self, query_embedding, top_k=5):
        """
        Query the index with a single embedding and return the most frequent label among the top_k results.
        """
        results = self.search(query_embedding, top_k=top_k)
        label_probs = []
        for value in results:
            count_dict = {}
            for label, _ in value:
                if label in count_dict:
                    count_dict[label] += 1
                else:
                    count_dict[label] = 1
            counts = []
            for label, count in count_dict.items():
                counts.append((label, count / top_k))
            label_probs.append(max(counts, key=lambda x: x[1]))
        return label_probs






