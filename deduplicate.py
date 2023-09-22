import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

class Deduplication(object):
    
    def __init__(self, raw_docs, threshold=0.87, n_components=100, vectorizer_params=None) -> None:
        
        self.stopwords_russian = stopwords.words("russian")
        self.raw_docs = raw_docs
        self.colname = 'text'
        self.svd = TruncatedSVD(n_components=n_components)
        vectorizer_params = vectorizer_params or {'stop_words': self.stopwords_russian}
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        self.threshold = threshold
        self.vectors = self.__vectorize_text()
        
    def __vectorize_text(self):
        
        vectors = self.vectorizer.fit_transform(self.raw_docs[self.colname])
        return self.svd.fit_transform(vectors)  # Используем TruncatedSVD вместо PCA
    
     
    def remove_duplicates(self):
        
        index_to_drop = set()
        
        for chunk_start in range(0, self.vectors.shape[0], 500):
        
            chunk_end = min(chunk_start + 500, self.vectors.shape[0])
            print(f"Processing rows {chunk_start} to {chunk_end}")

            # Using cosine_similarity
            chunk_similarities = cosine_similarity(self.vectors[chunk_start:chunk_end], self.vectors)

            # Identifying duplicate indices
            duplicates = np.argwhere(chunk_similarities >= self.threshold)
            valid_duplicates = [(chunk_start + x, y) for x, y in duplicates if (chunk_start + x) != y]

            # Update the index_to_drop set
            for duplicate in valid_duplicates:
                
                duplicate_texts = [(i, len(self.raw_docs[self.colname][i])) for i in duplicate]
                max_len_text_index = max(duplicate_texts, key=lambda x: x[1])[0]
                delete_indexes = [i for i in duplicate if i != max_len_text_index]
                index_to_drop.update(delete_indexes)

        self.raw_docs = self.raw_docs.drop(index = list(index_to_drop))
        
        return self.raw_docs
    
    def find_similar_news(self, new_query):
        
        print(f"Shape of tfidf matrix: {self.vectors.shape}")
    
        new_query = [new_query]
        new_query_vector = self.vectorizer.transform(new_query)
        sim = cosine_similarity(X = self.vectors, Y = new_query_vector)
        print(sim[0][0])
        ind = np.argsort(sim,axis = 0)[::-1][:5]
        print(ind.reshape(-1))
        for i in ind:
            print(i) # index number
            print(self.raw_docs[self.colname].tolist()[i[0]])