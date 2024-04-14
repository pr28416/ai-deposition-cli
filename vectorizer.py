from pinecone import Pinecone
from llama_index.embeddings.together import TogetherEmbedding

class Vectorizer:
    def __init__(self, pinecone_api_key, pinecone_index_name, togetherai_api_key):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.embedder = TogetherEmbedding(
            model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key=togetherai_api_key
        )
        
    def vectorize_single_query(self, query):
        return self.embedder.get_text_embedding(query)

    def vectorize_docs(self, docs):
        # Extract text from the 'text' field of each transcript segment
        combined_docs = [' '.join([doc['text'] for doc in docs[i:i+3]]) for i in range(0, len(docs), 3)]
        print("IN VECTORIZE_DOCS", combined_docs)
        vectors = [self.embedder.get_text_embedding(doc) for doc in combined_docs]
        return vectors

    def upsert_docs(self, doc_ids, docs, metadata=None):
        vectors = self.vectorize_docs(docs)
        # Chunk metadata to match the combined docs
        if metadata is None:
            metadata = [{}] * len(vectors)
        else:
            # Ensure metadata is a list of dictionaries with the same length as doc_ids
            assert len(metadata) == len(doc_ids), "Metadata and doc_ids must have the same length"
            # Chunk metadata to match combined docs
            chunked_metadata = []
            for i in range(0, len(metadata), 3):
                chunk = {'transcript': ' '.join([meta['text'] for meta in metadata[i:i+3]])}
                chunked_metadata.append(chunk)
            metadata = chunked_metadata
        # Adjust doc_ids to match the combined docs
        adjusted_doc_ids = [doc_ids[i] for i in range(0, len(doc_ids), 3)]
        self.index.upsert(vectors=[(id, vector, meta) for id, vector, meta in zip(adjusted_doc_ids, vectors, metadata)])
        
    def just_upsert(self, doc_ids, docs, metadata=None):
        vectors = self.vectorize_docs(docs)
        self.index.upsert(vectors=[(id, vector, meta) for id, vector, meta in zip(doc_ids, vectors, metadata)])

    def query_vector(self, vector, top_k=2, filter=None):
        print(vector)
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_values=True,
            include_metadata=True,
        )