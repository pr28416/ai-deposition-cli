from pinecone import Pinecone
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.readers.file import PDFReader
from pathlib import Path
from llama_index.core import Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import NodeParser
from whisper_func import VideoToTextModel
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
import asyncio

class DepositionSearcher:
    def __init__(self, pinecone_index_name, pinecone_api_key, togetherai_api_key):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.embedder = TogetherEmbedding(
            model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key=togetherai_api_key
        )
        self.vector_store = PineconeVectorStore(pinecone_index=self.index)
    
    def vectorize_text(self, text):
        return self.embedder.get_text_embedding(text)

    def query_text(self, text, top_k=3, filter=None):
        query_vector = self.vectorize_text(text)
        return self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            filter=filter,
        )

    def search_all(self, text):
        res = self.vectorize_text(text)
        # retrieve from Pinecone
        xq = res['data'][0]['embedding']

        # get relevant contexts (including the questions)
        res = self.index.query(vector=xq, top_k=5, include_metadata=True)
        return res 
