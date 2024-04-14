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



class DepositionVectorizer:
    def __init__(self, pinecone_index_name, pinecone_api_key, togetherai_api_key):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.embedder = TogetherEmbedding(
            model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key=togetherai_api_key
        )
        self.vector_store = PineconeVectorStore(pinecone_index=self.index)
        
    async def vectorize_and_upsert_pdf(self, pdf_path):
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20, include_metadata=True)
        nodes = node_parser.get_nodes_from_documents(documents)
        vectors = [{"id": node.node_id, "values": await self.vectorize_text(node.get_content()), "metadata": {**node.metadata, "text": node.get_content()}} for node in nodes]
        self.index.upsert(vectors=vectors)
        
    async def vectorize_and_upsert_video(self, video_path):
        video_to_text = VideoToTextModel()
        text = video_to_text.sentence_transcribe(video_path)
        vectors = [{"id": str(i), "values": await self.vectorize_text(seg["text"]), "metadata": {"start": seg["start"], "end": seg["end"], "video_path": video_path, "text": seg["text"]}} for i, seg in enumerate(text)]
        self.index.upsert(vectors=vectors)
      
    async def vectorize_text(self, text):
        return await self.embedder.aget_text_embedding(text)
    
    async def query_text(self, text, top_k=3, filter=None):
        query_vector = await self.vectorize_text(text)
        return self.index.query(vector=query_vector, top_k=top_k, include_values=False, include_metadata=True, filter=filter)

