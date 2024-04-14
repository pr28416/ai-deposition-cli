from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext, Document
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from vectorizer import Vectorizer
from whisper_func import VideoToTextModel
from llama_index.llms.together import TogetherLLM
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize document store
docstore = SimpleDocumentStore()

# Load documents into the document store
# Assuming 'docs' is a list of documents obtained from the VideoToTextModel
video_to_text_model = VideoToTextModel()
video_path = "fridman_altman_podcast_sample.mp4"  # Specify your video path here
docs = video_to_text_model.sentence_transcribe(video_path)
print(docs)

# Convert transcribed text to Document model format for node parsing
document_models = [Document(text=doc['text']) for doc in docs]

# Parse chunk hierarchy from text, load into storage
node_parser = HierarchicalNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(document_models)
leaf_nodes = get_leaf_nodes(nodes)

# Convert each document into a format suitable for the document store
docstore.add_documents(nodes)

# Define storage context
storage_context = StorageContext.from_defaults(docstore=docstore)

# Initialize the vectorizer
vectorizer = Vectorizer(pinecone_api_key=os.getenv("PINECONE_API_KEY"), 
                        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"), 
                        togetherai_api_key=os.getenv("TOGETHERAI_API_KEY"))

# Vectorize original documents (not Document models) using the correct function and upsert into Pinecone (Vector Store)
# Adjust doc_ids to match the individual documents
doc_ids = [str(i) for i in range(len(docs))]  
docs2 = [{'text': doc['text']} for doc in docs]
metadata = [{'text': doc['text'], 'start_time': doc['start'], 'end_time': doc['end']} for doc in docs]

# Adjust the call to just_upsert to pass the individual text documents instead of the combined docs
print("METADATA", metadata)
print("DOC_IDS", doc_ids)

vectorizer.upsert_docs(doc_ids, docs2, metadata=metadata)
# Load index into vector index and define retriever
from llama_index.core import VectorStoreIndex
for leaf in leaf_nodes:
    print(leaf.get_text())

base_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context, embed_model=vectorizer.embedder)
base_retriever = base_index.as_retriever(similarity_top_k=2)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

# llm = TogetherLLM(
#     model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.getenv("TOGETHERAI_API_KEY")
# )

# Instead of using RetrieverQueryEngine, directly return the documents found by the merger
query_str = "How does GPT-7 relate to proofs?"
merged_nodes = retriever.retrieve(query_str)
for node in merged_nodes:
    print(node.get_text())

query_vector = vectorizer.vectorize_single_query(query_str)

# Ensure the query_vector is correctly formatted before querying
if query_vector:
    query_vector = query_vector
    
    res = vectorizer.query_vector(query_vector)
    print(res["matches"][0]["metadata"])
else:
    print("Failed to generate query vector.")
