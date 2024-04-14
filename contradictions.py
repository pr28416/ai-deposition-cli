from llama_index.llms.anthropic import Anthropic

class ContradictionFinder:
    def __init__(self, deposition_vectorizer, anthropic_api_key):
        self.deposition_vectorizer = deposition_vectorizer
        self.llm = Anthropic(api_key=anthropic_api_key, model="claude-3-opus-20240229")

    # def generate_contradictory_chunk(self, video_file_name):
    #     # Fetch all chunks for the given video file
    #     # query = f"video_path:{video_file_name}"
    #     # chunks = self.deposition_vectorizer.query_text("a", top_k=10, filter={"video_path": {"$eq": video_file_name}})
    #     # # print(chunks)
    #     # # Order chunks by their id
    #     # chunks = sorted(chunks["matches"], key=lambda x: int(x["id"]))
    #     # print(chunks)
    #     # Generate a contradictory statement for the first chunk
    #     # first_chunk_text = chunks[0]["metadata"]["text"]
    #     # print(first_chunk_text)
    #     contradictory_statement = self._generate_with_anthropic(first_chunk_text)
        
    #     print(contradictory_statement)
    #     return contradictory_statement

    def _generate_with_anthropic(self, text):
      return self.llm.complete(f"Generate a contradictory statement to: {text}\n\nNo need to think very hard, and don't say anything else other than the contradictoy statement. Whatever you have to do, generate a contradiction.")

    def find_contradictions(self, video_file_name):
        chunks = self.deposition_vectorizer.query_text("a", top_k=10, filter={"video_path": {"$eq": video_file_name}})
        chunks = sorted(chunks["matches"], key=lambda x: int(x["id"]))
        
        for i, chunk in enumerate(chunks):
            print(chunk["metadata"]["text"])
            contradictory_statement = self._generate_with_anthropic(chunk["metadata"]["text"])
            print(f"Contradictory statement for chunk {i}: {contradictory_statement.text}")
            
            res = self.deposition_vectorizer.query_text(contradictory_statement.text, top_k=1, filter={"video_path": {"$eq": video_file_name}})
            print(res["matches"])
            print("====")
