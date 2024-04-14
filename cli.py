import os
import openai
import instructor
from pydantic import BaseModel
from dotenv import load_dotenv
from whisper_func import VideoToTextModel
import sys
from deposition_vectorizer import DepositionVectorizer

load_dotenv()

client = instructor.from_openai(openai.OpenAI())


class VideoSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    video_path: str


class PDFSegment(BaseModel):
    id: int
    page: int
    file_path: str


class DataManager:
    videoSegments: list[VideoSegment]
    pdfs: list[str]


vectorizer = DepositionVectorizer(
    os.getenv("PINECONE_INDEX_NAME"),
    os.getenv("PINECONE_API_KEY"),
    os.getenv("TOGETHERAI_API_KEY"),
)
dataManager = DataManager()


class Response(BaseModel):
    content: str


class AIContainer:
    def __init__(self, system_prompt):
        self.client = instructor.from_openai(openai.OpenAI())
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def ask(self, query, model="gpt-4-turbo-preview", response_model=Response):
        self.messages.append({"role": "user", "content": query})
        response = self.client.chat.completions.create(
            model=model, messages=self.messages, response_model=response_model
        )
        self.messages.append({"role": "ai", "content": str(response)})
        return response


def run_ai():

    ai = AIContainer("You are a legal assistant.")

    user_input = input(">>> ")
    if user_input == "exit":
        return False

    response = ai.ask(user_input)
    print(response.content)

    return True


def run():
    print("=======\nOptions:\n=======")
    print("\t(p) Upload document - PDF")
    print("\t(v) Upload video - MP4")
    print("\t(t) Talk to Depose AI")
    print("\t(e) Exit")
    user_input = input("Enter your command: ")
    if user_input == "p":
        print("Uploading PDF...")
    elif user_input == "v":
        print("Uploading video...")
    elif user_input == "t":
        while run_ai():
            pass
    elif user_input == "e":
        return False
    else:
        print("Invalid command. Please try again.")
    return True


def process_files_in_data():
    data_dir = "data"
    files = [
        file
        for file in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, file))
    ]
    if not files:
        print("No files in data directory.")
        return False

    videoModel = VideoToTextModel()
    for file in files:
        print(" - Processing", file)

        if file.endswith(".mp4"):
            print(" - Video file detected")

            vectorizer.vectorize_and_upsert_video(f"{data_dir}/{file}")

            # segments = videoModel.sentence_transcribe(f"{data_dir}/{file}")
            # video_path = f"{data_dir}/{file}"
            # for id, segment in enumerate(segments):
            #     start = segment["start"]
            #     end = segment["end"]
            #     text = segment["text"]
            #     video_segment = VideoSegment(
            #         id=id, start=start, end=end, text=text, video_path=video_path
            #     )
            #     dataManager.videoSegments.append(video_segment)

    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if not process_files_in_data():
            exit(1)

    print("====================================")
    print("           Depose AI CLI")
    print("====================================")
    print()

    while run_ai():
        pass
