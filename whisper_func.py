import whisper
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path

# from whispercpp import Whisper


class VideoToTextModel:

    def __init__(self):
        self.model = whisper.load_model("small.en")

    def raw_transcribe(self, video_path):
        return self.model.transcribe(video_path)

    def sentence_transcribe(self, video_path, folder="data/"):
        transcription = self.raw_transcribe(video_path)["segments"]

        intervals = [[]]
        for idx, segment in enumerate(transcription):
            intervals[-1].append(idx)
            if any([segment["text"].strip().endswith(i) for i in [".", "!", "?"]]):
                intervals.append([])

        segments = []
        video_name = Path(video_path).stem

        for interval in intervals:
            if len(interval) == 0:
                continue

            start_time = transcription[interval[0]]["start"]
            end_time = transcription[interval[-1]]["end"]

            cropped_video_path = (
                f"{folder}{video_name}[{interval[0]}][{interval[-1]}].mp4"
            )
            ffmpeg_extract_subclip(
                video_path, start_time, end_time, targetname=cropped_video_path
            )

            segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": " ".join([transcription[i]["text"] for i in interval])
                    .strip()
                    .replace("  ", " "),
                    "video_path": cropped_video_path,
                }
            )
        return segments


# Testing
if __name__ == "__main__":
    model = VideoToTextModel()
    print(model.sentence_transcribe("fridman_altman_podcast_sample.mp4"))
