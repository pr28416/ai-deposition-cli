from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    TextClip,
    ColorClip,
    CompositeVideoClip,
)
from whisper_func import VideoToTextModel


class VideoStitcher:
    def __init__(self, input_path, output_path, captions=False):
        self.input_path = input_path
        self.output_path = output_path
        self.captions = captions

    def stitch(self, segments):
        print(f"Stitching {self.input_path} to {self.output_path}")

        # Load the input video
        video = VideoFileClip(self.input_path)

        # Create a list to hold the clips
        clips = []

        # For each segment, cut the corresponding clip from the video and add it to the list
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            clip = video.subclip(start, end)
            clip.audio = video.audio.subclip(start, end)  # Set the audio of the clip

            # If captions are enabled, create a text clip and overlay it onto the video clip
            if self.captions and "text" in segment:
                # Split the text into words
                words = segment["text"].split()
                # Join the words back together with newlines inserted every 10 words
                caption_text = "\n".join(
                    " ".join(words[i : i + 10]) for i in range(0, len(words), 10)
                )
                text_clip = (
                    TextClip(caption_text, fontsize=24, color="white")
                    .set_position("bottom")
                    .set_duration(end - start)
                )

                # Create a black background clip and overlay the text clip onto it
                bg_clip = (
                    ColorClip((clip.size[0], text_clip.size[1]), col=[0, 0, 0])
                    .set_position(("center", "bottom"))
                    .set_duration(end - start)
                )
                caption_clip = CompositeVideoClip([bg_clip, text_clip])

                # Overlay the caption clip onto the video clip
                clip = CompositeVideoClip([clip, caption_clip])

            clips.append(clip)

        # Concatenate the clips and write the result to the output file
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(
            self.output_path, codec="libx264", audio_codec="aac"
        )  # Specify the audio codec


# Testing
if __name__ == "__main__":
    # Define the segments
    # segments = [
    #     {"start": 0, "end": 5},
    #     {"start": 10, "end": 15},
    #     {"start": 20, "end": 25},
    # ]
    videoModel = VideoToTextModel()
    segments = videoModel.sentence_transcribe("fridman_altman_podcast_sample.mp4")[::2]
    print(segments)

    # Create a VideoStitcher object
    stitcher = VideoStitcher("fridman_altman_podcast_sample.mp4", "output.mp4", True)

    # Stitch the video
    stitcher.stitch(segments)
