import cv2
import numpy as np
from typing import List, Dict


class SpeechIdentifier:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_embedder = cv2.dnn.readNetFromTorch("path/to/model.t7")
        self.face_ids = {}
        self.next_face_id = 0

    def identifySpeakers(
        self, segments: List[Dict[str, float]]
    ) -> List[Dict[str, List[int]]]:
        cap = cv2.VideoCapture(self.video_path)
        result = []

        for segment in segments:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(segment["start"] * cap.get(cv2.CAP_PROP_FPS)),
            )
            _, frame = cap.read()
            speakers = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for x, y, w, h in faces:
                face_roi = frame[y : y + h, x : x + w]
                face_blob = cv2.dnn.blobFromImage(
                    face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
                )
                self.face_embedder.setInput(face_blob)
                face_embedding = self.face_embedder.forward()
                face_embedding = face_embedding.flatten()

                min_distance = float("inf")
                closest_face_id = None
                for face_id, stored_embedding in self.face_ids.items():
                    distance = np.linalg.norm(stored_embedding - face_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        closest_face_id = face_id

                if closest_face_id is None:
                    closest_face_id = self.next_face_id
                    self.face_ids[closest_face_id] = face_embedding
                    self.next_face_id += 1

                speakers.append(closest_face_id)

            result.append(
                {"start": segment["start"], "end": segment["end"], "speakers": speakers}
            )

        return result


# Testing
if __name__ == "__main__":
    from testing import segments, sample_input_video_path

    identifier = SpeechIdentifier(sample_input_video_path)
    print(identifier.identifySpeakers(segments))
