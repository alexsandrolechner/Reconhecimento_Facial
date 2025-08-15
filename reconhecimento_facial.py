import os
from typing import List, Tuple

import cv2
import face_recognition
import numpy as np


def load_known_faces(image_dir: str, video_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """Carrega rostos conhecidos a partir de imagens e vídeos.

    Para as imagens, cada arquivo deve conter apenas um rosto. Para os vídeos,
    o primeiro rosto detectado é utilizado.
    """
    encodings: List[np.ndarray] = []
    names: List[str] = []

    # Imagens
    if os.path.isdir(image_dir):
        for filename in os.listdir(image_dir):
            filepath = os.path.join(image_dir, filename)
            try:
                image = face_recognition.load_image_file(filepath)
                faces = face_recognition.face_encodings(image)
                if faces:
                    encodings.append(faces[0])
                    names.append(os.path.splitext(filename)[0])
            except Exception as exc:  # pragma: no cover - logging omitted
                print(f"Erro ao processar {filepath}: {exc}")

    # Vídeos
    if os.path.isdir(video_dir):
        for filename in os.listdir(video_dir):
            filepath = os.path.join(video_dir, filename)
            capture = cv2.VideoCapture(filepath)
            frame_count = 0
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 5:
                    continue
                rgb_frame = frame[:, :, ::-1]
                faces = face_recognition.face_encodings(rgb_frame)
                if faces:
                    encodings.append(faces[0])
                    names.append(os.path.splitext(filename)[0])
                    break
            capture.release()

    return encodings, names


def recognize_from_webcam(known_encodings: List[np.ndarray], known_names: List[str]) -> None:
    """Inicializa a webcam e exibe a identificação dos rostos encontrados."""
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("Não foi possível acessar a webcam.")

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            detected_names = []
            for encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, encoding)
                name = "Desconhecido"
                face_distances = face_recognition.face_distance(known_encodings, encoding)
                if face_distances.size:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                detected_names.append(name)
            for (top, right, bottom, left), name in zip(face_locations, detected_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow("Reconhecimento Facial", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    image_dir = os.path.join(base_dir, "known_images")
    video_dir = os.path.join(base_dir, "known_videos")
    encodings, names = load_known_faces(image_dir, video_dir)
    recognize_from_webcam(encodings, names)


if __name__ == "__main__":
    main()
