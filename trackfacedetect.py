import cv2
import os
import numpy as np
from sort import *  # Importa o tracker SORT
from collections import deque

# Configurações do classificador de faces
classifier_folder = cv2.data.haarcascades
classifier_file = "haarcascade_frontalface_alt.xml"
face_detector = cv2.CascadeClassifier(os.path.join(classifier_folder, classifier_file))

# Configuração para média móvel
history_length = 5
position_history = deque(maxlen=history_length)

# Inicializa o SORT com parâmetros ajustados
tracker = Sort(max_age=100, min_hits=2, iou_threshold=0.5)
last_valid_position = None  # Armazena a última posição válida
primary_id = None  # ID dinâmico que controla o paddle

def capturar_video(centro_callback, video_running, screen_width):
    global last_valid_position, primary_id

    cap = cv2.VideoCapture(0)

    while video_running.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Detecta rostos na imagem
        faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)

        detections = []
        for (x, y, w, h) in faces:
            detections.append([x, y, x + w, y + h, 1])  # Score fixado em 1

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        # Atualiza o tracker
        tracked_objects = tracker.update(detections)

        # Determina o menor ID disponível
        min_id = None
        min_id_position = None

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            centro_x = (x1 + x2) // 2

            if min_id is None or obj_id < min_id:
                min_id = obj_id
                min_id_position = centro_x

        # Atualiza o ID principal e o controle do paddle
        if min_id_position is not None:
            centro_normalizado = min_id_position / screen_width  # Normaliza para [0, 1]
            last_valid_position = centro_normalizado  # Atualiza a última posição válida
            centro_callback(centro_normalizado)  # Envia a posição normalizada
            primary_id = min_id  # Atualiza o ID principal

        # Se nenhum ID for encontrado, mantém a última posição válida
        if min_id is None and last_valid_position is not None:
            centro_callback(last_valid_position)

        # Desenha bounding boxes e IDs na janela
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Rastreamento com Viola-Jones e SORT", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running.clear()
            break

    cap.release()
    cv2.destroyAllWindows()
