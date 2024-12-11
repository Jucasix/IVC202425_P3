import cv2
import os
import numpy as np
from rich.progress import track

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
last_valid_position = None  # Armazena a última posição válida do ID 1
primary_id = 1  # ID principal para controle do paddle

def capturar_video(centro_callback, video_running):
    global last_valid_position, primary_id

    cap = cv2.VideoCapture(0)

    while video_running.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Detecta rostos na imagem
        faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)

        # Filtrar detecções muito pequenas ou muito grandes
        detections = []
        for (x, y, w, h) in faces:
            if 50 < w < 300 and 50 < h < 300:  # Limitar detecções por tamanho
                detections.append([x, y, x + w, y + h, 1])  # Score fixado em 1

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        # Atualiza o tracker
        tracked_objects = tracker.update(detections)
        # print(tracked_objects)

        # Ensure tracked_objects is always a 2D array
        if len(tracked_objects) == 1:
            # Only one object, so it's automatically the one with the smallest ID
            object_with_min_id = tracked_objects[0]
            min_id_index = tracked_objects[0][4]
            print("Only one object detected:", min_id_index)
        elif len(tracked_objects) > 1:
            # Multiple objects, find the smallest ID
            ids = tracked_objects[:, -1]  # Extract the last column (IDs)
            min_id_index = np.argmin(ids)  # Get the index of the smallest ID
            object_with_min_id = tracked_objects[min_id_index]  # Retrieve the object
            print("Object with the smallest ID:", min_id_index)

        # Processa os objetos rastreados
        id_found = False
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            centro = ((x1 + x2) // 2, (y1 + y2) // 2)

            if obj_id == min_id_index:
                id_found = True
                last_valid_position = centro  # Atualiza a última posição válida
                position_history.append(centro)  # Adiciona ao histórico
                avg_x = int(np.mean([p[0] for p in position_history]))
                avg_y = int(np.mean([p[1] for p in position_history]))
                centro_callback((avg_x, avg_y))  # Envia posição suavizada para o paddle

            # Desenha a bounding box e o ID na imagem
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Mantém a última posição válida mesmo se o ID principal não for detectado
        if not id_found and last_valid_position is not None:
            centro_callback(last_valid_position)

        # Exibe a imagem com o rastreamento
        cv2.imshow("Rastreamento com Viola-Jones e SORT", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running.clear()
            break

    cap.release()
    cv2.destroyAllWindows()
