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
tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)  # Aumenta a resiliência do rastreamento
last_valid_position = None  # Armazena a última posição válida do ID 1

def capturar_video(centro_callback, video_running):
    global last_valid_position

    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture(0)

    while video_running.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Inverter a imagem para facilitar o controle
        frame = cv2.flip(frame, 1)

        # Detecta o rosto na imagem
        faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)

        # Converte as detecções para o formato esperado pelo SORT: [x1, y1, x2, y2, score]
        detections = []
        for (x, y, w, h) in faces:
            detections.append([x, y, x + w, y + h, 1])  # A confiança (score) é fixada como 1

        # Garante que `detections` seja uma matriz vazia no formato correto se não houver detecções
        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        # Atualiza o tracker e recebe os IDs atribuídos
        tracked_objects = tracker.update(detections)

        # Processa os objetos rastreados
        id_found = False
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            centro = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Apenas o ID 1 controla o paddle
            if obj_id == 1:
                id_found = True
                last_valid_position = centro  # Atualiza a última posição válida
                centro_callback(centro)  # Envia a posição do ID 1

            # Desenha a bounding box e o ID na imagem
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, centro, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Se o ID 1 não for encontrado, mantém a última posição válida
        if not id_found and last_valid_position is not None:
            centro_callback(last_valid_position)

        # Exibe a imagem com a detecção e rastreamento
        cv2.imshow("Deteccao de Rosto com Rastreamento", frame)

        # Fecha a janela com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running.clear()
            break

    cap.release()
    cv2.destroyAllWindows()
