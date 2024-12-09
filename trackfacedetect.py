import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from ultralytics import YOLO
from sort import *

# Inicializa o modelo YOLO
model = YOLO("yolov8n.pt")  # Use um modelo YOLOv8 leve
face_class_id = 0  # Classe "person", geralmente usada para rostos

# Inicializa o rastreador SORT
tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

# Configurações globais
last_valid_position = None  # Última posição válida do ID principal
primary_id = 1  # ID principal que controla o paddle

def capturar_video(centro_callback, video_running):
    global last_valid_position, primary_id

    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture(0)

    while video_running.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Inverte a imagem para facilitar o controle
        frame = cv2.flip(frame, 1)

        # Converte para RGB para o YOLO
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Faz a detecção com YOLO
        results = model(image_rgb, verbose=False)
        detections = []
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, class_id = result.tolist()
            if int(class_id) == face_class_id and conf > 0.5:  # Apenas classe "person" com confiança alta
                detections.append([x1, y1, x2, y2, conf])

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        # Atualiza o rastreador SORT
        tracked_objects = tracker.update(detections)

        # Processa os objetos rastreados
        id_found = False
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            centro = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Apenas o ID principal controla o paddle
            if obj_id == primary_id:
                id_found = True
                last_valid_position = centro  # Atualiza a última posição válida
                centro_callback(centro)  # Envia a posição para o controle do paddle

            # Desenha a bounding box e o ID na imagem
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, centro, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Se o ID principal não for encontrado, mantém a última posição válida
        if not id_found and last_valid_position is not None:
            centro_callback(last_valid_position)

        # Exibe a imagem com detecção e rastreamento
        cv2.imshow("Rastreamento com YOLO e SORT", frame)

        # Fecha a janela com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running.clear()
            break

    cap.release()
    cv2.destroyAllWindows()
