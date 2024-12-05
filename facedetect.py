import cv2
import os
import numpy as np
from collections import deque

# Configurações do classificador de faces
classifier_folder = cv2.data.haarcascades
classifier_file = "haarcascade_frontalface_alt.xml"
face_detector = cv2.CascadeClassifier(os.path.join(classifier_folder, classifier_file))

# Configuração para média móvel
history_length = 5  # Número de frames para suavização
position_history = deque(maxlen=history_length)

def capturar_video(centro_callback, video_running):
    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture(0)

    while video_running.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Inverter a imagem para facilitar o controle
        frame = cv2.flip(frame, 1)

        # Detecta o rosto na imagem
        faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)  # Redução em minNeighbors

        # Processa a detecção de rosto
        if len(faces) > 0:
            # Seleciona o primeiro rosto detectado
            (x, y, w, h) = faces[0]
            centro = (x + w // 2, y + h // 2)
            position_history.append(centro)  # Adiciona ao histórico

            # Calcula a média das últimas posições
            avg_x = int(np.mean([p[0] for p in position_history]))
            avg_y = int(np.mean([p[1] for p in position_history]))
            centro_suavizado = (avg_x, avg_y)
            centro_callback(centro_suavizado)  # Envia o centro suavizado

            # Desenha um retângulo ao redor do rosto
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Desenha o ponto vermelho no centro do rosto
            cv2.circle(frame, centro_suavizado, 5, (0, 0, 255), -1)

        # Exibe a imagem com a detecção de rosto
        cv2.imshow("Deteccao de Rosto", frame)

        # Fecha a janela com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running.clear()
            break

    cap.release()
    cv2.destroyAllWindows()
