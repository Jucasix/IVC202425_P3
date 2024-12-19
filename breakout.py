import pygame
from pygame.locals import *
import cv2
from trackfacedetect import capturar_video  # Importa a função de detecção de rosto
import numpy as np
import threading

pygame.init()

# Configuração da tela do jogo
screen_width = 600
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Breakout - Controle por Rosto')

# Definição de cores
bg = (234, 218, 184)
block_red = (242, 85, 96)
block_green = (86, 174, 87)
block_blue = (69, 177, 232)
paddle_col = (142, 135, 123)
paddle_outline = (100, 100, 100)
text_col = (78, 81, 139)

# Configurações do jogo
cols = 6
rows = 6
clock = pygame.time.Clock()
fps = 60
live_ball = False
game_over = 0

# Variável para armazenar a posição normalizada do rosto
centro_normalizado = 0.5
video_running = threading.Event()  # Flag para controlar a thread de captura de vídeo
video_running.set()  # Inicializa como ativo

# Callback para atualizar a posição do rosto
def atualizar_centro(centro):
    global centro_normalizado
    centro_normalizado = centro

# Função para exibir as instruções iniciais
def readme_window(window_name="README", width=800, height=300, font_scale=0.6, color=(255, 255, 255), thickness=1):
    image = np.zeros((height, width, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    text1 = "Usar a cabeca para jogar, movendo-a de um lado para o outro."
    text2 = "Deve jogar com o rosto virado de frente para a camara."
    text3 = "Deve estar entre 60 a 80 cm da camara."

    text_x = 25
    text_y1 = 100
    text_y2 = 150
    text_y3 = 200

    cv2.putText(image, text1, (text_x, text_y1), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    cv2.putText(image, text2, (text_x, text_y2), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    cv2.putText(image, text3, (text_x, text_y3), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    cv2.imshow(window_name, image)

# Classe Paddle
class Paddle:
    def __init__(self):
        self.reset()

    def move_to_position(self, centro_normalizado):
        # Mapeia a posição normalizada para a largura da tela
        self.x = int(centro_normalizado * (screen_width - self.width))
        self.rect.x = self.x

        # Verifica os limites da tela
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > screen_width:
            self.rect.right = screen_width

    def draw(self):
        pygame.draw.rect(screen, paddle_col, self.rect)
        pygame.draw.rect(screen, paddle_outline, self.rect, 3)

    def reset(self):
        self.height = 20
        self.width = int(screen_width / cols)
        self.x = int((screen_width / 2) - (self.width / 2))  # Inicializa x
        self.y = screen_height - (self.height * 2)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

# Classe Wall
class Wall:
    def __init__(self):
        self.width = screen_width // cols
        self.height = 50
        self.blocks = []
        self.create_wall()

    def create_wall(self):
        self.blocks = []
        for row in range(rows):
            block_row = []
            for col in range(cols):
                block_x = col * self.width
                block_y = row * self.height
                rect = pygame.Rect(block_x, block_y, self.width, self.height)
                if row < 2:
                    strength = 3
                elif row < 4:
                    strength = 2
                else:
                    strength = 1
                block_row.append([rect, strength])
            self.blocks.append(block_row)

    def draw_wall(self):
        for row in self.blocks:
            for block in row:
                if block[1] > 0:
                    if block[1] == 3:
                        block_col = block_blue
                    elif block[1] == 2:
                        block_col = block_green
                    else:
                        block_col = block_red
                    pygame.draw.rect(screen, block_col, block[0])
                    pygame.draw.rect(screen, bg, block[0], 2)

# Classe Ball
class Ball:
    def __init__(self, x, y):
        self.reset(x, y)

    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        if self.rect.left < 0 or self.rect.right > screen_width:
            self.speed_x *= -1
        if self.rect.top < 0:
            self.speed_y *= -1
        if self.rect.bottom > screen_height:
            self.game_over = -1

        if self.rect.colliderect(player_paddle.rect):
            self.speed_y *= -1

        for row in wall.blocks:
            for block in row:
                if self.rect.colliderect(block[0]):
                    self.speed_y *= -1
                    if block[1] > 1:
                        block[1] -= 1
                    else:
                        block[0] = pygame.Rect(0, 0, 0, 0)
                    break

        return self.game_over

    def draw(self):
        pygame.draw.circle(screen, paddle_col, (self.rect.x + self.ball_rad, self.rect.y + self.ball_rad), self.ball_rad)
        pygame.draw.circle(screen, paddle_outline, (self.rect.x + self.ball_rad, self.rect.y + self.ball_rad), self.ball_rad, 3)

    def reset(self, x, y):
        self.ball_rad = 10
        self.x = x - self.ball_rad
        self.y = y
        self.rect = pygame.Rect(self.x, self.y, self.ball_rad * 2, self.ball_rad * 2)
        self.speed_x = 4
        self.speed_y = -4
        self.speed_max = 5
        self.game_over = 0

wall = Wall()
player_paddle = Paddle()
ball = Ball(player_paddle.x + (player_paddle.width // 2), player_paddle.y - player_paddle.height)

readme_window()

thread_video = threading.Thread(target=capturar_video, args=(atualizar_centro, video_running, screen_width))
thread_video.start()

run = True
while run:
    clock.tick(fps)
    screen.fill(bg)

    if centro_normalizado is not None:
        player_paddle.move_to_position(centro_normalizado)

    wall.draw_wall()
    player_paddle.draw()
    ball.draw()

    if live_ball:
        game_over = ball.move()
        if game_over != 0:
            live_ball = False

    if not live_ball:
        if game_over == 0:
            text = 'CLICK ANYWHERE TO START'
        elif game_over == 1:
            text = 'YOU WON! CLICK ANYWHERE TO START'
        elif game_over == -1:
            text = 'YOU LOST! CLICK ANYWHERE TO START'
        font = pygame.font.SysFont('Constantia', 30)
        img = font.render(text, True, text_col)
        screen.blit(img, (100, screen_height // 2 + 100))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.MOUSEBUTTONDOWN and not live_ball:
            live_ball = True
            ball.reset(player_paddle.x + (player_paddle.width // 2), player_paddle.y - player_paddle.height)
            player_paddle.reset()

    pygame.display.update()

video_running.clear()
thread_video.join()
cv2.destroyAllWindows()
pygame.quit()
