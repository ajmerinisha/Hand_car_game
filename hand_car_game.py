import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import sys
import time

pygame.init()

# ----------------- GAME WINDOW -----------------
SCREEN_W, SCREEN_H = 600, 800
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Hand Gesture Car Game")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont(None, 36)

# ----------------- COLORS -----------------
WHITE = (255,255,255)
GRAY = (45,45,45)
ROAD = (60,60,60)
RED = (230,40,40)
BLUE = (60,140,255)
YELLOW = (240,210,0)

# ----------------- ROAD LIMITS -----------------
LEFT_MARGIN = 80
RIGHT_MARGIN = SCREEN_W - 80
ROAD_W = RIGHT_MARGIN - LEFT_MARGIN

# ----------------- CAR SETTINGS -----------------
CAR_W, CAR_H = 60, 100
car_x = SCREEN_W//2 - CAR_W//2
car_y = SCREEN_H - CAR_H - 40
car_speed = 250  # dynamic speed

# ----------------- OBSTACLE -----------------
class Obstacle:
    def __init__(self, x, w, h, speed):
        self.x = x
        self.y = -h
        self.w = w
        self.h = h
        self.speed = speed

    def update(self, dt):
        self.y += self.speed * dt

    def rect(self):
        return pygame.Rect(self.x, self.y, self.w, self.h)

def spawn_obstacle():
    w = random.randint(50, 120)
    h = random.randint(40, 80)
    x = random.randint(LEFT_MARGIN, RIGHT_MARGIN - w)
    speed = random.randint(250, 420)
    return Obstacle(x, w, h, speed)

# ----------------- MEDIAPIPE -----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
cap = cv2.VideoCapture(0)

def get_gesture(frame):
    """Returns hand x-position + gesture type (open / fist)."""
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None, None

    hand = result.multi_hand_landmarks[0]

    # Average X position of hand
    x_list = [lm.x for lm in hand.landmark]
    avg_x = np.mean(x_list)

    # Finger state detection (simple)
    tips = [4, 8, 12, 16, 20]
    open_fingers = 0

    for tip_id in tips[1:]:  # ignore thumb
        if hand.landmark[tip_id].y < hand.landmark[tip_id-2].y:
            open_fingers += 1

    if open_fingers >= 3:
        gesture = "open"      # accelerate
    else:
        gesture = "fist"      # brake

    return avg_x, gesture


# ----------------- GAME LOOP -----------------
def game_loop():
    global car_x, car_speed
    obstacles = []
    spawn_timer = 0
    score = 0
    game_over = False
    last_time = time.time()

    running = True
    while running:
        dt = time.time() - last_time
        last_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running=False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running=False
                if event.key == pygame.K_r and game_over:
                    return game_loop()

        ret, frame = cap.read()
        if not ret:
            break

        hand_x, gesture = get_gesture(frame)

        if hand_x is not None:
            # Map hand X → car X
            mapped = LEFT_MARGIN + (hand_x * ROAD_W)
            car_x = car_x + (mapped - car_x) * 0.25

            # Gesture control for speed
            if gesture == "open":
                car_speed = min(car_speed + 250*dt, 600)  # accelerate
            elif gesture == "fist":
                car_speed = max(car_speed - 400*dt, 150)  # brake

        # Spawn obstacles
        spawn_timer += dt
        if spawn_timer > 1.1:
            spawn_timer = 0
            obstacles.append(spawn_obstacle())

        # Update obstacles
        for ob in obstacles:
            ob.update(dt)

        # Remove old obstacles
        obstacles = [o for o in obstacles if o.y < SCREEN_H+100]

        # Collision check
        car_rect = pygame.Rect(int(car_x), int(car_y), CAR_W, CAR_H)
        for o in obstacles:
            if o.rect().colliderect(car_rect):
                game_over = True

        # Draw everything
        screen.fill(GRAY)

        # Road
        pygame.draw.rect(screen, ROAD, (LEFT_MARGIN, 0, ROAD_W, SCREEN_H))

        # Car
        pygame.draw.rect(screen, BLUE, car_rect, border_radius=10)

        # Obstacles
        for o in obstacles:
            pygame.draw.rect(screen, RED, o.rect(), border_radius=8)

        # Score
        score += dt * (car_speed / 50)
        score_text = FONT.render(f"Score: {int(score)}", True, WHITE)
        speed_text = FONT.render(f"Speed: {int(car_speed)}", True, WHITE)
        screen.blit(score_text, (10,10))
        screen.blit(speed_text, (10,45))

        # Game over
        if game_over:
            msg = FONT.render("GAME OVER - Press R to Restart", True, WHITE)
            screen.blit(msg, (SCREEN_W//2 - msg.get_width()//2, SCREEN_H//2))
            pygame.display.flip()
            continue

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()


game_loop()
