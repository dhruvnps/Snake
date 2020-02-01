import pygame
import neat
import os
import random

WIDTH, HEIGHT = 15, 15
SCALE = 42
SPEED = 5

UP, RIGHT, DOWN, LEFT = 1, 2, 3, 4

GREEN = (70, 215, 40)
BLACK = (20, 20, 20)
RED = (220, 0, 27)
WHITE = (200, 200, 200)


class Snake:
    def __init__(self):
        self.score = 0
        self.apple = Apple()
        self.position = [(int(WIDTH / 2) - i, int(HEIGHT / 2)) for i in range(3)]
        self.direction = RIGHT

    def move(self):
        if self.direction == UP:
            self.position.insert(0, (self.position[0][0], self.position[0][1] - 1))
        if self.direction == RIGHT:
            self.position.insert(0, (self.position[0][0] + 1, self.position[0][1]))
        if self.direction == DOWN:
            self.position.insert(0, (self.position[0][0], self.position[0][1] + 1))
        if self.direction == LEFT:
            self.position.insert(0, (self.position[0][0] - 1, self.position[0][1]))

        if self.apple.collide(self):
            while self.apple.collide(self):
                self.apple = Apple()
            self.score += 1
            print(self.score)
        else:
            self.position.pop()

    def collide(self):
        if not(0 <= self.position[0][0] < WIDTH and 0 <= self.position[0][1] < HEIGHT):
            return True
        # checks if snake has overlapped onto itself
        if sorted(list(set(self.position))) != sorted(self.position):
            return True

    def draw(self, win):
        self.apple.draw(win)
        for unit in self.position:
            pygame.draw.rect(win, GREEN, (unit[0] * SCALE, unit[1] * SCALE, SCALE, SCALE))


class Apple:
    def __init__(self):
        self.x = random.randint(0, WIDTH - 1)
        self.y = random.randint(0, HEIGHT - 1)

    def collide(self, snake):
        if self.x == snake.position[0][0] and self.y == snake.position[0][1]:
            return True

    def draw(self, win):
        pygame.draw.rect(win, RED, (self.x * SCALE, self.y * SCALE, SCALE, SCALE))


def main():
    win = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))
    clock = pygame.time.Clock()
    snake = Snake()

    while True:
        clock.tick(SPEED)
        win.fill(BLACK)
        snake.move()
        snake.draw(win)

        if snake.collide():
            break

        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                pygame.quit()
                quit()

            # to control snake direction
            if i.type == pygame.KEYDOWN:
                if i.key == pygame.K_UP:
                    snake.direction = UP
                if i.key == pygame.K_RIGHT:
                    snake.direction = RIGHT
                if i.key == pygame.K_DOWN:
                    snake.direction = DOWN
                if i.key == pygame.K_LEFT:
                    snake.direction = LEFT

        pygame.display.update()


if __name__ == "__main__":
    main()
