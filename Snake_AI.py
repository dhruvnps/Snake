import pygame
import neat
import os
import random
import math

WIDTH, HEIGHT = 5, 5
SCALE = 136
SPEED = math.inf
MAX_TIME = WIDTH * HEIGHT
GAP = 2

GREEN = (70, 215, 40)
PALE_GREEN = (20, 49, 14)
BLACK = (30, 30, 30)
RED = (220, 0, 27)
WHITE = (200, 200, 200)

WALL, SNAKE = 0, 1
UP, RIGHT, DOWN, LEFT = list(range(4))


class Snake:
    def __init__(self):
        self.timer = 0
        self.score = 0
        self.apple = Apple()
        self.pos = [(int(WIDTH / 2) - i, int(HEIGHT / 2)) for i in range(3)]
        self.direction = RIGHT

    def move(self):
        self.pos.insert(0, (next_unit(self.pos, self.direction)))

        if self.apple.collide(self):
            while self.apple.collide(self):
                self.apple = Apple()
            self.score += 1
            self.timer = 0
        else:
            self.pos.pop()

    def draw(self, win, color):
        if color == GREEN:
            self.apple.draw(win)
        for unit in self.pos:
            rect = (unit[0] * SCALE + GAP / 2, unit[1] * SCALE + GAP / 2, SCALE - GAP, SCALE - GAP)
            pygame.draw.rect(win, color, rect)


class Apple:
    def __init__(self):
        self.x = random.randint(0, WIDTH - 1)
        self.y = random.randint(0, HEIGHT - 1)

    def collide(self, snake):
        return (self.x, self.y) in snake.pos

    def draw(self, win):
        rect = (self.x * SCALE + GAP / 2, self.y * SCALE + GAP / 2, SCALE - GAP, SCALE - GAP)
        pygame.draw.rect(win, RED, rect)


def next_unit(pos, direction):
    if direction == UP:
        return pos[0][0], pos[0][1] - 1
    if direction == RIGHT:
        return pos[0][0] + 1, pos[0][1]
    if direction == DOWN:
        return pos[0][0], pos[0][1] + 1
    if direction == LEFT:
        return pos[0][0] - 1, pos[0][1]


def dead(pos):
    if not(0 <= pos[0][0] < WIDTH and 0 <= pos[0][1] < HEIGHT):
        return WALL
    if sorted(list(set(pos))) != sorted(pos):
        return SNAKE


def inputs(snake):
    inputs = []
    points = [None] * 4
    x, y = snake.pos[0][0], snake.pos[0][1]
    points[UP] = [(x, y - i) for i in range(y + 1)]
    points[RIGHT] = [(x + i, y) for i in range(WIDTH - x)]
    points[DOWN] = [(x, y + i) for i in range(HEIGHT - y)]
    points[LEFT] = [(x - i, y) for i in range(x + 1)]

    # adds fate of snake after one frame
    for direction_change in [-1, 0, 1]:
        direction = (snake.direction + direction_change) % 4
        next_snake_pos = snake.pos.copy()
        next_snake_pos.insert(0, (next_unit(next_snake_pos, direction)))
        next_snake_pos.pop()
        fate = dead(next_snake_pos)
        inputs.extend([fate == WALL, fate == SNAKE])

    # adds whether the apple is in given direction
    apple_view = [(snake.apple.x, snake.apple.y) in direction for direction in points]
    inputs.extend(apple_view[snake.direction:] + apple_view[:snake.direction])

    # adds whether part of the snake is in given direction
    self_view = [bool(set(snake.pos[1:]) & set(direction)) for direction in points]
    inputs.extend(self_view[snake.direction:] + self_view[:snake.direction])

    # adds distance from each of the 4 walls
    walls_view = [len(points[direction]) for direction in [UP, RIGHT, DOWN, LEFT]]
    inputs.extend(walls_view[snake.direction:] + walls_view[:snake.direction])

    return inputs


def main(genomes, config):
    active_nets = []
    active_genomes = []
    snakes = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        active_nets.append(net)
        snakes.append(Snake())
        genome.fitness = 0
        active_genomes.append(genome)

    win = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))
    clock = pygame.time.Clock()

    while True:
        clock.tick(SPEED)
        win.fill(BLACK)

        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                pygame.quit()
                quit()

        if len(snakes) > 0:
            best_snake = sorted(snakes, key=lambda x: x.score)[-1]

            for (x, snake) in enumerate(snakes):
                if dead(snake.pos) is not None or (snake.timer > MAX_TIME):
                    snakes.pop(x)
                    active_genomes.pop(x)
                    active_nets.pop(x)
                snake.draw(win, PALE_GREEN)

            if best_snake.score > 2:
                best_snake.draw(win, GREEN)

        # if all snakes are dead
        else:
            break

        for (x, snake) in enumerate(snakes):
            snake.move()
            snake.timer += 1
            active_genomes[x].fitness = snake.score

            outputs = active_nets[x].activate(inputs(snake))
            snake.direction = (snake.direction + (outputs.index(max(outputs)) - 1)) % 4

        pygame.display.update()


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    winner = p.run(main)

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
