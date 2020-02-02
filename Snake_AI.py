import pygame
import neat
import os
import random

WIDTH, HEIGHT = 40, 40
SCALE = 16
SPEED = 10
MAX_TIME = 200

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

GREEN = (70, 215, 40)
BLACK = (30, 30, 30)
RED = (220, 0, 27)
WHITE = (200, 200, 200)


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

    def draw(self, win):
        self.apple.draw(win)
        for unit in self.pos:
            pygame.draw.rect(win, GREEN, (unit[0] * SCALE, unit[1] * SCALE, SCALE, SCALE))


class Apple:
    def __init__(self):
        self.x = random.randint(0, WIDTH - 1)
        self.y = random.randint(0, HEIGHT - 1)

    def collide(self, snake):
        if self.x == snake.pos[0][0] and self.y == snake.pos[0][1]:
            return True

    def draw(self, win):
        pygame.draw.rect(win, RED, (self.x * SCALE, self.y * SCALE, SCALE, SCALE))


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
        return True
    # checks if snake has overlapped onto itself
    if sorted(list(set(pos))) != sorted(pos):
        return True


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
        #clock.tick(SPEED)
        win.fill(BLACK)

        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                pygame.quit()
                quit()

        if len(snakes) > 0:
            for (x, snake) in enumerate(snakes):
                if dead(snake.pos) or snake.timer > MAX_TIME:
                    snakes.pop(x)
                    active_genomes.pop(x)
                    active_nets.pop(x)
        else:
            break

        for (x, snake) in enumerate(snakes):
            snake.draw(win)
            snake.move()

            active_genomes[x].fitness = snake.score
            snake.timer += 1

            inputs = []
            for i in range(-1, 2):
                direction = (snake.direction + i) % 4
                next_snake_pos = snake.pos.copy()
                next_snake_pos.insert(0, (next_unit(next_snake_pos, direction)))
                next_snake_pos.pop()
                if dead(next_snake_pos):
                    inputs.append(0)
                else:
                    inputs.append(1)

            apple_view = []
            if snake.pos[0][0] == snake.apple.x and snake.pos[0][1] > snake.apple.y:
                apple_view.append(1)
            if snake.pos[0][1] == snake.apple.y and snake.pos[0][0] < snake.apple.y:
                apple_view.append(2)
            if snake.pos[0][0] == snake.apple.x and snake.pos[0][1] < snake.apple.x:
                apple_view.append(3)
            if snake.pos[0][1] == snake.apple.y and snake.pos[0][0] > snake.apple.x:
                apple_view.append(4)
            apple_view = [(i in apple_view) for i in range(4)]

            if direction == UP:
                inputs.extend(apple_view)
            if direction == RIGHT:
                inputs.extend(apple_view[1:] + apple_view[:1])
            if direction == DOWN:
                inputs.extend(apple_view[2:] + apple_view[:2])
            if direction == LEFT:
                inputs.extend(apple_view[3:] + apple_view[:3])

            outputs = active_nets[x].activate(inputs)
            snake.direction = (snake.direction + (outputs.index(max(outputs)) - 1)) % 4

        pygame.display.update()


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    winner = p.run(main, 50)

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
