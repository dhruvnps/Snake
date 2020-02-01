import pygame
import neat
import os
import random

WIDTH, HEIGHT = 15, 15
SCALE = 42
SPEED = 5

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

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
            #print(self.score)
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
            for snake in snakes:
                if snake.collide():
                    snakes.remove(snake)
        else:
            break

        for (x, snake) in enumerate(snakes):
            snake.draw(win)
            snake.move()

            #active_genomes[x].fitness = snake.score
            active_genomes[x].fitness += 0.1
            outputs = active_nets[x].activate((snake.position[0][0] - WIDTH,
                                               snake.position[0][1] - HEIGHT,
                                               snake.position[0][0],
                                               snake.position[0][1],
                                               snake.direction == UP,
                                               snake.direction == RIGHT,
                                               snake.direction == DOWN,
                                               snake.direction == LEFT))

            snake.direction = outputs.index(max(outputs))

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
