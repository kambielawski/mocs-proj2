from enum import Enum
import math
import random
from time import perf_counter

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

WORLD_SIZE = 128
MIDDLE = WORLD_SIZE // 2
ITERATIONS = 1000
RAYLEIGH_SCALE = 2.0

# 0: not visited
# 1: initial seed
# 2 .. ITERATIONS: visited by random walk
COLORS = ITERATIONS + 2


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def random_walk(dla, value, levy=False, gravity=False):
    row, col = np.random.randint(0, WORLD_SIZE, 2)
    while True:
        if gravity:
            # Bias movement towards the origin.
            weights = [
                2 + (row > MIDDLE) * 1, # UP
                2 + (row < MIDDLE) * 1, # DOWN
                2 + (col > MIDDLE) * 1, # LEFT
                2 + (col < MIDDLE) * 1  # RIGHT
            ]
        else:
            weights = [1, 1, 1, 1]
        direction = random.choices(list(Direction), weights, k=1)[0]
        if levy:
            # Randomly choose a value from a Rayleigh distribution, rounding up
            # to the nearest int. This produces a value that is at least one
            # and rarely much bigger.
            distance = math.ceil(np.random.rayleigh(scale=RAYLEIGH_SCALE))
        else:
            distance = 1
        for _ in range(distance):
            match direction:
                case Direction.UP:    row = max(row - 1, 1)
                case Direction.DOWN:  row = min(row + 1, WORLD_SIZE - 2)
                case Direction.LEFT:  col = max(col - 1, 1)
                case Direction.RIGHT: col = min(col + 1, WORLD_SIZE - 2)
            if np.count_nonzero(dla[row-1:row+2, col-1:col+2]) > 0:
                dla[row][col] = value
                return


def compute_dimensionality(dla):
    # TODO: try a bunch of different box lengths.
    box_length = 4

    # Finding all the non-zero pixels
    pixels = []
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            if dla[i, j] > 0:
                pixels.append((i, j))
    pixels = np.array(pixels)
    bins = np.arange(0, WORLD_SIZE, box_length)
    # Count number of boxes that contain at least 1 non-zero pixel
    # TODO: This should be equivalent to what was here before, but I'm getting
    # an error about dimensions not matching, so something's not right.
    H, _ = np.histogramdd(dla > 0, bins=(bins, bins))
    N = np.sum(H > 0)

    # Calculate the fractal dimension where N = box_count
    dimensions = np.log(N) / np.log(box_length)
    return dimensions


def initialize():
    dla = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint16)
    dla[MIDDLE][MIDDLE] = 1
    return dla


def save_image(dla, filename):
    # Setup a color map for this image. Start with the viridis color scheme,
    # scale to fit our data, and make sure that cells with value 0 are black.
    viridis = plt.colormaps['viridis'].resampled(COLORS)
    newcolors = viridis(np.linspace(0, 1, COLORS))
    newcolors[0] = np.array([0.0, 0.0, 0.0, 1.0]) # black

    # Render and save the image.
    plt.imshow(dla, cmap=colors.ListedColormap(newcolors))
    # plt.show()
    plt.savefig(filename)
    plt.close()


def run_trial(levy=False, gravity=False):
    start_time = perf_counter()
    dla = initialize()
    for i in range(ITERATIONS):
        random_walk(dla, i + 2, levy, gravity)
    save_image(dla, 'dla_0_0.png')
    elapsed = perf_counter() - start_time
    dimensionality = compute_dimensionality(dla)
    return elapsed, dimensionality


def main():
    print('Running unoptimized DLA...')
    elapsed, dimensionality = run_trial()
    print(f'Completed {ITERATIONS} iterations in {elapsed:0.2f} seconds.')
    print(f'Result has fractal dimension {dimensionality:0.2f}\n')

    print('Running DLA with Lévy flight optimization...')
    elapsed, dimensionality = run_trial(levy=True)
    print(f'Completed {ITERATIONS} iterations in {elapsed:0.2f} seconds.')
    print(f'Result has fractal dimension {dimensionality:0.2f}\n')

    print('Running DLA with gravity optimization...')
    elapsed, dimensionality = run_trial(gravity=True)
    print(f'Completed {ITERATIONS} iterations in {elapsed:0.2f} seconds.')
    print(f'Result has fractal dimension {dimensionality:0.2f}\n')

    print('Running DLA with both optimizations...')
    elapsed, dimensionality = run_trial(levy=True, gravity=True)
    print(f'Completed {ITERATIONS} iterations in {elapsed:0.2f} seconds.')
    print(f'Result has fractal dimension {dimensionality:0.2f}')



if __name__ == '__main__':
    main()
