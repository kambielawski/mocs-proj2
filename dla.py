from enum import Enum
import math
import random
from time import perf_counter

from sklearn.linear_model import LinearRegression
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
    image = plt.imread(dla)

    # Initialize lists to store the log of box sizes and log of box counts
    box_lengths = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
    log_box_sizes = np.log(box_lengths)
    log_box_counts = []

    # for each box_length in the list, count the number of boxes needed to cover the structure
    for box_length in box_lengths:
        box_count = 0

        # For each row and each column, move from index 0 to index[box_length] in intervals of box_length
        for x in range(0, image.shape[0], box_length):
            for y in range(0, image.shape[1], box_length):

                # If the box_length by box_length area contains a nonzero element, add 1 to box count
                if np.any(image[x:x+box_length, y:y+box_length]):
                    box_count += 1

        log_box_counts.append(np.log(box_count))

    # Perform linear regression on the log-log data
    x = np.array(log_box_sizes).reshape(-1,1)
    y = np.array(log_box_counts)
    model = LinearRegression().fit(x, y)
    estimated_dimension = -model.coef_[0]

    # Plot box_count as a function of box_length in log-log space
    plt.scatter(log_box_sizes, log_box_counts, label=None)
    plt.plot(log_box_sizes, model.predict(x), color='purple', label="Linear Fit")
    plt.xlabel('Log(Box Size)')
    plt.ylabel('Log(Box Count)')
    plt.legend()
    plt.title('')
    plt.grid(True)
    plt.show()

    return estimated_dimension


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
    save_image(dla, f'dla_{levy:d}_{gravity:d}.png')
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
