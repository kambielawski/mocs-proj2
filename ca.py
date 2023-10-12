import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from enum import Enum

EMPTY = 0
HEALTHY_CELL = 1
CANCER_CELL = 2

P_HEALTHY_TO_EMPTY_SPREAD = 0     # Probability of a healthy cell spreading to an empty cell
P_CANCER_TO_EMPTY_SPREAD = 0.2    # Probability of a cancer cell spreading to an empty cell
P_CANCER_TO_HEALTHY_SPREAD = 0.1  # Probability of a cancer cell spreading to a healthy cell
P_HEALTHY_TO_CANCER_SPREAD = 0.02 # Probability of a healthy cell "healing" a cancer cell 

# Initialize cell_types
N = 50  # cell_types size
BODY_SIZE = 20
# cell_types = np.random.choice([0, 1, 2], N*N, p=[0.7, 0.3]).reshape(N, N)
cell_types = np.zeros((N,N))
cell_types[(N//2)-BODY_SIZE : (N//2)+BODY_SIZE, (N//2)-BODY_SIZE : (N//2)+BODY_SIZE] = HEALTHY_CELL
cell_types[N//2, N//2] = CANCER_CELL

# cell_strengths = np.zeros((N,N))
# cell_types[N//2, N//2] = 1

def update(frameNum, img, cell_types, N):
    new_cell_types = cell_types.copy()
    for i in range(N):
        for j in range(N):
            current_cell_type = cell_types[i, j]

            left_neighbor_type = cell_types[i, (j-1)%N]
            right_neighbor_type = cell_types[i, (j+1)%N]
            upper_neighbor_type = cell_types[(i-1)%N, j]
            lower_neighbor_type = cell_types[(i+1)%N, j]

            neighborhood = [left_neighbor_type, right_neighbor_type, upper_neighbor_type, lower_neighbor_type]

            n_healthy_neighbors = sum([neighbor == HEALTHY_CELL for neighbor in neighborhood])
            n_cancer_neighbors = sum([neighbor == CANCER_CELL for neighbor in neighborhood])

            r = np.random.rand()

            # RULE SET
            if current_cell_type == EMPTY:
                p_healthy = P_HEALTHY_TO_EMPTY_SPREAD * n_healthy_neighbors
                p_cancer = P_CANCER_TO_EMPTY_SPREAD * n_cancer_neighbors 
                p_empty = 1 - p_healthy - p_cancer
            elif current_cell_type == CANCER_CELL:
                p_healthy = P_HEALTHY_TO_CANCER_SPREAD * n_healthy_neighbors
                p_cancer = 1 - p_healthy
                p_empty = 0
            else: # Healthy cell
                p_cancer = P_CANCER_TO_HEALTHY_SPREAD * n_cancer_neighbors
                p_healthy = 1 - p_cancer
                p_empty = 0
            
            next_cell_type = np.random.choice([EMPTY, HEALTHY_CELL, CANCER_CELL], p=[p_empty, p_healthy, p_cancer])

            new_cell_types[i,j] = next_cell_type

    print(f'Frame {frameNum} done')
    img.set_data(new_cell_types)
    # img.set_data(newSignals / 255)
    cell_types[:] = new_cell_types[:]
    return img,

# Visualization setup
fig, ax = plt.subplots()
img = ax.imshow(cell_types, interpolation='nearest')
ani = animation.FuncAnimation(fig, update, fargs=(img, cell_types, N), frames=np.arange(0,50), blit=True)

plt.show()

# ani.save('cancer.gif')
