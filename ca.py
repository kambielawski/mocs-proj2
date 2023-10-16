import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from enum import Enum
from numba import jit

# Initialize cell_types
EMPTY = 0
HEALTHY_CELL = 1
CANCER_CELL = 2

P_HEALTHY_TO_EMPTY_SPREAD = 0.25     # Probability of a healthy cell spreading to an empty cell
P_CANCER_TO_EMPTY_SPREAD = 0.2       # Probability of a cancer cell spreading to an empty cell
P_CANCER_TO_HEALTHY_SPREAD = 0.08    # Probability of a cancer cell spreading to a healthy cell
P_HEALTHY_TO_CANCER_SPREAD = 0.07    # Probability of a healthy cell "healing" a cancer cell 
ABLATION_RADIUS = 6

N = 50
BODY_SIZE = 24
N_FRAMES = 100
ABLATION_FRAME = 50

# Create the cell grid
cell_types = np.zeros((N,N))
cell_types[(N//2)-BODY_SIZE : (N//2)+BODY_SIZE, (N//2)-BODY_SIZE : (N//2)+BODY_SIZE] = HEALTHY_CELL
# cell_types[:, :] = HEALTHY_CELL
cell_types[N//2:N//2+2, N//2:N//2+2] = CANCER_CELL

def num_healthy_cells(grid):
    return np.count_nonzero(grid == HEALTHY_CELL)

def num_cancer_cells(grid):
    return np.count_nonzero(grid == CANCER_CELL)

def ablate(grid, R, position):
    x_center, y_center = position
    Y, X = np.ogrid[:grid.shape[0], :grid.shape[1]]
    dist_from_center = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    
    mask = dist_from_center <= R
    grid[mask] = 0
    
    return grid

def update_cell_types(frameNum, cell_types, N, ep):
    if frameNum == ABLATION_FRAME:
        new_cell_types = ablate(cell_types, ep['ABLATION_RADIUS'], (N//2, N//2))
    else:
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

                # RULE SET
                if current_cell_type == EMPTY:
                    p_healthy = ep['P_HEALTHY_TO_EMPTY_SPREAD'] * n_healthy_neighbors
                    p_cancer = ep['P_CANCER_TO_EMPTY_SPREAD'] * n_cancer_neighbors 
                    p_empty = 1 - p_healthy - p_cancer
                elif current_cell_type == CANCER_CELL:
                    p_healthy = ep['P_HEALTHY_TO_CANCER_SPREAD'] * n_healthy_neighbors
                    p_cancer = 1 - p_healthy
                    p_empty = 0
                else: # Healthy cell
                    p_cancer = ep['P_CANCER_TO_HEALTHY_SPREAD'] * n_cancer_neighbors
                    p_healthy = 1 - p_cancer
                    p_empty = 0
                
                next_cell_type = np.random.choice([EMPTY, HEALTHY_CELL, CANCER_CELL], p=[p_empty, p_healthy, p_cancer])

                new_cell_types[i,j] = next_cell_type

    return new_cell_types

def update(frameNum, img, cell_types, N, ep):
    new_cell_types = update_cell_types(frameNum, cell_types, N, ep)

    print(f'Frame {frameNum} done')
    img.set_data(new_cell_types)
    cell_types[:] = new_cell_types[:]
    return img,

experiment_params = {
'P_HEALTHY_TO_EMPTY_SPREAD': 0.25,     # Probability of a healthy cell spreading to an empty cell
'P_CANCER_TO_EMPTY_SPREAD': 0.2,       # Probability of a cancer cell spreading to an empty cell
'P_CANCER_TO_HEALTHY_SPREAD': 0.08,    # Probability of a cancer cell spreading to a healthy cell
'P_HEALTHY_TO_CANCER_SPREAD': 0.07,    # Probability of a healthy cell "healing" a cancer cell 
'ABLATION_RADIUS': 4
}

# # Visualization setup
fig, ax = plt.subplots()
img = ax.imshow(cell_types, interpolation='nearest')
ani = animation.FuncAnimation(fig, update, fargs=(img, cell_types, N, experiment_params), frames=np.arange(0,N_FRAMES), blit=True)

ani.save('ablation_6.gif')
# plt.show()


