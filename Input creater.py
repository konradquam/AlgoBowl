import numpy as np
from random import randint
import random
import time

r = 10
c = 10
grid = [[0 for i in range(c)] for j in range(r)]
grid = np.array(grid)
blob_size = int(r * c * 0.2) #must be an integer
start_index = (r-1, 0)
row, col = start_index
colors = [1, 2, 3, 4, 5, 6, 7, 8]
color_index = 0
random.seed(time.time())  # You can use any integer as the seed

def get_adjacent_cells(row, col):
    cells = []
    if row == r-1:
        if col == 0:
            cells.append((row-1, col))
            cells.append((row, col+1))
        elif col == c-1:
            cells.append((row-1, col))
            cells.append((row, col-1))
        else:
            cells.append((row-1, col))
            cells.append((row, col+1))
            cells.append((row, col-1))
    elif row == 0:
        if col == 0:
            cells.append((row+1, col))
            cells.append((row, col+1))
        elif col == c-1:
            cells.append((row+1, col))
            cells.append((row, col-1))
        else:
            cells.append((row+1, col))
            cells.append((row, col+1))
            cells.append((row, col-1))
    else:
        cells.append((row+1, col))
        cells.append((row -1, col))
        cells.append((row, col+1))
        cells.append((row, col-1))

    return cells

def fill(blob_size, row, col, color):
    if blob_size == 0:
        return
    grid[row][col] = color
    adjacent_cells = get_adjacent_cells(row, col)
    next_cells = []
    for cell in adjacent_cells:
        if grid[cell[0], cell[1]] == 0 and cell[0] < r-1:
            if grid[cell[0]+1, cell[1]] != 0:
                next_cells.append(cell)
            elif grid[cell[0]+1][cell[1]] == 0:
                i = 1
                while grid[cell[0]+i][cell[1]] == 0 and cell[0]+i < r-1:
                    i += 1
                    if grid[cell[0]+i, cell[1]] != 0:
                        next_cells.append((cell[0]+i-1, cell[1]))
                    elif cell[0]+i == r-1:
                        next_cells.append((cell[0]+i, cell[1]))
        elif grid[cell[0], cell[1]] == 0 and cell[0] == r-1:
            next_cells.append(cell)
    '''  
    if len(next_cells) == 0:
        return
    '''
    next_cell = next_cells[randint(0, len(next_cells) - 1)]
    fill(blob_size-1, next_cell[0], next_cell[1], color)


color = colors[color_index]
fill(blob_size, row, col, color)


print(grid)