import numpy as np
import sys

r = None
c = None
grid = []

with open("case1.txt", "r") as file:
    line_num = 0
    for line in file:
        line = line.strip()
        if line_num == 0:
            r = int(line[0])
            c = int(line[2])
        else:
            row = []
            for char in line:
                row.append(int(char))
            grid.append(row)
        line_num += 1

grid = np.array(grid)

print(f'{r} {c}')
print(grid)

'''
Builds adjacency set of same color, only along rows and columns, not diaganols as specified
'''
def group_adjacent(grid, r, c, i, j):
    visited = set()
    adjacent = set()
    adjacency_helper(grid, r, c, i, j, visited, adjacent)

'''
checks each adjacent index for the same color adds it to adjacent set if it. 
Also keeps a visited set to not revisit indices.
'''
def adjacency_helper(grid, r, c, i, j, visited, adjacent):
    visited.add((i, j))
    if 0 <= i-1 < r and 0<= j < c and not (i-1, j) in visited:
        if grid[i-1, j] == grid[i, j]:
            adjacent.add((i-1, j))
            adjacency_helper(grid, r, c, i-1, j, visited, adjacent)
    if 0 <= i+1 < r and 0<= j < c and not (i+1, j) in visited:
        if grid[i+1, j] == grid[i, j]:
            adjacent.add((i+1, j))
            adjacency_helper(grid, r, c, i+1, j, visited, adjacent)
    if 0 <= i < r and 0<= j-1 < c and not (i, j-1) in visited:
        if grid[i, j-1] == grid[i, j]:
            adjacent.add((i, j-1))
            adjacency_helper(grid, r, c, i, j-1, visited, adjacent)
    if 0 <= i < r and 0<= j+1 < c and not (i, j+1) in visited:
        if grid[i, j+1] == grid[i, j]:
            adjacent.add((i, j+1))
            adjacency_helper(grid, r, c, i, j+1, visited, adjacent)
