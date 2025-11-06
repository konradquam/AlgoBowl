import numpy as np
import sys

'''
Read file input/Create Board
'''
r = None
c = None
grid = []

if len(sys.argv) < 2:
    print("Usage: python3 strat1.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

with open(filename, "r") as file:
    first_line = file.readline();
    r, c = map(int, first_line.split());

    for line in file:
        cleaned_line = line.strip()
        curr_row = []
        for char in cleaned_line:
            curr_row.append(int(char))
        grid.append(curr_row)

grid = np.array(grid)

print(f'{r} {c}')
print(grid)

'''
Global variables
'''
moves = []
score = 0

'''
Helper method for identifying possible moves at given board state
'''
# Find clusters using adjacent cells
# If a cluster has size >= 2, add it to a list of possible moves


'''
Helper method for making a move/updating board
'''
# Take in a board and a move from our list
# Remove cluster from board, apply game rules as needed (gravity/left-shifting)
# Return new board state

'''
Tree search method
'''
# Build decision tree for a given board state
# Use helper method to generate list of all possible moves
# Create a child node for every move, and use helper method to get new board state/score increase

'''
Rolling horizon approach
'''
# Loop through board, beginning with starting position
# In every iteration, build/traverse decision tree to find best path(s) at a certain hard-coded depth
# Feed best options back into tree search until end state is found (helper method fails to return any possible moves)

'''
Output to console
'''
print(len(moves))
# All subsequent lines are each individual movement Rocket and Lucky take
for m in moves:
    print(m)



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
