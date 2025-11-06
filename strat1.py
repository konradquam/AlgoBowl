import numpy as np
import sys

'''
Read file input/Create Board
'''
# At global scope, read in all file data into 3 variables:
# r - row count
# c - column count
# grid - 2D list of cells, with integers representing cell colors
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
Helper method for identifying possible moves at given board state
'''
# Find clusters using adjacent cells
# Takes in a given board state
# If a cluster has size >= 2, add it to a list of possible moves
def find_clusters(board):
    clusters = set()
    for i in range(r):
        for j in range(c):
            adjacent = group_adjacent(board, r, c, i, j)
            if len(adjacent) >= 2 and adjacent not in clusters:
                clusters.add(adjacent)


'''
Helper method for making a move/updating board
'''
# Take in a board state and a cluster to delete
# Remove cluster from board, apply game rules as needed (gravity/left-shifting)
# Return new board state
def remove_cluster(board, cluster):

'''
Tree search method
'''
# Build decision tree for a given board state
# Use helper method to generate list of all possible moves
# Create a child node for every move, and use helper method to get new board state/score increase
def build_tree(board, depth, alpha):

'''
Rolling horizon approach
'''
# Loop through board, beginning with starting position
# In every iteration, build/traverse decision tree to find best path(s) at a certain hard-coded depth
# Feed best options back into tree search until end state is found (helper method fails to return any possible moves)
def run_game(board):
    while True:
        if(find_clusters() == 0):
            break;
        else:


'''
Helper method to determine points scored from final moves list
'''
# In essence, loop through final path and calculate cumulative score
# (Get points for each move with formula (n-1)^2)
# Each move has cluster size as second value
def determine_score(moves):
    score = 0

    for m in moves:
        _, cluster_size, _, _ = m
        move_score = (cluster_size - 1) ** 2
        score += move_score

    return score

'''
Output to console
'''
moves = [] # List of moves in selected path
score = 0 # Number of points awarded in decision path

moves = run_game(grid)

score = determine_score(moves)

print(score)
print(len(moves))
for m in moves:
    print(m)



'''
Builds adjacency set of same color, only along rows and columns, not diaganols as specified
'''
def group_adjacent(board, r, c, i, j):
    visited = set()
    adjacent = set()
    adjacency_helper(board, r, c, i, j, visited, adjacent)
    return adjacent

'''
checks each adjacent index for the same color adds it to adjacent set if it. 
Also keeps a visited set to not revisit indices.
'''
def adjacency_helper(board, r, c, i, j, visited, adjacent):
    visited.add((i, j))
    if 0 <= i-1 < r and 0<= j < c and not (i-1, j) in visited:
        if board[i-1, j] == board[i, j]:
            adjacent.add((i-1, j))
            adjacency_helper(board, r, c, i-1, j, visited, adjacent)
    if 0 <= i+1 < r and 0<= j < c and not (i+1, j) in visited:
        if board[i+1, j] == board[i, j]:
            adjacent.add((i+1, j))
            adjacency_helper(board, r, c, i+1, j, visited, adjacent)
    if 0 <= i < r and 0<= j-1 < c and not (i, j-1) in visited:
        if board[i, j-1] == board[i, j]:
            adjacent.add((i, j-1))
            adjacency_helper(board, r, c, i, j-1, visited, adjacent)
    if 0 <= i < r and 0<= j+1 < c and not (i, j+1) in visited:
        if board[i, j+1] == board[i, j]:
            adjacent.add((i, j+1))
            adjacency_helper(board, r, c, i, j+1, visited, adjacent)
