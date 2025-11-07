import numpy as np
import sys

'''
Read file input/Create Board
'''
# At global scope, read in all file data into 3 variables:
# r - row count
# c - column count
# grid - 2D list of cells, with integers representing cell colors
# note that 0's represent empy cells
r = None
c = None
STARTING_BOARD = []

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
        STARTING_BOARD.append(curr_row)

# convert grid to numpy
STARTING_BOARD = np.array(STARTING_BOARD)

print(f'{r} {c}')
print(STARTING_BOARD)

'''
Helper method for identifying possible moves at given board state
'''
# Find clusters using adjacent cells
# Takes in a given board state
# If a cluster has size >= 2, add it to a list of possible moves
def find_clusters(board):
    visited = set()
    clusters = []
    
    # Iterate through cells in board to group together into clusters
    for i in range(r):
        for j in range(c):
            # Skip any cells that have already been visited, meaning it's in another cluster
            # Also skip any empty cells
            if (i, j) in visited or board[i, j] == 0:
                continue;

            # Form cluster for current cell if it isn't part of a cluster
            cluster = group_adjacent(board, r, c, i, j)        
            for cell in cluster:
                visited.add(cell)

            # If cluster contains 2 or more cells, it's a valid move - add to list
            if len(cluster) >= 2:
                clusters.append(cluster)

    return clusters


'''
Helper method for making a move/updating board
'''
# Take in a board state and a cluster to delete
# Remove cluster from board, apply game rules as needed (gravity/left-shifting)
# Return new board state
def remove_cluster(board, cluster):
    # Copy board to modify (have to be explicit, since otherwise it's just a reference)
    modified_board = board.copy()

    # Remove cluster (set all cells to '0')
    for (i, j) in cluster:
        modified_board[i, j] = 0

    # Apply gravity to cells (make nonzero cells above '0' cells fall down)
    # Done by iterating through columns from the bottom-up
    for j in range(0, c):
        nonzero_cells = []
        for i in range(0, r):
            # If cell is nonzero, add to list
            if modified_board[i, j] != 0:
                nonzero_cells.append(modified_board[i, j])

        # Track row in current column to replace tiles
        current_row = r - 1
        
        # Rebuilding column from the bottom-up
        for cell in reversed(nonzero_cells):
            modified_board[current_row, j] = cell
            current_row -= 1

        # Once all nonzero cells have been placed, fill rest of column with '0'
        for i in range(0, current_row + 1):
            modified_board[i, j] = 0

    # Apply 'left-shift' for columns
    # WORK IN PROGRESS

    return modified_board
 
'''
Tree search method
'''
# Build decision tree for a given board state
# Use helper method to generate list of all possible moves
# Create a child node for every move, and use helper method to get new board state/score increase
#def build_tree(board, depth, alpha):

'''
Rolling horizon approach
'''
# Loop through board, beginning with starting position
# In every iteration, build/traverse decision tree to find best path(s) at a certain hard-coded depth
# Feed best options back into tree search until end state is found (helper method fails to return any possible moves)
'''
def run_game(board):
    while True:
        if(find_clusters() == 0):
            break;
        else:
'''

'''
Helper method to determine points scored from final moves list
'''
# In essence, loop through final path and calculate cumulative score
# (Get points for each move with formula (n-1)^2)
# Each move has cluster size as second value
'''
def determine_score(moves):
    score = 0

    for m in moves:
        _, cluster_size, _, _ = m
        move_score = (cluster_size - 1) ** 2
        score += move_score

    return score
'''

'''
Output to console
'''

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


'''
Builds adjacency set of same color, only along rows and columns, not diaganols as specified
'''
def group_adjacent(board, r, c, i, j):
    visited = set()
    adjacent = set()
    # Only find ajacent indices for possible colors 1-8
    if 1 <= board[i, j] <= 8:
        # Add initial cell, then find all adjacent ones with helper method
        adjacent.add((i, j)) 
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


# --- TESTING ---
clusters = find_clusters(STARTING_BOARD)
print(f"Found {len(clusters)} clusters:")
for idx, cl in enumerate(clusters):
    print(f"  Cluster {idx + 1} (size {len(cl)}): {cl}")

new_board =  None
# Try removing the first cluster
if clusters:
    print("\nRemoving first cluster")
    new_board = remove_cluster(STARTING_BOARD, clusters[0])
    print(new_board)

if clusters:
    print("\nRemoving fifth cluster")
    new_board = remove_cluster(new_board, clusters[4])
    print(new_board)

print("\n\nBUG TO FIX - Clusters have to be recalculated after moving, since this just tries to delete the cluster from it's old space (which is already empty)")
if clusters:
    print("\nRemoving third cluster")
    new_board = remove_cluster(new_board, clusters[2])
    print(new_board)
