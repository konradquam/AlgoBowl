import numpy as np
import sys

# HELPERS 
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
    gravity_board = board.copy()

    # Remove cluster (set all cells to '0')
    for (i, j) in cluster:
        gravity_board[i, j] = 0

    # Apply gravity to cells (make nonzero cells above '0' cells fall down)
    # Done by iterating through columns from the bottom-up
    for j in range(0, c):
        nonzero_cells = []
        for i in range(0, r):
            # If cell is nonzero, add to list
            if gravity_board[i, j] != 0:
                nonzero_cells.append(gravity_board[i, j])

        # Track row in current column to replace tiles
        current_row = r - 1
        
        # Rebuilding column from the bottom-up
        for cell in reversed(nonzero_cells):
            gravity_board[current_row, j] = cell
            current_row -= 1

        # Once all nonzero cells have been placed, fill rest of column with '0'
        for i in range(0, current_row + 1):
            gravity_board[i, j] = 0

    # Apply 'left-shift' for columns
    # WORK IN PROGRESS
    nonzero_column_indices = []
    for col in range(0, c):
        column_empty = True

        for row in range(0, r):
            if gravity_board[row, col] != 0:
                column_empty = False
                break;

        if not column_empty:
            nonzero_column_indices.append(col)

    # Create an empty copy of the board
    shifted_board = np.zeros_like(gravity_board)

    # Populate with the contents of every nonzero column
    new_col = 0
    for old_col in nonzero_column_indices:
        for row in range(0, r):
            shifted_board[row, new_col] = gravity_board[row, old_col]
        new_col += 1

    return shifted_board

def determine_score(moves):
    score = 0

    for m in moves:
        # # TODO: delete TESTING
        # print(f'Move: {m}')
        _, cluster_size, _, _ = m
        move_score = (cluster_size - 1) ** 2
        score += move_score

    return score

# END HELPERS 

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

# TODO: amounts of removals are incorrect (for example, on case1.txt input, it is outputting that it is removing 3 squares of 1 when it should be removing 4)
def run_game(board):
    
    # while there are clusters
    path = []
    clusters_used = [] # tracks all of the clusters that we used so that they can be removed from the board
    while True:
        
        # # TODO: delete TESTING 
        # print(f'Running game with board {board} and path {path}')

        clusters = find_clusters(board)
        if (len(clusters) == 0):
            return (moves, clusters_used) # TODO: delete clusters_used

        # search for best path at max depth
        (best_score, best_path, best_clusters_used) = find_best_path(board, path, clusters_used, 0) # TODO: dump best_score? 

        # TODO: only apply first move from search? (see optimizations file)
        # update the board with the new moves
        # possible alternative solution:
        # for each path in best_path:
          # cluster = find_cluster(path) # TODO: create a helper function that can take a row, column, and color and get a single cluster
            # board = remove_cluster(board, cluster)
        
        for cluster in best_clusters_used:
            board = remove_cluster(board, cluster)

        # update the moves array with the new moves
        moves.extend(best_path)

# returns (score, path, clusters_used)
def find_best_path(board, path, clusters_used, depth):
    
    # # TODO: delete TESTING
    # print(f'Finding best path for board \n{board}, \npath {path}, and depth {depth}')

    clusters = find_clusters(board)
    # TODO: how are we getting to a depth of 5????? 
    # if we've hit max depth OR we've run out of clusters
    if (depth >= MAX_DEPTH or len(clusters) == 0):
        # # TODO: delete TESTING
        # print(f'moves (path): {path}')
        score = determine_score(path)
        return (score, path, clusters_used)

    # TODO: will this reset the best path and score? 
    current_best_score = -1
    current_best_path = []

    for cluster in clusters:
        # remove cluster
        next_board = remove_cluster(board, cluster)

        # TODO: could we make the recursion work without copying all of the data? 
        # create copy of clusters, and append next cluster
        next_clusters_used = []
        next_clusters_used = clusters_used.copy()
        next_clusters_used.append(cluster)

        # create copy of moves
        next_path = []
        next_path = path.copy()

        # append path
        # (color, amount, row, column)
        (row, col) = cluster.pop()
        color = int(board[row][col]) # TODO: ensure this is indexed properly (the top left corner might be messing it up)
        amount = len(cluster)

        # # TODO: delete TESTING
        # print(f'next path: {(color , amount, row, col)}')

        next_path.append((color , amount, row, col))

        # recursion: find next best path
        depth += 1
        (child_score, child_path, child_clusters) = find_best_path(next_board, next_path, next_clusters_used, depth) 

        # compare paths
        if (child_score > current_best_score):
            current_best_score = child_score
            current_best_path = child_path
            current_best_clusters = child_clusters
    
    # return once we search all adjacent paths (clusters)
    return (current_best_score, current_best_path, current_best_clusters)
'''
Helper method to determine points scored from final moves list
'''
# In essence, loop through final path and calculate cumulative score
# (Get points for each move with formula (n-1)^2)
# Each move has cluster size as second value

# MAIN 
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
MAX_DEPTH = 2

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

print("INPUT:") # TODO: delete
print(f'{r} {c}')
print(STARTING_BOARD)

'''
Output to console
'''
moves = [] # List of moves in selected path
score = 0 # Number of points awarded in decision path
(moves, clusters_used) = run_game(STARTING_BOARD) # TODO: delete clusters_used
# TODO: determine score is breaking here, but not in the run game function 
# # TODO: delete TESTING
# print(f'moves: {moves}')

# TODO: when moves are building in the main game loop, they are being appended together as independent arrays instead of appending as a single array. For example:
# getting moves: [[(), ()], [(), ()]] when we should be getting [(), (), (), ()]

score = determine_score(moves)

print("\nOUTPUT:") # TODO: delete
print(score)
print(len(moves))
print("color amount row col") # TODO: delete 
for m in moves:
    print(m)

# TODO: delete
# print(clusters_used)
# for c in clusters_used:
#     print(c)

# moves:
# 1 4 4 1
# (color(1-8)) (number of squares removed) (row of any square in the cluster) (column of any square in the cluster)
# rows: bottom row is 1, increasing upwards. NOTE: out solution currently starts (0,0) from top left corner
# columns: leftmost column is 1, increasing right 

# TODO: append the first element in the cluster (which is a tuple) to the moves array by breaking it up into a row and a column 


# # --- TESTING ---
# clusters = find_clusters(STARTING_BOARD)
# print(f"Found {len(clusters)} clusters:")
# for idx, cl in enumerate(clusters):
#     print(f"  Cluster {idx + 1} (size {len(cl)}): {cl}")

# new_board =  None
# # Try removing the first cluster
# if clusters:
#     print("\nRemoving third cluster (gravity test)")
#     new_board = remove_cluster(STARTING_BOARD, clusters[2])
#     print(new_board)

# if clusters:
#     print("\nRemoving first cluster (left shift setup)")
#     new_board = remove_cluster(new_board, clusters[0])
#     print(new_board)

# if clusters:
#     print("\nRemoving fourth cluster (left shift test)")
#     new_board = remove_cluster(new_board, clusters[3])
#     print(new_board)
    
# '''
# print("\n\nBUG TO FIX - Clusters have to be recalculated after moving, since this just tries to delete the cluster from it's old space (which is already empty)")
# if clusters:
#     print("\nRemoving third cluster")
#     new_board = remove_cluster(new_board, clusters[2])
#     print(new_board)
# '''
