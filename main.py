import copy
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
                continue

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

    # print("Removing cluster:", cluster)

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
Rolling horizon approach - build/rebuild decision tree with find_best_path()
'''
# Loop through board, beginning with starting position
# In every iteration, build/traverse decision tree to find best path(s) at a certain hard-coded depth
# Feed best options back into tree search until end state is found (helper method fails to return any possible moves)

# TODO: amounts of removals are incorrect (for example, on case1.txt input, it is outputting that it is removing 3 squares of 1 when it should be removing 4)
def run_game(board):
    
    total_score = 0
    moves = []
    current_board = board.copy()
    while True:
        
        # # TODO: delete TESTING 
        # print(f'Running game with board {board} and path {path}')

        # Get all moves for current board state. If none, game is over, so return current moves/clusters used 
        clusters = find_clusters(current_board)
        if (len(clusters) == 0):
            break # TODO: delete clusters_used

        # search for best path at max depth
        best_score, best_path, best_clusters = find_best_path(current_board, [], [], 0)         
        
        # If returned path is empty, no more valid moves, so return
        if not best_path:
            break
        # TODO: dump best_score? 

        # TODO: only apply first move from search? (see optimizations file)
        # update the board with the new moves
        # possible alternative solution:
        # for each path in best_path:
          # cluster = find_cluster(path) # TODO: create a helper function that can take a row, column, and color and get a single cluster
            # board = remove_cluster(board, cluster)
        
        # Follow the first move on our 'best path'
        first_move = best_path[0]
        first_cluster = best_clusters[0]

        color, cluster_size, row, col = first_move

        for cluster in best_clusters:
            current_board = remove_cluster(current_board, cluster)
        for move in best_path:
            moves.append(move)

        #current_board = remove_cluster(current_board, first_cluster)

        # Move was taken, so update our lists accordingly
        #moves.append(first_move)
        total_score = determine_score(moves)

    return moves, total_score, current_board

def find_best_path(board, path, clusters_so_far, depth):
    
    clusters = find_clusters(board)
    
    # If no moves are found or depth is reached, end recursion
    if len(clusters) == 0 or depth >= MAX_DEPTH:
        return determine_score(path), path, clusters_so_far

    # Greedy fallback - if MAX_DEPTH is 1, save work by avoiding recursion altogether
    if MAX_DEPTH == 1:
        # Get largest cluster in list
        best_cluster = max(clusters, key=len)  # largest cluster
        
        # Get cluster properties
        row, col = list(best_cluster)[0]
        color = int(board[row, col])
        cluster_size = len(best_cluster)

        # Calculate score
        move_score = (cluster_size - 1) ** 2
        
        # Remove cluster
        next_board = remove_cluster(board, best_cluster)
        
        # Return relevant information
        return move_score, [(color, cluster_size, row, col)], [best_cluster]

    #print(f"Depth {depth}: {len(clusters)} clusters: {clusters}")
    BEAM_SIZE = 5

    # Generate a list of all moves and store all relevant information about them
    candidates = []
    for cluster in clusters:
        cluster_size = len(cluster)
        row, col = list(cluster)[0]
        color = int(board[row, col])

        move_score = (cluster_size - 1) ** 2

        next_board = remove_cluster(board, cluster)

        candidates.append((move_score, cluster, color, cluster_size, row, col, next_board))

    # Sort candidates based on score, then take the best options (based on beam size)
    def get_score(candidate):
        move_score, _, _, _, _, _, _ = candidate
        return move_score

    candidates.sort(key = get_score, reverse = True)

    # Remove all but the best moves
    if(len(candidates) > BEAM_SIZE):
        candidates = candidates[:BEAM_SIZE]

    current_best_score = -1
    current_best_path = []
    current_best_clusters = []

    # Iterate through best options and find which path to take
    for move_score, cluster, color, cluster_size, row, col, next_board in candidates:
        next_path = path + [(color, cluster_size, row, col)]
        next_clusters = clusters_so_far + [cluster]

        # Recurse on path
        child_score, child_path, child_clusters = find_best_path(next_board, next_path, next_clusters, depth + 1)

        # compare paths
        if(child_score > current_best_score):
            current_best_score = child_score
            current_best_path = child_path
            current_best_clusters = child_clusters

    # return once we search all adjacent paths (clusters)
    return current_best_score, current_best_path, current_best_clusters

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
MAX_DEPTH = 1

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

# Determine MAX_DEPTH to search based on board size
board_size = r * c

if(board_size <= 50):
    MAX_DEPTH = 6
elif(board_size <= 300):
    MAX_DEPTH = 5
elif(board_size <= 600):
    MAX_DEPTH = 4
elif(board_size <= 1000):
    MAX_DEPTH = 3
elif(board_size <= 2000):
    MAX_DEPTH = 2
elif(board_size <= 10000):
    MAX_DEPTH = 1

#print(MAX_DEPTH)
MAX_DEPTH = 6
#print(MAX_DEPTH)
'''
Output to console
'''
moves = [] # List of moves in selected path
score = 0 # Number of points awarded in decision path

(moves, score, clusters_used) = run_game(STARTING_BOARD) # TODO: delete clusters_used

# Output that converts coordinates into 1-indexed pairs with origin in the bottom left
print(score)
print(len(moves))
for (color, cluster_size, row, col) in moves:
    converted_row_index = r - row
    converted_col_index = col + 1
    print(str(color) + " " + str(cluster_size) + " " + str(converted_row_index) + " " + str(converted_col_index))


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

