import numpy as np
import sys

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
Read file input/Create Board
'''
# At global scope, read in all file data into 3 variables:
# r - row count
# c - column count
# STARTING_BOARD - 2D list of cells, with integers representing cell colors
# note that 0's represent empy cells
r = None
c = None
STARTING_BOARD = []

if len(sys.argv) < 3:
    print("Usage: python3 verifier.py <input_filename> <output_filename>")
    sys.exit(1)

input_filename = sys.argv[1]

with open(input_filename, "r") as file:
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
Read output file input
'''

output_filename = sys.argv[2]

total_score = None
num_moves = None
moves = []

# Read in all file data:
# total_score - points
# num_moves - number of moves in solution
# moves - collection of all moves
with open(output_filename, "r") as file:
    total_score = int(file.readline().strip())
    num_moves = int(file.readline().strip())
    
    for _ in range(num_moves):
        parts = list(map(int, file.readline().split()))
        moves.append(parts)  # [color, size, row, col]



# Simulate game according to provided moves, and catch any of the following problems:

# Illegal Moves:
# Moves of size 1
# Moves using partial clusters
# Moves using multiple colors
# Move not possible for given board state

# Parameter Errors:
# Number of moves does not match specified amount
# Points do not match moves
# Cell specified in move matches specified color

def verify_solution():
    # Prepare board state and scoring
    current_board = STARTING_BOARD.copy()
    total_score_computed = 0

    print("\nOutput score: " + str(total_score))
    print("Output move count: " + str(num_moves))
    print("\nMoves in format (color, cluster size, row, column)\n")

    # Process each move
    for move_id, (color, size, row, col) in enumerate(moves, start=1):
        print(f"Move {move_id} in output: (" + str(color) + ", " + str(size) + ", " + str(row) + ", " + str(col) + ")")
        # Convert coordinates from 1-indexed bottom-left origin to array indices
        i = r - row
        j = col - 1

        # Check cluster size
        if(size == 1):
            print("Illegal move - cluster of size 1 cannot be removed")
            return False

        # Check coordinate validity
        if not ((0 <= i < r) and (0 <= j < c)):
            print("Illegal move - coordinates of specified cell out of bounds")
            return False

        # Check cell color
        cell_color = current_board[i, j]
        if cell_color != color:
            print("Illegal move - move color doesn't match cell color")
            return False


        # Find the connected cluster
        cluster = group_adjacent(current_board, r, c, i, j)
        cluster_size = len(cluster)
        

        # Make sure cluster found from specified cell matches specified size
        if cluster_size != size:
            print("Illegal move - cluster size does not match specified value")
            return False

        # Make sure no partial clusters are taken in a move 
        full_cluster = group_adjacent(current_board, r, c, i, j)
        full_cluster_size = len(full_cluster)
        if size < full_cluster_size:
            print("Illegal move - partial cluster removal detected")
            return False


        # Check that cluster is homogenous
        multiple_colors = False
        for (x, y) in cluster:
            cell_color = current_board[x, y]
            if cell_color != color:
                multiple_colors = True

        if multiple_colors:
            print("Illegal move - color mismatch in cluster")
            return False

        # Remove cluster and update board state
        current_board = remove_cluster(current_board, cluster)

        # Update score
        total_score_computed += (size - 1) ** 2 

    # Compare final computed score
    if total_score_computed != total_score:
        print("Score mismatch - moves do not yield specified final point value")
        return False
    # Check moves in output
    if len(moves) != num_moves:
        print("Move mismatch - number of provided moves not equal to specified amount in output")
        return False

    print("All tests passed - output is valid")
    return True

# Run verifier on given input/output files
verify_solution()
