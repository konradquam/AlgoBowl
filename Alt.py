import copy
import numpy as np
import sys
import random as rand
import time
import math
import statistics

from tensorflow.python.ops.numpy_ops.np_random import randint

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
    if 0 <= i - 1 < r and 0 <= j < c and not (i - 1, j) in visited:
        if board[i - 1, j] == board[i, j]:
            adjacent.add((i - 1, j))
            adjacency_helper(board, r, c, i - 1, j, visited, adjacent)
    if 0 <= i + 1 < r and 0 <= j < c and not (i + 1, j) in visited:
        if board[i + 1, j] == board[i, j]:
            adjacent.add((i + 1, j))
            adjacency_helper(board, r, c, i + 1, j, visited, adjacent)
    if 0 <= i < r and 0 <= j - 1 < c and not (i, j - 1) in visited:
        if board[i, j - 1] == board[i, j]:
            adjacent.add((i, j - 1))
            adjacency_helper(board, r, c, i, j - 1, visited, adjacent)
    if 0 <= i < r and 0 <= j + 1 < c and not (i, j + 1) in visited:
        if board[i, j + 1] == board[i, j]:
            adjacent.add((i, j + 1))
            adjacency_helper(board, r, c, i, j + 1, visited, adjacent)


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

def find_a_path(board):
    moves = []
    boards = []
    while True:
        possible_moves = find_clusters(board)
        if len(possible_moves) == 0:
            break
        selected_move = possible_moves[rand.randint(0, len(possible_moves) - 1)]
        move_cells = list(selected_move)

        cluster_size = len(move_cells)
        row, col = move_cells[0]
        color = board[row, col]
        moves.append((color, cluster_size, row, col))

        boards.append(board)
        board = remove_cluster(board, selected_move)
    return moves, boards

def find_alt_path(moves, boards):
    move_to_change = rand.randint(0, len(moves) - 1)
    new_moves = moves[:move_to_change]
    new_boards = boards[:move_to_change]
    next_moves, next_boards = find_a_path(boards[move_to_change])

    return new_moves + next_moves, new_boards + next_boards

def choose_path(og_path, new_path, sigma):
    cost = determine_score(og_path) - determine_score(new_path)
    rho = math.exp(-cost/sigma)
    if np.random.uniform(1, 0, 1) < rho:
        return new_path, cost
    return og_path, 0

def calculate_initial_sigma(INITIAL_BOARD):
    scores = []
    for i in range(10):
        moves, _ = find_a_path(INITIAL_BOARD)
        score = determine_score(moves)
        scores.append(score)

    return statistics.pstdev(scores)

def run_game(board):
    rand.seed(time.time())
    path = find_a_path(board)
    final_score = determine_score(path)

#     return path, final_score, path

def run_simulated_annealing(board, total_iterations):
    INITIAL_BOARD = board
    clusters = find_clusters(INITIAL_BOARD)
    initial_path, initial_path_boards = find_a_path(INITIAL_BOARD)
    best_path = initial_path
    best_path_boards = initial_path_boards
    initial_score = determine_score(initial_path)
    decreaseFactor = 0.99
    stuckCount = 0
    iterations = len(clusters)
    sigma = calculate_initial_sigma(INITIAL_BOARD)
    for _ in range(total_iterations):
        curr_path = find_alt_path(best_path, best_path_boards)
        for i in range (0, iterations):
            new_path, new_path_boards = find_alt_path(best_path, best_path_boards)
            curr_path, _ = choose_path(best_path, new_path, sigma)


        sigma *= decreaseFactor
        if determine_score(best_path) == determine_score(curr_path):
            stuckCount += 1
        else:
            stuckCount = 0
            sigma = calculate_initial_sigma(INITIAL_BOARD)
        if stuckCount > 80:
            sigma += 2

    return best_path



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


'''
Output to console
'''
moves = []  # List of moves in selected path
score = 0  # Number of points awarded in decision path
total_iterations = 1000

# (moves, score, clusters_used) = run_game(STARTING_BOARD)  # TODO: delete clusters_used
moves = run_simulated_annealing(STARTING_BOARD, total_iterations)

# Output that converts coordinates into 1-indexed pairs with origin in the bottom left
print(determine_score(moves))
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

