# AlgoBOWL

Fall 2025 AlgoBOWL project

Team Members:

Jack Baer

Tyler Collingridge

Konrad Quam

---

## Problem Description

Solver for the 'Same Game' problem

---

## Chosen Approach

Build a decision tree from the starting board state and simulate the game to a certain number of moves (depth). Looking at the best option(s) at this point, create a new tree and continue the simulation, repeating until the end of the game (i.e. when no more clusters can be removed).

---

## Documentation

### `group_adjacent(board, r, c, i, j)`
Finds all adjacent cells of the same color (no diagonals) starting from position `(i, j)`

**Parameters:**
- `board (np.ndarray)`: The current game board
- `r (int)`: Number of rows
- `c (int)`: Number of columns
- `i (int)`, `j (int)`: Starting cell coordinates

**Returns:**  
`set[tuple[int, int]]` — Set of coordinates belonging to the same-color cluster

---

### `adjacency_helper(board, r, c, i, j, visited, adjacent)`
Recursive helper for `group_adjacent()` that explores neighboring cells and fills the adjacency set

**Parameters:**
- `board (np.ndarray)`
- `r, c (int)`: Board dimensions
- `i, j (int)`: Current cell being examined
- `visited (set)`: Cells already checked
- `adjacent (set)`: Cells in the current cluster

**Returns:**  
None (modifies `visited` and `adjacent` in place)

---

### `find_clusters(board)`
Scans the board to identify all clusters (groups of 2 or more adjacent same-colored cells)

**Parameters:**
- `board (np.ndarray)`: The current board state

**Returns:**  
`list[set[tuple[int, int]]]` — A list of valid clusters

---

### `remove_cluster(board, cluster)`
Removes all cells in a given cluster from the board, then applies:
- **Gravity:** Nonzero cells fall down
- **Left-shift:** Empty columns are removed and columns to the right are shifted left

**Parameters:**
- `board (np.ndarray)`: Current game state
- `cluster (set[tuple[int, int]])`: Cluster to remove

**Returns:**  
`np.ndarray` — New board state after removal and adjustments

---

### `determine_score(moves)`
Computes total score based on a list of performed moves using the formula:  
\[
\text{score} = \sum{(n - 1)^2}
\]

**Parameters:**
- `moves (list[tuple[int, int, int, int]])`:  
  Each move is `(color, cluster_size, row, column)`

**Returns:**  
`int` — Total calculated score

---

### `run_game(board)`
Main control loop that repeatedly:
1. Finds all possible clusters
2. Uses `find_best_path()` to explore move sequences up to a fixed tree depth
3. Rebuilds tree based on best move (rolling horizon)
4. Continues until no moves remain

**Parameters:**
- `board (np.ndarray)`: Starting board

**Returns:**  
`tuple[list, int, list]` — `(moves, score, clusters_used)`

---

### `find_best_path(board, path, clusters_so_far, depth)`
Recursively explores all valid move sequences up to a specified search depth, returning the path that yields the highest score

**Parameters:**
- `board (np.ndarray)`: Current board state
- `path (list)`: Current sequence of moves
- `clusters_used (list)`: Clusters removed along the path
- `depth (int)`: Current recursion depth

**Returns:**  
`tuple[int, list, list]` — `(best_score, best_path, best_clusters)`

---

