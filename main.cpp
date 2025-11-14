#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <cstdint>
#include <algorithm>
#include <string>
#include <utility>

using namespace std;

int R, C;
int MAX_DEPTH = 1;

using Cluster = vector<pair<int,int>>;
using Board = vector<vector<uint8_t>>;
struct Move { 
    int color, size, row, col; 
};

// BFS approach to clustering, using a queue to visit cells and check adjacency
inline Cluster group_adjacent(const Board &board, int i, int j, vector<vector<bool>> &visited) {
    const int color = board[i][j];
    if (color == 0) return {};
    Cluster cluster;
    cluster.reserve(32); // Reserve memory upfront - baseline assumes clusters will be smaller than 32 cells

    // Custom BFS implementation to explore adjacent cells with a queue
    queue<pair<int,int>> q;
    q.push({i,j});
    visited[i][j] = true;

    while (!q.empty()) {
        auto [r,c] = q.front(); q.pop();
        cluster.emplace_back(r,c);
        if (r > 0 && !visited[r-1][c] && board[r-1][c]==color) { 
	    visited[r-1][c]=true; q.push({r-1,c}); 
	}
        if (r+1 < R && !visited[r+1][c] && board[r+1][c]==color) { 
	    visited[r+1][c]=true; q.push({r+1,c}); }
        if (c > 0 && !visited[r][c-1] && board[r][c-1]==color) { 
	    visited[r][c-1]=true; q.push({r,c-1}); }
        if (c+1 < C && !visited[r][c+1] && board[r][c+1]==color) { 
	    visited[r][c+1]=true; q.push({r,c+1}); }
        }
    return cluster;
}

// Cluster detection (uses single visited matrix)
inline vector<Cluster> find_clusters(const Board &board) {
    vector<vector<bool>> visited(R, vector<bool>(C, false));
    vector<Cluster> clusters;
    clusters.reserve(R*C/8); // Reserve memory upfront - baseline assumes equal spread of colors

    for (int i=0; i<R; ++i) {
        for (int j=0; j<C; ++j) {
            if (board[i][j]==0 || visited[i][j]) continue;
            Cluster cluster = group_adjacent(board,i,j,visited);
            if (cluster.size() >= 2) clusters.push_back(move(cluster));
        }
    }
    return clusters;
}

// given a path in path array, needs to find a single cluster 
inline Cluster find_cluster(Board &board, const Move &single_move) {
    vector<vector<bool>> visited(R, vector<bool>(C, false));
    return group_adjacent(board, single_move.row, single_move.col, visited);
}

// Apply a move to the given board, altering it by reference
inline void remove_cluster(Board &board, const Cluster &cluster) {
    // turn removed cluster cells to 0
    for (auto &p : cluster) board[p.first][p.second] = 0;

    // gravity
    for (int j=0; j<C; ++j) {
        int write = R-1;
        for (int i=R-1; i>=0; --i) {
            if (board[i][j] != 0) {
                board[write][j] = board[i][j];
                if (write != i) {
		    board[i][j] = 0;
		}
		--write;
            }
        }
        for (; write>=0; --write) {
	    board[write][j] = 0;
	}
    }

    // left-shifting
    int writeCol = 0;
    for (int j=0; j<C; ++j) {
        bool empty = true;
        for (int i=0; i<R; ++i) {
            if (board[i][j] != 0) { 
		empty=false; 
		break; 
	    }
        }
        if (!empty) {
            if (writeCol != j) {
                for (int i=0; i<R; ++i) {
                    board[i][writeCol] = board[i][j];
                    board[i][j] = 0;
                }
            }
            ++writeCol;
        }
    }
}


// Score calculation algorithm
inline int determine_score(const vector<Move> &moves) {
    int score=0;
    for (auto &m : moves) score += (m.size-1)*(m.size-1);
    return score;
}


// Structure to store information for beam search - comparing all candidates for moves to take
struct Candidate {
    int score;
    int color;
    int size;
    int row, col;
    Cluster cluster;
    Board next_board;
};

// Recursive beam search (rolling horizon)
tuple<int, vector<Move>> find_best_path(Board &board, vector<Move> &path, int depth)
{
    auto clusters = find_clusters(board);
    if (clusters.empty() || depth >= MAX_DEPTH)
        return {determine_score(path), path};

    const int BEAM_SIZE = 8;
    vector<Candidate> candidates;
    candidates.reserve(clusters.size());

    for (auto &cl : clusters) {
        int color = board[cl[0].first][cl[0].second];
        int size = cl.size();
        int move_score = (size-1)*(size-1);

        Board next = board; // local copy (cheaper than copy-on-write)
        remove_cluster(next, cl);

        candidates.push_back({move_score, color, size, cl[0].first, cl[0].second, move(cl), move(next)});
    }

    partial_sort(candidates.begin(),
                 candidates.begin() + min(BEAM_SIZE, (int)candidates.size()),
                 candidates.end(),
                 [](auto &a, auto &b){ return a.score > b.score; });

    if ((int)candidates.size() > BEAM_SIZE)
        candidates.resize(BEAM_SIZE);

    int best_score=-1;
    vector<Move> best_path;

    for (auto &cand : candidates) {
        path.push_back({cand.color, cand.size, cand.row, cand.col});
        auto [child_score, child_path] =
            find_best_path(cand.next_board, path, depth+1);
        if (child_score > best_score) {
            best_score = child_score;
            best_path = child_path;
        }
        path.pop_back();
    }

    return {best_score, best_path};
}

// Main Game Function
tuple<vector<Move>, int, Board> run_game(Board board) {
    vector<Move> moves;
    int total_score=0;

    while (true) {
        auto clusters = find_clusters(board);
        if (clusters.empty()) {
		break;
	}

        vector<Move> path;
        auto [best_score, best_path] = find_best_path(board, path, 0);
        if (best_path.empty()) {
	       	break;
	}

        for (int i=0; i < best_path.size(); i++){
            auto &m = best_path[i];
            auto c = find_cluster(board, best_path[i]);
            remove_cluster(board, c);
            moves.push_back(m);
            total_score = determine_score(moves);
        }
        /*
        auto &m = best_path[0];
        auto c = find_cluster(board, best_path[0]); 

        remove_cluster(board, c);

        moves.push_back(m);
        total_score = determine_score(moves);
        */
    }
    return {moves, total_score, board};
}


// Program Entry Point
int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        cerr << "Usage: ./<executable_name> <input.txt>\n";
        return 1;
    }

    ifstream in(argv[1]);
    if (!in) { cerr << "Cannot open input file.\n"; return 1; }

    in >> R >> C;
    Board board(R, vector<uint8_t>(C));
    for (int i=0; i<R; ++i) {
        string line; in >> line;
        for (int j=0; j<C; ++j) board[i][j] = line[j]-'0';
    }

    // Set MAX_DEPTH based on grid size
    int board_size = R*C;
    
    if (board_size <= 50) {
	MAX_DEPTH=12;
    }
    else if (board_size <= 250) {
	MAX_DEPTH=10;
    }
    else if (board_size <= 500) {
	MAX_DEPTH=8;
    }
    else if (board_size <= 2000) {
	MAX_DEPTH=7;
    }
    else if (board_size <= 4000) {
	MAX_DEPTH=6;
    }
    else if (board_size <= 10000) {
	MAX_DEPTH=4;
    }
    else {
	MAX_DEPTH=3;
    }

    auto [moves, score, final_board] = run_game(board);

    cout << score << "\n" << moves.size() << "\n";
    for (auto &m : moves) {
        int converted_row = R - m.row;
        int converted_col = m.col + 1;
        cout << m.color << " " << m.size << " " << converted_row << " " << converted_col << "\n";
    }
}

