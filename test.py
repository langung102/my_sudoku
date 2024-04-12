import heapq
from collections import defaultdict
import time
import resource

def find_empty_location(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None

def get_neighbors(board, row, col):
    neighbors = []
    
    # Get numbers already present in the row, column, and 3x3 sub-grid
    row_values = set(board[row])
    col_values = set(board[i][col] for i in range(9))
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    square_values = set(board[i][j] for i in range(start_row, start_row + 3) for j in range(start_col, start_col + 3))
    
    # Get unique numbers available for this cell
    unique_numbers = set(range(1, 10)) - (row_values | col_values | square_values)
    
    # Generate neighbor nodes with priorities based on uniqueness
    for num in unique_numbers:
        new_board = [row[:] for row in board]
        new_board[row][col] = num
        priority = len(unique_numbers)  # Higher priority for cells with fewer choices
        neighbors.append((priority, new_board))
    
    return neighbors

def best_first_search(board):
    priority_queue = []
    heapq.heappush(priority_queue, (0, board))
    
    while priority_queue:
        _, current_board = heapq.heappop(priority_queue)
        
        empty_location = find_empty_location(current_board)
        if not empty_location:
            return current_board
        
        row, col = empty_location
        
        # Get neighbor nodes and add them to the priority queue
        neighbors = get_neighbors(current_board, row, col)
        for neighbor_priority, neighbor_board in neighbors:
            heapq.heappush(priority_queue, (neighbor_priority, neighbor_board))
    
    return None

def solve_puzzle():
    filename = 'sudoku_puzzles.txt'  # Change this to the path of your file
    puzzles = read_matrix_from_file(filename)
    
    start_time = time.time()
    peak_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    for puzzle in puzzles:
        solved_board = best_first_search(puzzle)
        if solved_board:
            print("Solution:")
            for row in solved_board:
                print(row)
            print("\n")
        else:
            print("No solution exists for this puzzle.")
    
    end_time = time.time()
    peak_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - peak_memory_usage
    execution_time = end_time - start_time
    
    print("Total execution time:", execution_time, "seconds")
    print("Peak memory usage:", peak_memory_usage, "kilobytes")

def read_matrix_from_file(filename):
    matrices = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        matrix = []
        for line in lines:
            line = line.strip()
            if line:
                row = [int(num) for num in line.split()]
                matrix.append(row)
            else:
                if matrix:
                    matrices.append(matrix)
                    matrix = []
        if matrix:  # in case the file doesn't end with an empty line
            matrices.append(matrix)
    return matrices

solve_puzzle()