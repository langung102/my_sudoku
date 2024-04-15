import heapq
from collections import defaultdict
import time
import resource
import math
from generator import generate_sudoku

PUZZLE_SIZE = 16
SUB_GRID = int(math.sqrt(PUZZLE_SIZE))
def is_valid(board, row, col, num):
    # Check if the number is already present in the current row
    for i in range(PUZZLE_SIZE):
        if board[row][i] == num:
            return False

    # Check if the number is already present in the current column
    for i in range(PUZZLE_SIZE):
        if board[i][col] == num:
            return False

    # Check if the number is already present in the current 3x3 box
    start_row = (row // SUB_GRID ) * SUB_GRID 
    start_col = (col // SUB_GRID ) * SUB_GRID 
    for i in range(SUB_GRID ):
        for j in range(SUB_GRID ):
            if board[start_row + i][start_col + j] == num:
                return False

    return True


def find_empty_location(board):
    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            if board[i][j] == 0:
                return i, j
    return None

def find_empty_location_bfs(board):
    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            if board[i][j] == 0:
                return i, j
    return None

def DFS(board):
    stack = [board]

    while stack:
        current_board = stack.pop()

        empty_position = find_empty_location(current_board)
        if empty_position is None:
            # No empty positions left, the Sudoku is solved
            return current_board

        row, col = empty_position
        neighbors = get_neighbors_dfs(current_board, row, col)
        for neighbor_board in neighbors:
            stack.append(neighbor_board)

    # No solution found
    return None

def get_neighbors_dfs(board, row, col):
    neighbors = []
    
    # Get numbers already present in the row, column, and 3x3 sub-grid
    row_values = set(board[row])
    col_values = set(board[i][col] for i in range(PUZZLE_SIZE))
    start_row, start_col = SUB_GRID * (row // SUB_GRID), SUB_GRID * (col // SUB_GRID)
    square_values = set(board[i][j] for i in range(start_row, start_row + SUB_GRID) for j in range(start_col, start_col + SUB_GRID))
    
    # Get unique numbers available for this cell
    unique_numbers = set(range(1, 10)) - (row_values | col_values | square_values)
    
    # Generate neighbor nodes with priorities based on uniqueness
    for num in unique_numbers:
        new_board = [row[:] for row in board]
        new_board[row][col] = num
        neighbors.append((new_board))
    
    return neighbors

def get_neighbors_bestfs(board):
    neighbors = []
    row_new = 0
    col_new = 0
    unique_numbers = set(range(20))
    
    for row in range(PUZZLE_SIZE):
        for col in range(PUZZLE_SIZE):
            if (board[row][col] != 0):
                continue
            # Get numbers already present in the row, column, and 3x3 sub-grid
            row_values = set(board[row])
            col_values = set(board[i][col] for i in range(PUZZLE_SIZE))
            start_row, start_col = SUB_GRID * (row // SUB_GRID), SUB_GRID * (col // SUB_GRID)
            square_values = set(board[i][j] for i in range(start_row, start_row + SUB_GRID) for j in range(start_col, start_col + SUB_GRID))
            
            # Get unique numbers available for this cell
            tmp = set(range(1, 10)) - (row_values | col_values | square_values)
            if (len(tmp) < len(unique_numbers)):
                unique_numbers = tmp
                row_new = row
                col_new = col
    
    # Generate neighbor nodes with priorities based on uniqueness
    for num in unique_numbers:
        new_board = [row[:] for row in board]
        new_board[row_new][col_new] = num
        # priority = len(unique_numbers)  # Higher priority for cells with fewer choices
        neighbors.append((new_board))
    return neighbors

def best_first_search(board):
    stack = [board]
    
    while stack:
        current_board = stack.pop()
        
        empty_location = find_empty_location(current_board)
        if not empty_location:
            return current_board
        
        # Get neighbor nodes and add them to the priority queue
        neighbors = get_neighbors_bestfs(current_board)
        for neighbor_board in neighbors:
            stack.append(neighbor_board)
    
    return None

def valid_row(row, grid):
    s = set()
    for i in range(PUZZLE_SIZE):
        # Checking for values outside 0 and 9;
        # 0 is considered valid because it
        # denotes an empty cell.
        # Removing zeros and the checking for values and
        # outside 1 and 9 is another way of doing
        # the same thing.
        if grid[row][i] < 0 or grid[row][i] > PUZZLE_SIZE:
            print("Invalid value")
            return -1
        else:
            # Checking for repeated values.
            if grid[row][i] != 0:
                if grid[row][i] in s:
                    return 0
                else:
                    s.add(grid[row][i])
    return 1

# Function to check if a given column is valid. It will return:
# -1 if the column contains an invalid value
# 0 if the column contains repeated values
# 1 is the column is valid.
def valid_col(col, grid):
    s = set()
    for i in range(PUZZLE_SIZE):
        # Checking for values outside 0 and 9;
        # 0 is considered valid because it
        # denotes an empty cell.
        # Removing zeros and the checking for values and
        # outside 1 and 9 is another way of doing
        # the same thing.
        if grid[i][col] < 0 or grid[i][col] > PUZZLE_SIZE:
            print("Invalid value")
            return -1
        else:
            # Checking for repeated values.
            if grid[i][col] != 0:
                if grid[i][col] in s:
                    return 0
                else:
                    s.add(grid[i][col])
    return 1

# Function to check if all the subsquares are valid. It will return:
# -1 if a subsquare contains an invalid value
# 0 if a subsquare contains repeated values
# 1 if the subsquares are valid.
def valid_subsquares(grid):
    for row in range(0, PUZZLE_SIZE, SUB_GRID):
        for col in range(0, PUZZLE_SIZE, SUB_GRID):
            s = set()
            for r in range(row, row + SUB_GRID):
                for c in range(col, col + SUB_GRID):
                    # Checking for values outside 0 and 9;
                    # 0 is considered valid because it
                    # denotes an empty cell.
                    # Removing zeros and the checking for values and
                    # outside 1 and 9 is another way of doing
                    # the same thing.
                    if grid[r][c] < 0 or grid[r][c] > PUZZLE_SIZE:
                        print("Invalid value")
                        return -1
                    else:
                        # Checking for repeated values
                        if grid[r][c] != 0:
                            if grid[r][c] in s:
                                return 0
                            else:
                                s.add(grid[r][c])
    return 1

# Function to check if the board invalid.
def valid_board(grid):
    for i in range(PUZZLE_SIZE):
        res1 = valid_row(i, grid)
        res2 = valid_col(i, grid)
        # If a row or a column is invalid, then the board is invalid.
        if res1 < 1 or res2 < 1:
            print("The board is invalid")
            return False
    # if any one the subsquares is invalid, then the board is invalid.
    res3 = valid_subsquares(grid)
    if res3 < 1:
        print("The board is invalid")
        return False
    else:
        print("The board is valid")
        return True

def solve_puzzle():
    filename = 'sudoku_puzzles.txt'  # Change this to the path of your file
    puzzles = read_matrix_from_file(filename)
    
    start_time = time.time()
    peak_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    for puzzle in puzzles:
        solved_board = DFS((puzzle))
        if solved_board and valid_board(puzzle):
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