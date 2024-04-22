import time
import resource
from generator import generate_sudoku
import matplotlib.pyplot as plt
import numpy as np
import math
from multiprocessing import Process, Lock, Manager
import tracemalloc

PUZZLE_SIZE = 9
SUB_GRID = int(math.sqrt(PUZZLE_SIZE))
NUM_OF_CATEGORIES = 20
STEP = 1/NUM_OF_CATEGORIES
NUM_OF_INSTANCES = 10
NUM_OF_REPEAT = 5

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

def get_neighbors_gbfs(board):
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
    
    # Generate neighbor nodes
    for num in unique_numbers:
        new_board = [row[:] for row in board]
        new_board[row_new][col_new] = num
        neighbors.append((new_board))
    return neighbors

def greedy_best_first_search(board):
    stack = [board]
    
    while stack:
        current_board = stack.pop()
        
        empty_location = find_empty_location(current_board)
        if not empty_location:
            return current_board
        
        # Get neighbor nodes at spot have minumum number of candidates and add them to stack
        neighbors = get_neighbors_gbfs(current_board)
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
        # print("The board is invalid")
        return False
    else:
        # print("The board is valid")
        return True

lock = Lock()

def solve_puzzle(is_read_from_file=False, type_algorithm="DFS"):
    if (is_read_from_file):
        filename = 'sudoku_puzzles.txt'  # Change this to the path of your file
        puzzles = read_matrix_from_file(filename)
        
        start_time = time.time()
        tracemalloc.start()
        peak_memory_usage = tracemalloc.get_traced_memory()[0]
        
        for puzzle in puzzles:
            if (type_algorithm=="DFS"):
                solved_board = DFS((puzzle))
            elif (type_algorithm=="GBFS"):
                solved_board = greedy_best_first_search((puzzle))
            if solved_board and valid_board(puzzle):
                print("Solution:")
                for row in solved_board:
                    print(row)
                print("\n")
            else:
                print("No solution exists for this puzzle.")
        
        end_time = time.time()
        peak_memory_usage = tracemalloc.get_traced_memory()[0] - peak_memory_usage
        execution_time = end_time - start_time
        
        print("Total execution time:", execution_time, "seconds")
        print("Memory usage:", peak_memory_usage, "Bytes")
    else:
        manager = Manager()

        avg_execution_time_dfs = manager.list([0] * NUM_OF_CATEGORIES)
        avg_peak_memory_usage_dfs = manager.list([0] * NUM_OF_CATEGORIES)
        avg_execution_time_gbfs = manager.list([0] * NUM_OF_CATEGORIES)
        avg_peak_memory_usage_gbfs = manager.list([0] * NUM_OF_CATEGORIES)

        p = [0]*NUM_OF_CATEGORIES  # Proportion of fixed cells
        for i in range(0,NUM_OF_CATEGORIES): # different categories
            p[i] = i*STEP
            print("Iteration with p=", p[i])
            for j in range(0,NUM_OF_INSTANCES): # instances were created per categor
                puzzle = generate_sudoku(p[i])
                def repeat(share_avg_execution_time_dfs, share_avg_peak_memory_usage_dfs, share_avg_execution_time_gbfs, share_avg_peak_memory_usage_gbfs):
                    #DFS
                    start_time = time.time()
                    tracemalloc.start()
                    peak_memory_usage = tracemalloc.get_traced_memory()[0]
                    solved_board = DFS((puzzle))
                    peak_memory_usage = tracemalloc.get_traced_memory()[0] - peak_memory_usage
                    tracemalloc.stop()
                    end_time = time.time()
                    execution_time = end_time - start_time

                    if not solved_board or not valid_board(puzzle):
                        print("No solution exists for this puzzle.")
                    with lock:
                        share_avg_execution_time_dfs[i] += execution_time
                        share_avg_peak_memory_usage_dfs[i] += peak_memory_usage
                    #GBFS
                    start_time = time.time()
                    tracemalloc.start()
                    peak_memory_usage = tracemalloc.get_traced_memory()[0]
                    solved_board = greedy_best_first_search((puzzle))
                    peak_memory_usage = tracemalloc.get_traced_memory()[0] - peak_memory_usage
                    tracemalloc.stop()
                    end_time = time.time()
                    execution_time = end_time - start_time

                    if not solved_board or not valid_board(puzzle):
                        print("No solution exists for this puzzle.")
                    with lock:
                        share_avg_execution_time_gbfs[i] += execution_time
                        share_avg_peak_memory_usage_gbfs[i] += peak_memory_usage

                processes = []
                for n in range(NUM_OF_REPEAT):
                    process = Process(target=repeat, args=(avg_execution_time_dfs, avg_peak_memory_usage_dfs, avg_execution_time_gbfs, avg_peak_memory_usage_gbfs))
                    process.start()
                    processes.append(process)
    
                for process in processes:
                    process.join()

                avg_execution_time_dfs[i] = avg_execution_time_dfs[i]/NUM_OF_REPEAT
                avg_peak_memory_usage_dfs[i] = avg_peak_memory_usage_dfs[i]/NUM_OF_REPEAT
                avg_execution_time_gbfs[i] = avg_execution_time_gbfs[i]/NUM_OF_REPEAT
                avg_peak_memory_usage_gbfs[i] = avg_peak_memory_usage_gbfs[i]/NUM_OF_REPEAT

        open('output.txt', 'w').close()
        f = open("output.txt", "a")
        f.write("DFS:\n")
        f.write("Max execution time: " + str(max(avg_execution_time_dfs)) + " seconds\n")
        f.write("Max memory usage: " + str(max(avg_peak_memory_usage_dfs)) + " bytes\n")
        f.write("Average execution time: " + str(sum(avg_execution_time_dfs)/len(avg_execution_time_dfs)) + " seconds\n")
        f.write("Average memory usage: " + str(sum(avg_peak_memory_usage_dfs)/len(avg_peak_memory_usage_dfs)) + " bytes\n\n")
        f.write("GBFS:\n")
        f.write("Max execution time: " + str(max(avg_execution_time_gbfs)) + " seconds\n")
        f.write("Max memory usage: " + str(max(avg_peak_memory_usage_gbfs)) + " bytes\n")
        f.write("Average execution time: " + str(sum(avg_execution_time_gbfs)/len(avg_execution_time_gbfs)) + " seconds\n")
        f.write("Average memory usage: " + str(sum(avg_peak_memory_usage_gbfs)/len(avg_peak_memory_usage_gbfs)) + " bytes\n\n")
        f.close()
                    
        # print("DFS:")
        # print("Max execution time:", max(avg_execution_time_dfs), "seconds")
        # print("Max peak memory usage:", max(avg_peak_memory_usage_dfs), "kilobytes")
        # print("Average execution time:", sum(avg_execution_time_dfs)/len(avg_execution_time_dfs), "seconds")
        # print("Average peak memory usage:", sum(avg_peak_memory_usage_dfs)/len(avg_peak_memory_usage_dfs), "kilobytes")

        # print("GBFS:")
        # print("Max execution time:", max(avg_execution_time_gbfs), "seconds")
        # print("Max peak memory usage:", max(avg_peak_memory_usage_gbfs), "kilobytes")
        # print("Average execution time:", sum(avg_execution_time_gbfs)/len(avg_execution_time_gbfs), "seconds")
        # print("Average peak memory usage:", sum(avg_peak_memory_usage_gbfs)/len(avg_peak_memory_usage_gbfs), "kilobytes")
        # Create the bar chart
        plt.bar(p, avg_execution_time_dfs, width=0.03, color="blue") 
        plt.title("DFS")
        plt.xlabel('Proportion of fixed cells (%)')
        plt.ylabel('Average execution time (s)')
        plt.savefig('DFS_exe_time.png')
        # plt.show()
        plt.clf()

        plt.bar(p, avg_peak_memory_usage_dfs, width=0.03, color="blue")
        plt.title("DFS")
        plt.xlabel('Proportion of fixed cells (%)')
        plt.ylabel('Average memory usage (bytes)')
        plt.savefig('DFS_memory.png')
        # # plt.show()
        plt.clf()

        plt.bar(p, avg_execution_time_gbfs, width=0.03, color="blue")
        plt.title("GBFS")
        plt.xlabel('Proportion of fixed cells (%)')
        plt.ylabel('Average execution time (s)')
        plt.savefig('GBFS_exec_time.png')
        # plt.show()
        plt.clf()

        plt.bar(p, avg_peak_memory_usage_gbfs, width=0.03, color="blue")
        plt.title("GBFS")
        plt.xlabel('Proportion of fixed cells (%)')
        plt.ylabel('Average memory usage (bytes)')
        plt.savefig('GBFS_memory.png')
        # plt.show()
        plt.clf()

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

solve_puzzle(True, "DFS")
solve_puzzle(True, "GBFS")
solve_puzzle()