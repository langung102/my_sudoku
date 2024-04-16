import random
import math

PUZZLE_SIZE = 9
SUB_GRID = int(math.sqrt(PUZZLE_SIZE))

def generate_sudoku(p):
    if p < 0 or p > 1:
        raise ValueError("p must be in the range (0, 1)")
    
    # Empty 9x9 matrix
    matrix = [[0 for _ in range(PUZZLE_SIZE)] for _ in range(PUZZLE_SIZE)]
    
    # Fill diagonal blocks
    for i in range(0, PUZZLE_SIZE, SUB_GRID):
        nums = list(range(1, PUZZLE_SIZE+1))
        random.shuffle(nums)
        for j in range(SUB_GRID):
            for k in range(SUB_GRID):
                matrix[i+j][i+k] = nums.pop()
    
    # Solve the puzzle
    solve_sudoku(matrix)
    
    # Randomly remove cells based on proportion p
    num_to_remove = int(PUZZLE_SIZE*PUZZLE_SIZE * (1 - p))
    cells = [(i, j) for i in range(PUZZLE_SIZE) for j in range(PUZZLE_SIZE)]
    random.shuffle(cells)
    for _ in range(num_to_remove):
        i, j = cells.pop()
        matrix[i][j] = 0
    
    return matrix

def solve_sudoku(matrix):
    def is_valid(num, row, col):
        # Check row
        for j in range(PUZZLE_SIZE):
            if matrix[row][j] == num:
                return False
        # Check column
        for i in range(PUZZLE_SIZE):
            if matrix[i][col] == num:
                return False
        # Check subgrid
        start_row, start_col = SUB_GRID * (row // SUB_GRID), SUB_GRID * (col // SUB_GRID)
        for i in range(SUB_GRID):
            for j in range(SUB_GRID):
                if matrix[start_row + i][start_col + j] == num:
                    return False
        return True

    def solve():
        for i in range(PUZZLE_SIZE):
            for j in range(PUZZLE_SIZE):
                if matrix[i][j] == 0:
                    for num in range(1, PUZZLE_SIZE+1):
                        if is_valid(num, i, j):
                            matrix[i][j] = num
                            if solve():
                                return True
                            matrix[i][j] = 0
                    return False
        return True

    solve()