import random

def generate_sudoku(p):
    if p <= 0 or p >= 1:
        raise ValueError("p must be in the range (0, 1)")
    
    # Empty 9x9 matrix
    matrix = [[0 for _ in range(9)] for _ in range(9)]
    
    # Fill diagonal blocks
    for i in range(0, 9, 3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for j in range(3):
            for k in range(3):
                matrix[i+j][i+k] = nums.pop()
    
    # Solve the puzzle
    solve_sudoku(matrix)
    
    # Randomly remove cells based on proportion p
    num_to_remove = int(81 * (1 - p))
    cells = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(cells)
    for _ in range(num_to_remove):
        i, j = cells.pop()
        matrix[i][j] = 0
    
    return matrix

def solve_sudoku(matrix):
    def is_valid(num, row, col):
        # Check row
        for j in range(9):
            if matrix[row][j] == num:
                return False
        # Check column
        for i in range(9):
            if matrix[i][col] == num:
                return False
        # Check subgrid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if matrix[start_row + i][start_col + j] == num:
                    return False
        return True

    def solve():
        for i in range(9):
            for j in range(9):
                if matrix[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(num, i, j):
                            matrix[i][j] = num
                            if solve():
                                return True
                            matrix[i][j] = 0
                    return False
        return True

    solve()

# Example usage:
p = 0.05  # Proportion of fixed cells
while p < 1:
    sudoku_matrix = generate_sudoku(p)
    for row in sudoku_matrix:
        print(row)
    print()
    p = p + 0.05