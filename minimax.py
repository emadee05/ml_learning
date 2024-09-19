import numpy as np

player = 'x'
ai = 'o'

def create_board():
    return [['-' for _ in range(3)] for _ in range(3)]

def print_board(board):
    for row in board:
        print(' | '.join(row))
        print('-----')

def is_full(board): 
    for i in range(3):
        for j in range(3):
            if(board[i][j] == '-'):
                return False
    return True

def check_for_win(board, player):
    for row in range(3):
        if board[row][0] == player and board[row][1] == player and board[row][2] == player:
            return True
    for col in range(3):
        if board[0][col] == player and board[1][col] == player and board[2][col] == player:
            return True
    
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True
    if board[0][2] == player and board[1][1] == player and board[2][0] == player:
        return True
    return False

def player_result(board):
    if check_for_win(board, ai):
        return 1
    if check_for_win(board, player):
        return -1
    else:
        return 0
    
def minimax_algorithm(board, depth, alpha, beta, maximize):
    score = player_result(board)

    if score != 0 or is_full(board):
        return score
    
    if maximize: 
        best = -np.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == '-':
                    board[i][j] = ai
                    score = minimax_algorithm(board, depth+1, alpha, beta, False)
                    board[i][j] = '-'
                    best = max(score, best)
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break
        return best
    else:
        best = np.inf 
        for i in range(3):
            for j in range(3):
                if board[i][j] == '-':
                    board[i][j] = player
                    score = minimax_algorithm(board, depth+1, alpha, beta, True)
                    board[i][j] = '-'
                    best = min(score, best)
                    alpha = min(alpha, best)
                    if beta <= alpha:
                        break
        return best

def best_move(board):
    best_move_value = -np.inf
    best_move = (-10, -10)
    for i in range(3):
        for j in range(3):
            if (board[i][j] == '-'):
                board[i][j] = ai
                move_value = minimax_algorithm(board, 0, -np.inf, np.inf, False)
                board[i][j] = '-'
                if move_value > best_move_value:
                    best_move_value = move_value
                    best_move = (i, j)
    return best_move

def main():
    board = create_board()
    game_over = False
    print("Welcome to Tic Tac Toe!")
    print_board(board)

    while not game_over:
        move_row = int(input("Enter your move (row)"))
        move_col = int(input("Enter your move (col)"))
        row,col = (move_row, move_col)
        if board[row][col] == '-':
            board[row][col] = player
            print_board(board)
        else:
            print("Invalid move!")
        if check_for_win(board, player):
            print("You win!")
            game_over = True
        elif is_full(board):
            print("Tie!")
            game_over = True

        print("AI Turn")
        ai_move = best_move(board)
        board[ai_move[0]][ai_move[1]] = ai
        print_board(board)

        if check_for_win(board, ai):
            print("AI Won")
            game_over = True
        elif is_full(board):
            print("Tie")
            game_over = True

if __name__ == "__main__":
    main()