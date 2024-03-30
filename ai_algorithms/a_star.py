import numpy as np
from game_rules import constants as c 
from game_rules import game_logic as game
from ai_algorithms import heuristic as h


def a_star(board: np.ndarray, ai_piece: int, opponent_piece: int) -> int:
    best_score = float('-inf')
    best_move = -1
    for col in game.available_moves(board):
        simulated_board = game.simulate_move(board, ai_piece, col)
        cur_score = h.calculate_board_score(simulated_board, ai_piece, opponent_piece)
        if cur_score > best_score:
            best_move = col
            best_score = cur_score
    return best_move


def a_star_adversarial(board: np.ndarray, ai_piece: int, opponent_piece: int) -> int:
    move_score = float('-inf')
    best_move = -1
    best_opponent = 0
    possible_moves = game.available_moves(board)
    if len(possible_moves) == 1: return possible_moves[0]

    for col in possible_moves:
        simulated_board = game.simulate_move(board, ai_piece, col)
        opponent_col = a_star(simulated_board, opponent_piece, ai_piece)  
        opponent_simulated_board = game.simulate_move(simulated_board, opponent_piece, opponent_col)
        cur_score = h.calculate_board_score(opponent_simulated_board, ai_piece, opponent_piece)
        if cur_score > move_score:
            best_opponent = opponent_col + 1
            best_move = col
            move_score = cur_score
   
    print("Dica de jogada: coluna " + str(best_opponent))
    return best_move





