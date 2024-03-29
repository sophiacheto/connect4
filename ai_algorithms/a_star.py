import numpy as np
from game_rules import constants as c 
from game_rules import game_logic as game
import logging
from ai_algorithms import heuristic as h

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

def a_star(board: np.ndarray, ai_piece: int, opponent_piece: int) -> int:
    best_score = float('-inf')
    best_move = -1
    for col in range(c.COLUMNS):
        if not game.is_valid(board, col): continue
        cur_score = 0
        simulated_board = game.simulate_move(board, ai_piece, col)
        cur_score = h.calculate_board_score(simulated_board, ai_piece, opponent_piece)
        if cur_score > best_score:
            best_move = col
            best_score = cur_score
    return best_move


def a_star_adversarial(board: np.ndarray, ai_piece: int, opponent_piece: int) -> int:
    move_score = float('-inf')
    best_move = -1
    best_opponent = 0;
    for col in range(c.COLUMNS):
        if not game.is_valid(board, col): continue
        cur_score = 0
        simulated_board = game.simulate_move(board, ai_piece, col)

        opponent_col = a_star(simulated_board, opponent_piece, ai_piece)  
        opponent_simulated_board = game.simulate_move(simulated_board, opponent_piece, opponent_col)
        cur_score = h.calculate_board_score(opponent_simulated_board, ai_piece, opponent_piece)

        if cur_score > move_score:
            best_opponent = opponent_col + 1
            best_move = col
            move_score = cur_score

    # logging.info(f"Score = {move_score}")    
    print("Próximo passo sugerido: coluna " + str(best_opponent))
    return best_move





