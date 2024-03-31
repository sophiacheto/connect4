from game_rules import constants as c, game_logic as game
from ai_algorithms import heuristic as h
import numpy as np


def alpha_beta(board: np.ndarray):
    """Return the best column chose by alpha_beta algorithm"""
    children = get_children(board, c.AI_PIECE)
    depth_limit = 5    # qtd de níveis abaixo do atual que serão calculados
    best_move = -1
    best_score = float('-inf')
    for (child, col) in children:
        if game.winning_move(child, c.AI_PIECE):  # se alguma jogada já for vitoriosa, não calcula outras
            best_move = col                                                   
            break
        score = calculate(child, 1, float('-inf'), float('+inf'), depth_limit, False)
        if score > best_score:
            best_score = score
            best_move = col
    return best_move




def calculate(board: np.ndarray, depth: int, alpha: int, beta: int, depth_limit: int, maximizing):
    """Return the accumulated score for the current move"""

    if depth == depth_limit or game.winning_move(board, 1) or game.winning_move(board, 2) or game.is_game_tied(board):
        return h.calculate_board_score(board, c.AI_PIECE, c.HUMAN_PIECE)

    if maximizing:
        maxEval = float('-inf')
        children = get_children(board, c.AI_PIECE)
        for (child, _) in children:
            eval = calculate(child, depth+1, alpha, beta, depth_limit, False)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    
    else:
        minEval = float('+inf')
        children = get_children(board, c.HUMAN_PIECE)
        for (child, _) in children:
            eval = calculate(child, depth+1, alpha, beta, depth_limit, True)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval



def get_children(board, piece) -> None:
    """Return children of the actual state board"""
    children = []
    if game.available_moves(board) == -1: return children
    for col in game.available_moves(board):  
        copy_board = game.simulate_move(board, piece, col)   
        children.append((copy_board, col)) 
    return children


