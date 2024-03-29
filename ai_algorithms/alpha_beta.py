from game_rules import constants as c, game_logic as game
from ai_algorithms import heuristic as h
import time, logging, numpy as np


NODES_VISITED = 1
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def alpha_beta(board: np.ndarray):
    start_time = time.time()
    global NODES_VISITED
    NODES_VISITED = 1
    children = get_children(board, c.AI_PIECE)
    melhor_jogada = -1
    melhor_score = float('-inf')
    for (child, col) in children:
        score = calcular(child, 0, float('-inf'), float('+inf'), 4, False)
        if score > melhor_score:
            melhor_score = score
            melhor_jogada = col
    end_time = time.time()
    logging.info(f"Tempo de resposta = {end_time-start_time}")
    return melhor_jogada



def calcular(board: np.ndarray, depth: int, alpha: int, beta: int, depth_limit: int, maximizing):
    if depth == depth_limit or game.winning_move(board, 1) or game.winning_move(board, 2):
        return h.calculate_board_score(board, c.AI_PIECE, c.HUMAN_PIECE)

    
    if maximizing:
        maxEval = float('-inf')
        children = get_children(board, c.AI_PIECE)
        for (child, col) in children:
            eval = calcular(child, depth+1, alpha, beta, depth_limit, False)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    
    else:
        minEval = float('+inf')
        children = get_children(board, c.HUMAN_PIECE)
        for (child, col) in children:
            eval = calcular(child, depth+1, alpha, beta, depth_limit, True)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval



def get_children(board, piece) -> None:
        children = []
        if game.available_moves(board) == -1: return  
        for col in game.available_moves(board):  
            copy_board = game.simulate_move(board, piece, col)   
            children.append((copy_board, col)) 
        return children
