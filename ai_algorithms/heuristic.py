
from game_rules import constants as c
import numpy as np

def calculate_board_score(board: np.ndarray, piece: int, opponent_piece: int) -> int:
    score = 0

    # Check horizontal
    for col in range(c.COLUMNS - 3):
        for r in range(c.ROWS):
            segment = [board[r][col + i] for i in range(4)]
            score += weights(segment, piece, opponent_piece)

    # Check vertical
    for col in range(c.COLUMNS):
        for r in range(c.ROWS - 3):
            segment = [board[r + i][col] for i in range(4)]
            score += weights(segment, piece, opponent_piece)

    # Check ascending diagonal
    for col in range(c.COLUMNS - 3):
        for r in range(c.ROWS - 3):
            segment = [board[r + i][col + i] for i in range(4)]
            score += weights(segment, piece, opponent_piece)

    # Check descending diagonal
    for col in range(c.COLUMNS - 3):
        for r in range(3, c.ROWS):
            segment = [board[r - i][col + i] for i in range(4)]
            score += weights(segment, piece, opponent_piece)

    return score


def weights(segment: list, piece: int, opponent_piece: int) -> int:
    if piece in segment and opponent_piece in segment: return 0
    if segment.count(piece) == 1: return 1
    if segment.count(piece) == 2: return 10
    if segment.count(piece) == 3: return 50
    if segment.count(piece) == 4: return 1000
    if segment.count(opponent_piece) == 1: return -1
    if segment.count(opponent_piece) == 2: return -10
    if segment.count(opponent_piece) == 3: return -50
    if segment.count(opponent_piece) == 4: return -2000
    return 0