import game_rules.constants as c
import numpy as np
import math
import pygame
from game_rules.board import Board
from ai_algorithms import a_star as g, alpha_beta as a, mcts as m


def human_move(bd: Board, interface: any, board: np.ndarray, turn: int, event: any) -> bool:
	"""Set the column of human move"""
	col = get_human_column(interface, event)
	if not is_valid(board, col): return False 

	pygame.draw.rect(interface.screen, c.BACKGROUND_COLOR, (0,0, interface.width, interface.pixels-14))   
	make_move(bd, interface, board, turn, col)
	return True


def get_human_column(interface: any, event: any):
	"""Gets the column that the mouse selected"""
	posx = event.pos[0]
	col = int(math.floor(posx/interface.pixels)) - 2
	return col


def available_moves(board: np.ndarray) -> list | int:
	avaiable_moves = []
	for i in range(c.COLUMNS):
		if(board[5][i])==0:
			avaiable_moves.append(i)
	return avaiable_moves if len(avaiable_moves) > 0 else -1


def ai_move(bd: Board, interface: any, game_mode: int, board: np.ndarray, turn: int) -> int:
	"""Set the column of the AI move"""
	ai_column = get_ai_column(board, game_mode)
	game_over = make_move(bd, interface, board, turn, ai_column)
	return game_over
	

def get_ai_column(board: Board, game_mode: int) -> int:
	"""Select the chose ai algorithm to make a move"""
	chosen_column = 0
	if game_mode == 2:
		chosen_column = g.a_star(board, c.AI_PIECE, c.HUMAN_PIECE)
	elif game_mode == 3:
		chosen_column = g.a_star_adversarial(board, c.AI_PIECE, c.HUMAN_PIECE)
	elif game_mode == 4:
		chosen_column = a.alpha_beta(board)
	elif game_mode == 5:
		chosen_column = m.mcts(board)
	return chosen_column


def simulate_move(board: np.ndarray, piece: int, col: int) -> np.ndarray:
	"""Simulate a move in a copy of the board"""
	board_copy = board.copy()
	row = get_next_open_row(board_copy, col)
	drop_piece(board_copy, row, col, piece)
	return board_copy


def make_move(bd: Board, interface: any, board: np.ndarray, turn: int, move: int):
	"""Make the move and see if the move is a winning one"""

	row = get_next_open_row(board, move)
	drop_piece(board, row, move, turn)	# adds the new piece to the data matrix
	interface.draw_new_piece(row+1, move+2, turn) 	# adds new piece to the screen
	pygame.display.update()
	bd.print_board()

	return winning_move(board, turn) or is_game_tied(board)


def get_next_open_row(board: np.ndarray, col: int) -> int:
	"""Given a column, return the first row avaiable to set a piece"""
	for row in range(c.ROWS):
		if board[row][col] == 0:
			return row
	return -1


def drop_piece(board: np.ndarray, row: int, col: int, piece: int) -> None:
	"""Insert a piece into board on correct location"""
	board[row][col] = piece


def is_game_tied(board: np.ndarray) -> bool:
	"""Assert if the game is tied"""
	if winning_move(board, c.AI_PIECE) or winning_move(board, c.HUMAN_PIECE): return False
	for i in range(len(board)):
		for j in range(len(board[0])):
			if board[i][j]==0: return False
	return True


def is_valid(board: np.ndarray, col: int) -> bool:
	"""Analize if chosen column is valid"""
	if not 0 <= col < c.COLUMNS: return False
	row = get_next_open_row(board, col)
	return 0 <= row <= 5


def winning_move(board: np.ndarray, piece: int) -> bool:
	"""Return if the selected move will win the game"""
	def check_horizontal(board: np.ndarray, piece: int) -> bool:
		"""Check winning condition on horizontal lines"""
		for col in range(c.COLUMNS-3):
			for row in range(c.ROWS):
				if board[row][col] == piece and board[row][col+1] == piece and board[row][col+2] == piece and board[row][col+3] == piece:
					return True
			

	def check_vertical(board: np.ndarray, piece: int) -> bool:
		"""Check winning condition on vertical lines"""
		for col in range(c.COLUMNS):
			for row in range(c.ROWS-3):
				if board[row][col] == piece and board[row+1][col] == piece and board[row+2][col] == piece and board[row+3][col] == piece:
					return True
				

	def check_ascending_diagonal(board: np.ndarray, piece: int) -> bool:
		"""Check winning condition on ascending diagonal lines"""
		for col in range(c.COLUMNS-3):
			for row in range(c.ROWS-3):
				if board[row][col] == piece and board[row+1][col+1] == piece and board[row+2][col+2] == piece and board[row+3][col+3] == piece:
					return True
				

	def check_descending_diagonal(board: np.ndarray, piece: int) -> bool:
		"""Check winning condition on descending diagonal lines"""
		for col in range(c.COLUMNS-3):
			for row in range(3, c.ROWS):
				if board[row][col] == piece and board[row-1][col+1] == piece and board[row-2][col+2] == piece and board[row-3][col+3] == piece:
					return True
				
	return check_vertical(board, piece) or check_horizontal(board, piece) or check_ascending_diagonal(board, piece) or check_descending_diagonal(board, piece)
			
