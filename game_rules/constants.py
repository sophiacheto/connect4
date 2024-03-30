# Colors
BOARD_COLOR = (81, 103, 171) # Purple
BACKGROUND_COLOR = (250,250,250) # White 
SHADOW_COLOR = (170, 170, 170) # Gray
PLAYER_COLOR = (167, 212, 212) # Blue
IA_COLOR = (242, 133, 174) # Pink
BUTTON_TEXT_COLOR = (0,0,0) # Black

PIECES_COLORS = [BACKGROUND_COLOR, PLAYER_COLOR, IA_COLOR]
HUMAN_PIECE = 1
AI_PIECE = 2

# Constants for the data matrix
ROWS = 6
COLUMNS = 7

# Constants for the board image
SQUARESIZE = 100    # size of each square that will divide the screen 
RADIUS = 43     # radius of each player piece

# Constants for the interface (scren)
WIDTH = (4+COLUMNS) * SQUARESIZE    # width of the screen = board + 2 empty columns on each side of the screen
HEIGHT = (ROWS+2) * SQUARESIZE    # height of the screen = board + 1 empty row under and above the board
   
        
       