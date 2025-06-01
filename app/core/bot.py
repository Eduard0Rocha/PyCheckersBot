from core.utils import is_request_valid

def evaluate_board(board: list[list[str]], player: str) -> int:
    """
    Evaluates the board state for the given player.

    Args:
        board (list[list[str]]): The 8x8 checkers board.
        player (str): The player to evaluate for ("r" or "b").

    Returns:
        int: Evaluation score. Positive = good for player, negative = bad.
    """
    # NOTE: These values can be adjusted to fine-tune evaluation performance
    piece_values = {
        "r": 1, "R": 2,  # red regular and red king
        "b": -1, "B": -2  # black regular and black king
    }

    # Compute the score for the player Red
    score = 0
    for row in board:
        for cell in row:
            score += piece_values.get(cell, 0)

    # If the player is Red, return the score, otherwise return the negated score (from Red's perspective)
    return score if player == "r" else -score

def get_legal_moves(board: list[list[str]], player: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Returns all legal moves for the given player on a checkers board.

    Legal moves include:
    - Simple diagonal moves into adjacent dark empty squares.
    - Single capture moves (jumping over an opponent piece).

    Each move is represented as a tuple:
        ((from_row, from_col), (to_row, to_col))

    Args:
        board (list[list[str]]): The 8x8 game board.
        player (str): The current player ('r' for red, 'b' for black).

    Returns:
        list[tuple[tuple[int, int], tuple[int, int]]]: A list of legal move tuples.
    """
    # Define movement directions for each piece type.
    # Regular pieces move forward; kings can move in all four diagonal directions.
    directions = {
        'r': [(-1, -1), (-1, 1)],                           # Red moves up
        'b': [(1, -1), (1, 1)],                             # Black moves down
        'R': [(-1, -1), (-1, 1), (1, -1), (1, 1)],          # Red king
        'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)]           # Black king
    }

    # Define which characters represent opponent pieces
    opponent_pieces = {'b', 'B'} if player == 'r' else {'r', 'R'}
    
    # Define which characters belong to the current player
    player_pieces = {'r', 'R'} if player == 'r' else {'b', 'B'}

    legal_moves = []       # Stores regular diagonal moves
    capture_moves = []     # Stores jump moves (captures)

    # Traverse the entire board
    for i in range(8):
        for j in range(8):
            piece = board[i][j]

            # Skip empty cells and opponent pieces
            if piece not in player_pieces:
                continue

            # Get allowed directions for this specific piece
            for dx, dy in directions[piece]:
                ni, nj = i + dx, j + dy

                # ---- SIMPLE MOVE ----
                # Check if target square is within bounds and empty
                if 0 <= ni < 8 and 0 <= nj < 8 and board[ni][nj] == ' ':
                    legal_moves.append(((i, j), (ni, nj)))

                # ---- CAPTURE MOVE ----
                # Coordinates of landing square after a jump
                ci, cj = i + 2 * dx, j + 2 * dy

                # Check if in bounds, there's an opponent in between, and destination is empty
                if (
                    0 <= ci < 8 and 0 <= cj < 8 and        # landing square within board
                    board[ni][nj] in opponent_pieces and   # opponent piece in between
                    board[ci][cj] == ' '                   # landing square is empty
                ):
                    capture_moves.append(((i, j), (ci, cj)))

    # According to standard rules, if any capture is possible, it must be taken
    return capture_moves if capture_moves else legal_moves

# TODO: documentation and comments
def play(board: list[list[str]], player: str, depth: int = 3):

    if not is_request_valid(board, player, depth):
        return {"error": "invalid request"}

    return {"move": "not implemented"} # TODO

# Limit exports to only the play function
__all__ = ["play"]