
def validate_board(board: list[list[str]]) -> bool:
    """
    Validates an 8x8 board for allowed characters.

    Args:
        board (list[list[str]]): A two-dimensional list representing the board.

    Returns:
        bool: True if the board is valid, False otherwise.
    """
    # Check if the board has exactly 8 rows
    if len(board) != 8:
        return False

    # Check if each row has exactly 8 columns
    if any(len(row) != 8 for row in board):
        return False

    # Define the set of allowed characters
    valid_chars = {'r', 'R', 'b', 'B', ' '}

    red_pieces = 0
    black_pieces = 0

    for i in range(8):
        for j in range(8):
            cell = board[i][j]
            if cell not in valid_chars:
                return False

            # Pieces must only be on dark squares (i + j) % 2 == 1
            if cell in {'r', 'R', 'b', 'B'} and (i + j) % 2 == 0:
                return False

            # Count pieces
            if cell in {'r', 'R'}:
                red_pieces += 1
            elif cell in {'b', 'B'}:
                black_pieces += 1

    # Check piece count limits
    if red_pieces > 12 or black_pieces > 12:
        return False

    return True

def validate_player(player: str) -> bool:
    """
    Validates if the player identifier is valid.

    Args:
        player (str): The player identifier ('r' or 'b').

    Returns:
        bool: True if the player is valid, False otherwise.
    """
    return player in {'r', 'b'}

def validate_depth(depth: int) -> bool:
    """
    Validates the search depth parameter.

    Args:
        depth (int): The depth value.

    Returns:
        bool: True if the depth is non-negative, False otherwise.
    """
    return depth >= 0


def is_request_valid(board: list[list[str]], player: str, depth: int) -> bool:
    """
    Validates a move request consisting of a board state, a player, and a depth value.

    Args:
        board (list[list[str]]): The checkers board.
        player (str): The current player ('r' or 'b').
        depth (int): The search depth.

    Returns:
        bool: True if all parameters are valid, False otherwise.
    """
    return (
        validate_board(board) and
        validate_player(player) and
        validate_depth(depth)
    )

def print_board(board: list[list[str]]) -> None:
    """
    Prints the checkers board in a human-readable format for debugging.

    Args:
        board (list[list[str]]): The 8x8 board to print.
    """
    print("   " + " ".join(str(col) for col in range(8)))  # Column headers
    print(" +" + "-" * 17 + "+")  # Top border
    for i, row in enumerate(board):
        row_str = " ".join(cell if cell != ' ' else '.' for cell in row)
        print(f"{i}| {row_str} |")  # Row index and row contents
    print(" +" + "-" * 17 + "+")  # Bottom border

# Limit exports to only the is_request_valid and print_board functions
__all__ = ["is_request_valid", "print_board"]