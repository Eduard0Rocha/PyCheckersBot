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



# TODO: documentation and comments
def play(board: list[list[str]], player: str, depth: int = 3):

    if not is_request_valid(board, player, depth):
        return {"error": "invalid request"}

    return {"move": "not implemented"} # TODO

# Limit exports to only the play function
__all__ = ["play"]