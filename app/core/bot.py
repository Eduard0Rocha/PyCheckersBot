import math
from core.utils import is_request_valid, print_board

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

def check_game_over(
    board: list[list[str]], 
    player: str, 
    legal_moves: list[tuple[tuple[int, int], tuple[int, int]]] | None = None
) -> dict:
    """
    Checks if the game has ended, optionally using precomputed legal moves.

    Args:
        board (list[list[str]]): The current board state.
        player (str): The current player ('r' or 'b').
        legal_moves (list or None): Precomputed legal moves for the player.

    Returns:
        dict: {
            "game_over": bool,
            "winner": Optional[str]  # 'r', 'b', or None
        }
    """
    opponent = 'b' if player == 'r' else 'r'
    player_pieces = {'r', 'R'} if player == 'r' else {'b', 'B'}
    opponent_pieces = {'b', 'B'} if player == 'r' else {'r', 'R'}

    # Check if the player or opponent has no pieces
    player_has_pieces = any(cell in player_pieces for row in board for cell in row)
    opponent_has_pieces = any(cell in opponent_pieces for row in board for cell in row)

    if not player_has_pieces:
        return {"game_over": True, "winner": opponent}
    if not opponent_has_pieces:
        return {"game_over": True, "winner": player}

    # Use provided legal moves or generate them
    if legal_moves is None:
        legal_moves = get_legal_moves(board, player)

    # If no legal moves, the player loses
    if not legal_moves:
        return {"game_over": True, "winner": opponent}

    return {"game_over": False, "winner": None}

def apply_move(board: list[list[str]], move: tuple[tuple[int, int], tuple[int, int]]) -> list[list[str]]:
    """
    Applies a move to the given board and returns the resulting new board.

    A move can be:
    - A simple diagonal move to an adjacent empty square.
    - A capture move, where the opponent's piece is removed after a jump.

    The function also handles piece promotion:
    - A red piece ('r') becomes a king ('R') if it reaches the top row (0).
    - A black piece ('b') becomes a king ('B') if it reaches the bottom row (7).

    Args:
        board (list[list[str]]): The current 8x8 board state.
        move (tuple): A tuple representing the move:
            ((from_row, from_col), (to_row, to_col))

    Returns:
        list[list[str]]: A new board state after applying the move.
    """
    from_row, from_col = move[0]
    to_row, to_col = move[1]

    # Make a deep copy of the board to avoid modifying the original
    new_board = [row.copy() for row in board]

    # Move the piece from its original position
    piece = new_board[from_row][from_col]
    new_board[from_row][from_col] = ' '
    new_board[to_row][to_col] = piece

    # If it's a capture move (distance of 2), remove the captured opponent piece
    if abs(to_row - from_row) == 2:
        mid_row = (from_row + to_row) // 2
        mid_col = (from_col + to_col) // 2
        new_board[mid_row][mid_col] = ' '

    # Handle promotion to king if reaching the last row
    if piece == 'r' and to_row == 0:
        new_board[to_row][to_col] = 'R'
    elif piece == 'b' and to_row == 7:
        new_board[to_row][to_col] = 'B'

    return new_board

def get_additional_captures(
    board: list[list[str]], 
    start_pos: tuple[int,int], 
    piece: str
) -> list[list[tuple[tuple[int, int], tuple[int, int]]]]:
    """
    Recursively finds all possible chained capture sequences starting from a given position.

    A capture sequence is a list of jump moves a piece can make in succession,
    each jumping over an opponent's piece and landing on an empty square.

    Args:
        board (list[list[str]]): The current state of the 8x8 checkers board.
        start_pos (tuple[int, int]): The starting position of the piece.
        piece (str): The character representing the piece ('r', 'R', 'b', or 'B').

    Returns:
        list[list[tuple]]: A list of capture sequences. Each sequence is a list of moves:
            [((from_row, from_col), (to_row, to_col)), ...]
    """
    directions = {
        'r': [(-1, -1), (-1, 1)],  # Red moves upward
        'b': [(1, -1), (1, 1)],    # Black moves downward
        'R': [(-1, -1), (-1, 1), (1, -1), (1, 1)],  # Red king
        'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)]   # Black king
    }

    # Determine which pieces are opponents
    opponent_pieces = {'b', 'B'} if piece.lower() == 'r' else {'r', 'R'}
    
    # Stores all complete capture paths found
    sequences = []

    def dfs(board_state: list[list[str]], path: list[tuple[tuple[int, int], tuple[int, int]]], row: int, col: int, current_piece: str):
        """
        Recursive DFS (depth-first search) to find all valid capture chains from current position.

        Args:
            board_state (list[list[str]]): Current board state after previous moves.
            path (list): Sequence of moves accumulated so far.
            row (int): Current row of the piece.
            col (int): Current column of the piece.
        """

        found = False  # Flag to track if any capture was made at this level

        for dx, dy in directions[current_piece]:
            mid_r, mid_c = row + dx, col + dy
            end_r, end_c = row + 2 * dx, col + 2 * dy

            # Ensure positions are within board bounds
            if (
                0 <= mid_r < 8 and 0 <= mid_c < 8 and
                0 <= end_r < 8 and 0 <= end_c < 8 and
                board_state[mid_r][mid_c] in opponent_pieces and
                board_state[end_r][end_c] == ' '
            ):
                # Clone the board to simulate this move
                new_board = [r.copy() for r in board_state]

                # Promote the piece if it reaches the last row
                new_piece = current_piece
                if current_piece == 'r' and end_r == 0:
                    new_piece = 'R'  # Promote red pawn to king
                elif current_piece == 'b' and end_r == 7:
                    new_piece = 'B'  # Promote black pawn to king

                # Remove the original piece and the captured opponent piece
                new_board[row][col] = ' '
                new_board[mid_r][mid_c] = ' '

                # Place the promoted piece (or original if no promotion) on the destination square
                new_board[end_r][end_c] = new_piece
                
                # Add this jump to the path and continue recursively
                dfs(new_board, path + [((row, col), (end_r, end_c))], end_r, end_c, new_piece)
                found = True

        # If no more captures available, save the current path if it's non-empty
        if not found and path:
            sequences.append(path)

    # Start DFS from the given position
    dfs(board, [], *start_pos, piece)
    return sequences


def minimax(
    board: list[list[str]], 
    player: str, 
    depth: int, 
    alpha: float, 
    beta: float, 
    maximizing_player: bool
) -> tuple[int, list[tuple[tuple[int, int], tuple[int, int]]]]:
    """
    Minimax algorithm with alpha-beta pruning for checkers.

    Args:
        board (list[list[str]]): The current 8x8 game board.
        player (str): The player whose move is being evaluated ('r' or 'b').
        depth (int): The maximum depth of the search tree.
        alpha (float): The best score that the maximizer can guarantee so far.
        beta (float): The best score that the minimizer can guarantee so far.
        maximizing_player (bool): True if we are maximizing for the current player.

    Returns:
        tuple:
            - int: The evaluation score of the board.
            - list: The best move sequence as a list of move tuples.
    """
    # Check if the game is over or if we've reached the maximum depth
    game_status = check_game_over(board, player)
    if depth == 0 or game_status["game_over"]:
        # Evaluate the current board for the player and return
        return evaluate_board(board, player if maximizing_player else ('r' if player == 'b' else 'b')), None

    # Generate all possible legal moves for the current player
    legal_moves = get_legal_moves(board, player)
    best_move = None  # This will store the best move sequence

    if maximizing_player:

        max_eval = -math.inf  # Initialize with worst possible value for maximizer

        # Explore each possible move
        for move in legal_moves:

            new_board = apply_move(board, move)  # Apply the move and get the new board

            # Check if this was a capture (i.e., move over two rows)
            if abs(move[1][0] - move[0][0]) == 2:
                piece = new_board[move[1][0]][move[1][1]]

                additional_captures = get_additional_captures(new_board, move[1], piece)

                # Handle multi-jump sequences (chained captures)
                if additional_captures:

                    for sequence in additional_captures:

                        temp_board = new_board
                        for m in sequence:
                            temp_board = apply_move(temp_board, m)

                        # Recursive call for the same player (still in capture sequence)
                        eval_score, _ = minimax(temp_board, player, depth, alpha, beta, True)

                        # Update best evaluation if this move is better
                        if eval_score > max_eval:
                            max_eval = eval_score
                            best_move = [move] + sequence

                        # Alpha-beta pruning: update alpha and prune if necessary
                        alpha = max(alpha, eval_score)
                        if beta <= alpha:
                            break  # Prune branch
                    continue  # Skip normal recursion since captures were handled

            # Regular recursive call for next player's turn
            eval_score, _ = minimax(new_board, 'b' if player == 'r' else 'r', depth - 1, alpha, beta, False)

            # Update best evaluation and move if this one is better
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = [move]

            # Alpha-beta pruning
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Prune branch
        return max_eval, best_move

    else:

        min_eval = math.inf  # Initialize with worst possible value for minimizer

        for move in legal_moves:
            new_board = apply_move(board, move)

            # Check if this was a capture
            if abs(move[1][0] - move[0][0]) == 2:
                piece = board[move[0][0]][move[0][1]]
                additional_captures = get_additional_captures(new_board, move[1], piece)

                # Handle chained captures
                if additional_captures:
                    for sequence in additional_captures:
                        temp_board = new_board
                        for m in sequence:
                            temp_board = apply_move(temp_board, m)

                        # Recursive call, still minimizer's perspective
                        eval_score, _ = minimax(temp_board, player, depth, alpha, beta, False)

                        if eval_score < min_eval:
                            min_eval = eval_score
                            best_move = [move] + sequence

                        beta = min(beta, eval_score)
                        if beta <= alpha:
                            break  # Prune
                    continue

            # Normal move
            eval_score, _ = minimax(new_board, 'b' if player == 'r' else 'r', depth - 1, alpha, beta, True)

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = [move]

            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Prune
        return min_eval, best_move

def play(board: list[list[str]], player: str, depth: int = 3):
    """
    Given a board and current player, compute the best move sequence using minimax.

    Args:
        board (list[list[str]]): Current 8x8 checkers board state.
        player (str): The player to move ('r' or 'b').
        depth (int): Maximum depth for minimax search (default 3).

    Returns:
        dict: {
            "move": list of moves (each move is ((from_row, from_col), (to_row, to_col)))
        }
        or {"error": str} if input invalid or no moves available.
    """
    # Validate the request parameters and board state
    if not is_request_valid(board, player, depth):
        return {"error": "invalid request"}

    # Use minimax to get the best evaluation score and best move sequence
    _, best_move_sequence = minimax(board, player, depth, -math.inf, math.inf, True)

    if best_move_sequence is None or len(best_move_sequence) == 0:
        # No legal moves available, player loses or game over
        return {"error": "no moves available"}

    # Return the best move sequence found by minimax
    return {"move": best_move_sequence}


# Limit exports to only the play function
__all__ = ["play"]