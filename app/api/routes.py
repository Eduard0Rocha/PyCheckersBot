from fastapi import APIRouter
from api.models import MoveRequest
from core.bot import play as bot_play

router = APIRouter()

@router.post("/play", summary="Get best move from bot", response_description="Best move sequence according to the bot's evaluation")
def play(request: MoveRequest):
    """
    Receives the current board state, player to move, and optional parameters.

    - **description** (optional): A textual description of the board or scenario.
    - **board**: 8x8 list representing the board squares.
    - **player**: current player to move ('r' or 'b').
    - **depth** (optional): search depth for the minimax algorithm (minimum 1). Default is 3 if omitted.

    Returns the best move sequence calculated by the AI bot.
    """
    result = bot_play(request.board, request.player, request.depth)
    return result
