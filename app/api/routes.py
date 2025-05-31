from fastapi import APIRouter
from api.models import MoveRequest
from core.bot import play as bot_play

router = APIRouter()

# TODO: documentation for the route (add example of request)
@router.post("/play")
def play(request: MoveRequest):
    result = bot_play(request.board, request.player, request.depth)
    return result
