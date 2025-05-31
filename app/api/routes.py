from fastapi import APIRouter
from api.models import MoveRequest

router = APIRouter()

# TODO: documentation for the route (add example of request)
@router.post("/play")
def play(request: MoveRequest):
    
    return {} # TODO
