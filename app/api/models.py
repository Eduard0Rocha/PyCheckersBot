from pydantic import BaseModel, Field
from typing import Optional

class MoveRequest(BaseModel):
    description: Optional[str] = Field(
        None,
        description="Optional description of the board state or scenario."
    )
    board: list[list[str]] = Field(
        ...,
        description=(
            "8x8 matrix representing the current board state.\n"
            "Each cell contains a string representing a piece:\n"
            "- 'r': red regular piece\n"
            "- 'R': red king piece\n"
            "- 'b': black regular piece\n"
            "- 'B': black king piece\n"
            "- ' ': empty cell"
        )
    )
    player: str = Field(
        ...,
        description="The current player to make a move ('r' for red or 'b' for black)."
    )
    depth: int = Field(
        3,
        description="The search depth for the minimax algorithm. Default is 3."
    )