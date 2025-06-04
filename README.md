# PyCheckersBot

PyCheckersBot is a simple API-based checkers bot implemented in Python using FastAPI. It allows clients to request the best move for a given board state using the Minimax algorithm with Alpha-Beta pruning.

## Overview

This project provides a backend for a checkers-playing bot. Given a valid board state and the current player, the bot returns the best move sequence based on its internal evaluation function.

The AI supports:
- **Standard movement rules** for regular and king pieces
- **Turn management** between red (`r`) and black (`b`) players
- **Forced captures** when available, as required by checkers rules
- **Automatic promotion** to king when reaching the last row
- **Chained captures** in a single turn, including after promotion
- **Board validation** to ensure consistent and playable game states
- Configurable **search depth** using the `depth` parameter, affecting AI lookahead

## Tech Stack

- **Python 3.10+**
- **FastAPI** – RESTful API framework
- **Pydantic** – Data validation
- **Uvicorn** – ASGI server for running the app

## Algorithm (Brief Explanation)

The bot uses the **[Minimax algorithm](https://en.wikipedia.org/wiki/Minimax)**, a classic decision rule used in two-player, turn-based games to minimize the possible loss for a worst-case scenario. To improve performance, it applies **[Alpha-Beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)**, which reduces the number of nodes evaluated in the game tree by skipping branches that cannot influence the final decision.

Each board state is evaluated recursively up to a configurable depth, with the bot trying to maximize its own advantage while minimizing the opponent’s potential gain.

## Running the App

```bash
# Clone the repo
git clone https://github.com/Eduard0Rocha/PyCheckersBot.git
cd PyCheckersBot

# Install dependencies
pip install -r requirements.txt

# Change to the app directory
cd app

# Run the FastAPI server
uvicorn main:app
```

Once running, you can access the interactive documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Example Request

**POST** /play

```json
{
  "description": "Example 0 - Opening position",
  "board": [
    [" ", "b", " ", "b", " ", "b", " ", "b"],
    ["b", " ", "b", " ", "b", " ", "b", " "],
    [" ", "b", " ", "b", " ", "b", " ", "b"],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    ["r", " ", "r", " ", "r", " ", "r", " "],
    [" ", "r", " ", "r", " ", "r", " ", "r"],
    ["r", " ", "r", " ", "r", " ", "r", " "]
  ],
  "player": "r",
  "depth": 3
}
```

**Response:**

```json
{
  "move": [
    [[5, 0], [4, 1]]
  ]
}
```

## Rules and Assumptions

- The bottom of the board (rows **5 to 7**) is the **red side** (`r`).
- The top of the board (rows **0 to 2**) is the **black side** (`b`).
- **Captures are mandatory** if available.
- If a piece is promoted during a capture sequence, it **can continue capturing as a king in the same turn**.

## Project Structure

```bash
PyCheckersBot/
├── app/
│ ├── api/
│ │ ├── models.py    # Pydantic schemas for request payload validation and API documentation
│ │ └── routes.py    # API route handlers (FastAPI)
│ ├── core/
│ │ ├── bot.py       # Core bot logic: minimax, move generation, etc.
│ │ └── utils.py     # Input validation utilities and debugging tools for board state representation
│ └─── main.py       # FastAPI app instance and router registration
├── input_examples/  # JSON files with request examples for testing
├── LICENSE          # MIT License
├── README.md        # Project documentation
└── requirements.txt # Python dependencies
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
