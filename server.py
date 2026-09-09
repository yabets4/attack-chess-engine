"""FastAPI backend that serves ChessNet predictions for the React chess app.

Run with:
    python server.py                      # default port 8001
    python server.py --port 8001          # explicit port
    python server.py --model ../model_best.pt  # custom model path
"""

import os
import sys
import argparse
import glob
import time
import uuid
import gc
from pathlib import Path

import torch
import numpy as np
import chess
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

# Add src directory to path for model imports
src_dir = str(Path(__file__).resolve().parent / "src")
sys.path.insert(0, src_dir)

from model import ChessNet
from board_encoding import to_matrix, move_to_index, index_to_move, BOARD_CHANNELS, MOVE_CHANNELS

app = FastAPI(title="ChessNet Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://attack-chess-ui.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
device = None
checkpoints_dir = None

# ── Player tracking ──
sessions = {}          # session_id -> { last_seen, created, games_played, total_moves }
ACTIVE_TIMEOUT = 30    # seconds before a session is considered inactive
total_games_trained = 0
total_moves_played = 0


class MoveRequest(BaseModel):
    fen: str
    temperature: float = 0.7
    checkpoint: str = "latest"


class EvalRequest(BaseModel):
    fen: str
    checkpoint: str = "latest"


class TrainRequest(BaseModel):
    fens: List[str]
    moves: List[str]
    outcome: float


class MinimaxMoveRequest(BaseModel):
    fen: str
    depth: int = 3
    backend: str = "gpu"


class MinimaxEvalRequest(BaseModel):
    fen: str


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Convert a FEN string to a model input tensor."""
    board = chess.Board(fen)
    matrix = to_matrix(board)  # (BOARD_CHANNELS, 8, 8)
    tensor = torch.from_numpy(matrix).unsqueeze(0)  # (1, BOARD_CHANNELS, 8, 8)
    return tensor.to(device)


def get_legal_move_indices(board: chess.Board) -> list:
    """Get indices of all legal moves in the current position."""
    indices = []
    for move in board.legal_moves:
        idx = move_to_index(move)
        indices.append(idx)
    return indices


def load_model(checkpoint_name: str = "latest"):
    """Load a model checkpoint (memory-optimised for small instances)."""
    global model, device

    if device is None:
        device = torch.device("cpu")

    # Find checkpoint path
    if checkpoint_name == "latest":
        best_path = os.path.join(checkpoints_dir, "model_best.pt")
        if os.path.exists(best_path):
            ckpt_path = best_path
        else:
            ep_files = sorted(glob.glob(os.path.join(checkpoints_dir, "model_ep*.pt")))
            if ep_files:
                ckpt_path = ep_files[-1]
            else:
                raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    else:
        ckpt_path = os.path.join(checkpoints_dir, checkpoint_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Create model with float32 (not float64) and no dropout for inference
    net = ChessNet(num_res=17, filters=256, se_ratio=4, p_drop=0.0)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)

    # Free memory: delete optimizer state if present, strip gradients
    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    gc.collect()
    return net


def policy_to_move(policy_logits: torch.Tensor, board: chess.Board, temperature: float = 0.7) -> chess.Move:
    """Convert policy logits to a legal move using temperature sampling."""
    legal_indices = get_legal_move_indices(board)

    if not legal_indices:
        raise ValueError("No legal moves available")

    # Extract logits for legal moves
    legal_logits = policy_logits[0, legal_indices]  # (num_legal,)

    # Apply temperature
    if temperature > 0:
        probs = torch.softmax(legal_logits / temperature, dim=0)
        # Sample from distribution
        idx = torch.multinomial(probs, 1).item()
    else:
        # Greedy selection
        idx = torch.argmax(legal_logits).item()

    return index_to_move(legal_indices[idx])


def evaluate_position(board: chess.Board) -> float:
    """Evaluate a position using the model's value head."""
    fen = board.fen()
    tensor = fen_to_tensor(fen)

    with torch.no_grad():
        _, value = model(tensor)

    return value.item()


def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
    """Minimax with alpha-beta pruning."""
    if depth == 0 or board.is_game_over():
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0
            elif result == "0-1":
                return -1.0
            return 0.0
        return evaluate_position(board)

    legal_moves = list(board.legal_moves)

    if maximizing:
        max_eval = float("-inf")
        for move in legal_moves:
            board.push(move)
            # After push, it's the opponent's turn, so next call should be minimizing
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in legal_moves:
            board.push(move)
            # After push, it's the opponent's turn, so next call should be maximizing
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval


@app.middleware("http")
async def track_activity(request: Request, call_next):
    """Track per-session activity on every API call."""
    global total_moves_played
    session_id = request.headers.get("X-Session-Id") or request.cookies.get("session_id")
    if session_id and session_id in sessions:
        sessions[session_id]["last_seen"] = time.time()
    response = await call_next(request)
    # Track move completions
    if request.url.path == "/api/move" and request.method == "POST":
        total_moves_played += 1
    return response


@app.post("/api/admin/heartbeat")
async def heartbeat(request: Request):
    """Register or keep alive a player session."""
    body = await request.json()
    session_id = body.get("session_id") or str(uuid.uuid4())
    now = time.time()
    if session_id not in sessions:
        sessions[session_id] = {
            "created": now,
            "last_seen": now,
            "games_played": 0,
            "total_moves": 0,
        }
    else:
        sessions[session_id]["last_seen"] = now
    return {"session_id": session_id}


@app.post("/api/admin/game-over")
async def game_over(request: Request):
    """Record that a session finished a game."""
    body = await request.json()
    session_id = body.get("session_id")
    if session_id and session_id in sessions:
        sessions[session_id]["games_played"] += 1
    return {"ok": True}


@app.get("/api/admin/stats")
async def admin_stats():
    """Return current player statistics."""
    now = time.time()
    # Prune stale sessions
    stale = [sid for sid, s in sessions.items() if now - s["last_seen"] > ACTIVE_TIMEOUT]
    for sid in stale:
        del sessions[sid]

    active = len(sessions)
    total_unique = len(sessions)  # currently active unique players
    total_games = sum(s["games_played"] for s in sessions.values()) + total_games_trained

    return {
        "active_players": active,
        "total_games_played": total_games,
        "total_moves_played": total_moves_played,
        "sessions": [
            {
                "session_id": sid[:8] + "...",
                "games_played": s["games_played"],
                "active_seconds": int(now - s["created"]),
            }
            for sid, s in sorted(sessions.items(), key=lambda x: x[1]["last_seen"], reverse=True)
        ],
    }


@app.on_event("startup")
async def startup_event():
    """Print startup info — model loads lazily on first request."""
    global checkpoints_dir
    checkpoints_dir = str(Path(__file__).resolve().parent)
    print(f"Checkpoints directory: {checkpoints_dir}")
    print("Model will load lazily on first request to save memory.")


@app.get("/api/checkpoints")
async def get_checkpoints():
    """List available checkpoints."""
    global checkpoints_dir

    if checkpoints_dir is None:
        checkpoints_dir = str(Path(__file__).resolve().parent)

    checkpoints = ["latest"]
    for f in sorted(glob.glob(os.path.join(checkpoints_dir, "model_*.pt"))):
        name = os.path.basename(f)
        if name not in checkpoints:
            checkpoints.append(name)

    return {"checkpoints": checkpoints}


@app.post("/api/move")
async def get_move(req: MoveRequest):
    """Get a move from the model for the given position."""
    global model, checkpoints_dir

    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN")

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    # Load checkpoint if needed
    if req.checkpoint != "latest" or model is None:
        try:
            model = load_model(req.checkpoint)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # Get model prediction
    tensor = fen_to_tensor(req.fen)
    with torch.no_grad():
        policy_logits, value = model(tensor)

    # Convert to move
    move = policy_to_move(policy_logits, board, req.temperature)

    return {"move": move.uci(), "value": value.item()}


@app.post("/api/eval")
async def evaluate(req: EvalRequest):
    """Evaluate a position."""
    global model, checkpoints_dir

    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN")

    # Load checkpoint if needed
    if req.checkpoint != "latest" or model is None:
        try:
            model = load_model(req.checkpoint)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    value = evaluate_position(board)
    return {"value": value}


@app.post("/api/minimax/move")
async def minimax_move(req: MinimaxMoveRequest):
    """Get a move using minimax with alpha-beta pruning."""
    global model, checkpoints_dir

    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN")

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    # Load model if needed
    if model is None:
        try:
            model = load_model("latest")
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # Get all legal moves and evaluate each
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = float("-inf") if board.turn == chess.WHITE else float("inf")

    for move in legal_moves:
        board.push(move)
        score = minimax(board, req.depth - 1, float("-inf"), float("inf"), board.turn == chess.WHITE)
        board.pop()

        if board.turn == chess.WHITE:
            if score > best_score:
                best_score = score
                best_move = move
        else:
            if score < best_score:
                best_score = score
                best_move = move

    if best_move is None:
        raise HTTPException(status_code=500, detail="No move found")

    return {"move": best_move.uci(), "value": best_score}


@app.post("/api/minimax/eval")
async def minimax_eval(req: MinimaxEvalRequest):
    """Evaluate a position using minimax."""
    global model, checkpoints_dir

    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN")

    # Load model if needed
    if model is None:
        try:
            model = load_model("latest")
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    value = evaluate_position(board)
    return {"value": value}


@app.post("/api/train")
async def train(req: TrainRequest):
    """Train the model on a completed game (online learning)."""
    global model, checkpoints_dir

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert FENs and moves to training data
    boards = []
    move_indices = []
    values = []

    for fen, move_uci in zip(req.fens, req.moves):
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)

            if move not in board.legal_moves:
                continue

            # Encode board
            tensor = fen_to_tensor(fen)
            boards.append(tensor.squeeze(0))

            # Encode move
            move_idx = move_to_index(move)
            move_indices.append(move_idx)

            # Value target (from the perspective of the player who moved)
            value = req.outcome
            if board.turn == chess.BLACK:
                value = -value
            values.append(value)

        except Exception as e:
            print(f"Error processing position: {e}")
            continue

    if not boards:
        raise HTTPException(status_code=400, detail="No valid positions to train on")

    # Prepare tensors
    board_tensor = torch.stack(boards).to(device)
    move_tensor = torch.tensor(move_indices, dtype=torch.long).to(device)
    value_tensor = torch.tensor(values, dtype=torch.float32).to(device)

    # Train for a few steps
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    # Mini-batch training
    batch_size = min(32, len(boards))
    for i in range(0, len(boards), batch_size):
        batch_boards = board_tensor[i:i+batch_size]
        batch_moves = move_tensor[i:i+batch_size]
        batch_values = value_tensor[i:i+batch_size]

        policy_logits, value_pred = model(batch_boards)

        policy_loss = policy_loss_fn(policy_logits, batch_moves)
        value_loss = value_loss_fn(value_pred.squeeze(), batch_values)

        loss = policy_loss + 0.1 * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    model.eval()

    # Track global training count
    global total_games_trained
    total_games_trained += 1

    # Save checkpoint
    checkpoint_name = f"model_online_{len(glob.glob(os.path.join(checkpoints_dir, 'model_online_*.pt')))}.pt"
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)

    return {
        "moves_trained": len(boards),
        "avg_policy_loss": total_policy_loss / max(num_batches, 1),
        "avg_value_loss": total_value_loss / max(num_batches, 1),
        "total_games_trained": 1,
        "checkpoint_saved": True,
        "checkpoint_name": checkpoint_name,
    }


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="ChessNet Backend Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--checkpoints-dir", type=str, default=None, help="Path to checkpoints directory")

    args = parser.parse_args()

    if args.checkpoints_dir:
        checkpoints_dir = args.checkpoints_dir
    elif args.model:
        checkpoints_dir = str(Path(args.model).parent)

    if args.model:
        try:
            model = load_model(args.model)
            print(f"Loaded model from {args.model}")
        except Exception as e:
            print(f"Error loading model: {e}")

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
