"""Parse PGN files into (board, move, value, aggression) training samples."""

import chess.pgn
import numpy as np

from board_encoding import to_matrix, move_to_index
from style import compute_aggression


def parse_pgn_games(pgn_path, max_games=None):
    """Yield chess.pgn.Game objects from a PGN file."""
    with open(pgn_path, encoding='utf-8', errors='replace') as f:
        count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if max_games is not None and count >= max_games:
                break
            yield game
            count += 1


def _parse_result(game) -> float:
    """Extract game result as a value target from White's perspective.

    Returns +1.0 for 1-0, -1.0 for 0-1, 0.0 for draw or unknown.
    """
    result = game.headers.get('Result', '*')
    if result == '1-0':
        return 1.0
    elif result == '0-1':
        return -1.0
    else:
        return 0.0


def game_to_samples(game):
    """Convert a game into training samples.

    Returns list of (board_flat, move_index, value_for_side_to_move, aggression).
    """
    result_white = _parse_result(game)
    board = game.board()
    samples = []
    for move in game.mainline_moves():
        x = to_matrix(board)  # (19, 8, 8)

        # Value from the perspective of the side to move
        if board.turn == chess.WHITE:
            value = result_white
        else:
            value = -result_white

        aggression = compute_aggression(board, move)
        y = move_to_index(move)

        samples.append((x, y, np.float32(value), np.float32(aggression)))
        board.push(move)
    return samples