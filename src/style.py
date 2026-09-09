"""Aggression scoring for style-shaping loss.

Computes a 0-1 score measuring how aggressive a move is, used during
Phase-2 fine-tuning to amplify the policy gradient on attacking moves.
"""

import chess

# Piece values for sacrifice detection
_PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

# Squares in / around the center
_CENTER = {chess.D4, chess.D5, chess.E4, chess.E5}
_WIDE_CENTER = _CENTER | {chess.C3, chess.C4, chess.C5, chess.C6,
                           chess.D3, chess.D6,
                           chess.E3, chess.E6,
                           chess.F3, chess.F4, chess.F5, chess.F6}


def _king_zone(king_sq: int) -> set:
    """Return the set of squares within king-attack distance (king ring + 1)."""
    kr, kc = divmod(king_sq, 8)
    zone = set()
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = kr + dr, kc + dc
            if 0 <= r < 8 and 0 <= c < 8:
                zone.add(r * 8 + c)
    return zone


def compute_aggression(board: chess.Board, move: chess.Move) -> float:
    """Score a move's aggression on a 0-1 scale.

    Components (max raw ≈ 1.6, clamped to 1.0):
      - capture: +0.20, sacrifice: up to +0.40
      - gives check: +0.25
      - centre control: +0.10
      - piece advancement toward opponent: +0.10
      - king-zone pressure: +0.20
      - retreat penalty: -0.10
    """
    score = 0.0

    # --- capture ---
    if board.is_capture(move):
        score += 0.20
        # Sacrifice bonus: attacker worth more than victim
        attacker_type = board.piece_type_at(move.from_square)
        victim_type = board.piece_type_at(move.to_square)
        if victim_type is None:
            # en passant
            victim_type = chess.PAWN
        att_val = _PIECE_VALUE.get(attacker_type, 0)
        vic_val = _PIECE_VALUE.get(victim_type, 0)
        if att_val > vic_val:
            # sacrificial capture — the bigger the sacrifice, the bigger the bonus
            score += min(0.40, 0.10 * (att_val - vic_val))

    # --- gives check ---
    board.push(move)
    if board.is_check():
        score += 0.25
    board.pop()

    # --- centre control ---
    if move.to_square in _CENTER:
        score += 0.10
    elif move.to_square in _WIDE_CENTER:
        score += 0.05

    # --- piece advancement ---
    piece_type = board.piece_type_at(move.from_square)
    if piece_type and piece_type != chess.KING:
        from_rank = move.from_square // 8
        to_rank = move.to_square // 8
        if board.turn == chess.WHITE:
            advance = to_rank - from_rank
        else:
            advance = from_rank - to_rank
        if advance > 0:
            score += min(0.10, 0.025 * advance)
        elif advance < -1:
            score -= 0.10  # retreat penalty

    # --- king-zone pressure ---
    opp_king_sq = board.king(not board.turn)
    if opp_king_sq is not None:
        zone = _king_zone(opp_king_sq)
        if move.to_square in zone:
            score += 0.20

    return max(0.0, min(1.0, score))
