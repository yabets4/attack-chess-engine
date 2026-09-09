import chess
import numpy as np

# 0-5:  white pawn, knight, bishop, rook, queen, king
# 6-11: black pawn, knight, bishop, rook, queen, king
# 12:   side to move (all 1s if white)
# 13:   white kingside castling
# 14:   white queenside castling
# 15:   black kingside castling
# 16:   black queenside castling
# 17:   en passant square
# 18:   half-move clock (normalised 0-1, capped at 100)
BOARD_CHANNELS = 19
MOVE_CHANNELS = 64 * 64  # from_square * 64 + to_square


def to_matrix(board: chess.Board) -> np.ndarray:
    """Encode a python-chess Board as a (BOARD_CHANNELS, 8, 8) float32 tensor."""
    m = np.zeros((BOARD_CHANNELS, 8, 8), dtype=np.float32)

    # --- piece planes 0-11 ---
    for color in (chess.WHITE, chess.BLACK):
        offset = 0 if color == chess.WHITE else 6
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                           chess.ROOK, chess.QUEEN, chess.KING):
            for sq in board.pieces(piece_type, color):
                r, c = divmod(sq, 8)
                m[offset + piece_type - 1, r, c] = 1.0

    # --- side to move ---
    if board.turn == chess.WHITE:
        m[12] = 1.0

    # --- castling rights ---
    if board.has_kingside_castling_rights(chess.WHITE):
        m[13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        m[14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        m[15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        m[16] = 1.0

    # --- en passant ---
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        m[17, r, c] = 1.0

    # --- half-move clock (normalised) ---
    m[18] = min(board.halfmove_clock / 100.0, 1.0)

    return m


def move_to_index(move: chess.Move) -> int:
    """Map a chess.Move to a flat index in [0, 4096)."""
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int) -> chess.Move:
    """Inverse of move_to_index."""
    from_sq = idx // 64
    to_sq = idx % 64
    return chess.Move(from_sq, to_sq)