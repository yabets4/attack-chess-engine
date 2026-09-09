"""Build numpy dataset from PGN files — compact uint8 storage.

Outputs:
  boards.npy     — uint8    (N, BOARD_CHANNELS, 8, 8)  ← 4× smaller than float32
  moves.npy      — int64    (N,)
  values.npy     — float32  (N,)   game result from side-to-move perspective
  aggression.npy — float32  (N,)   move aggression score 0-1

Board channels 0-17 are binary (0 or 1).
Channel 18 (half-move clock) is stored as uint8 (0-255), normalised to 0-1 at load time.
"""

import os
import sys
import argparse
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgn_parser import parse_pgn_games, game_to_samples

CHUNK_SIZE = 200_000  # ~230 MB per chunk in uint8 (vs 930 MB in float32)


def _board_to_uint8(board_f32: np.ndarray) -> np.ndarray:
    """Convert a float32 (BOARD_CHANNELS, 8, 8) board to uint8.

    Channels 0-17: binary, stored as 0/1 uint8.
    Channel 18: half-move clock fraction, scaled to 0-255.
    """
    out = np.zeros_like(board_f32, dtype=np.uint8)
    out[:18] = (board_f32[:18] > 0.5).astype(np.uint8)
    out[18] = np.clip(board_f32[18] * 255, 0, 255).astype(np.uint8)
    return out


def _save_chunk(out_dir, idx, boards, moves, values, aggression):
    """Save one chunk of data as temporary .npy files."""
    prefix = os.path.join(out_dir, f'_chunk{idx}')
    # Convert boards to uint8 before stacking to save memory + disk
    boards_u8 = np.stack([_board_to_uint8(b) for b in boards])
    np.save(f'{prefix}_boards.npy', boards_u8)
    np.save(f'{prefix}_moves.npy', np.array(moves, dtype=np.int64))
    np.save(f'{prefix}_values.npy', np.array(values, dtype=np.float32))
    np.save(f'{prefix}_aggr.npy', np.array(aggression, dtype=np.float32))
    del boards_u8
    return len(moves)


def _merge_chunks(out_dir, num_chunks, total_positions):
    """Merge chunk files into final memory-mapped arrays, then clean up."""
    from board_encoding import BOARD_CHANNELS

    board_shape = (total_positions, BOARD_CHANNELS, 8, 8)
    boards_out = np.lib.format.open_memmap(
        os.path.join(out_dir, 'boards.npy'), mode='w+',
        dtype=np.uint8, shape=board_shape)
    moves_out = np.lib.format.open_memmap(
        os.path.join(out_dir, 'moves.npy'), mode='w+',
        dtype=np.int64, shape=(total_positions,))
    values_out = np.lib.format.open_memmap(
        os.path.join(out_dir, 'values.npy'), mode='w+',
        dtype=np.float32, shape=(total_positions,))
    aggr_out = np.lib.format.open_memmap(
        os.path.join(out_dir, 'aggression.npy'), mode='w+',
        dtype=np.float32, shape=(total_positions,))

    offset = 0
    for i in range(num_chunks):
        prefix = os.path.join(out_dir, f'_chunk{i}')
        cb = np.load(f'{prefix}_boards.npy', mmap_mode='r')
        cm = np.load(f'{prefix}_moves.npy', mmap_mode='r')
        cv = np.load(f'{prefix}_values.npy', mmap_mode='r')
        ca = np.load(f'{prefix}_aggr.npy', mmap_mode='r')

        n = len(cm)
        boards_out[offset:offset + n] = cb
        moves_out[offset:offset + n] = cm
        values_out[offset:offset + n] = cv
        aggr_out[offset:offset + n] = ca
        offset += n
        print(f'  merged chunk {i} ({n} positions, {offset}/{total_positions})')

        del cb, cm, cv, ca
        for suffix in ('_boards.npy', '_moves.npy', '_values.npy', '_aggr.npy'):
            os.remove(f'{prefix}{suffix}')

    del boards_out, moves_out, values_out, aggr_out
    gc.collect()


def build_dataset(pgn_paths, out_dir, max_games=None):
    os.makedirs(out_dir, exist_ok=True)

    # Remove old data to free disk space
    for old in ('boards.npy', 'moves.npy', 'values.npy', 'aggression.npy'):
        old_path = os.path.join(out_dir, old)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f'  removed old {old}')

    buf_boards, buf_moves, buf_values, buf_aggr = [], [], [], []
    chunk_idx = 0
    total_positions = 0
    total_games = 0

    for path in pgn_paths:
        print(f'Processing {path} ...')
        game_count = 0

        for game in parse_pgn_games(path, max_games):
            for x, y, v, a in game_to_samples(game):
                buf_boards.append(x)
                buf_moves.append(y)
                buf_values.append(v)
                buf_aggr.append(a)

            game_count += 1
            if game_count % 1000 == 0:
                print(f'  ... {game_count} games')

            if len(buf_boards) >= CHUNK_SIZE:
                n = _save_chunk(out_dir, chunk_idx,
                                buf_boards, buf_moves, buf_values, buf_aggr)
                total_positions += n
                chunk_idx += 1
                print(f'  [chunk {chunk_idx - 1}: {n} pos, {total_positions} total]')
                buf_boards, buf_moves, buf_values, buf_aggr = [], [], [], []
                gc.collect()

            if max_games is not None and game_count >= max_games:
                break

        print(f'  {game_count} games from {os.path.basename(path)}')
        total_games += game_count

    if buf_boards:
        n = _save_chunk(out_dir, chunk_idx,
                        buf_boards, buf_moves, buf_values, buf_aggr)
        total_positions += n
        chunk_idx += 1
        print(f'  [chunk {chunk_idx - 1}: {n} pos, {total_positions} total]')
        del buf_boards, buf_moves, buf_values, buf_aggr
        gc.collect()

    print(f'\nMerging {chunk_idx} chunks ({total_positions} positions)...')
    _merge_chunks(out_dir, chunk_idx, total_positions)

    # Verify
    boards = np.load(os.path.join(out_dir, 'boards.npy'), mmap_mode='r')
    moves = np.load(os.path.join(out_dir, 'moves.npy'), mmap_mode='r')
    values = np.load(os.path.join(out_dir, 'values.npy'), mmap_mode='r')
    aggr = np.load(os.path.join(out_dir, 'aggression.npy'), mmap_mode='r')

    boards_mb = os.path.getsize(os.path.join(out_dir, 'boards.npy')) / 1e6
    print(f'\nDone - {total_games} games, {total_positions} positions -> {out_dir}')
    print(f'  boards:     {boards.shape}  {boards.dtype}  ({boards_mb:.0f} MB)')
    print(f'  moves:      {moves.shape}   {moves.dtype}')
    print(f'  values:     {values.shape}   mean={np.mean(values):.3f}')
    print(f'  aggression: {aggr.shape}   mean={np.mean(aggr):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn', nargs='+', required=True,
                        help='One or more PGN file paths')
    parser.add_argument('--out', required=True,
                        help='Output directory for .npy files')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Max games per PGN file (default: all)')
    args = parser.parse_args()
    build_dataset(args.pgn, args.out, max_games=args.max_games)