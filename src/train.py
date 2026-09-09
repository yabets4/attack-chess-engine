"""Two-phase training loop for ChessNet.

Phase 1 — General pre-training:
  python src/train.py --boards .../boards.npy --moves .../moves.npy \
      --values .../values.npy --out checkpoints/general --epochs 15

Phase 2 — Attack fine-tuning with style shaping:
  python src/train.py --boards .../boards.npy --moves .../moves.npy \
      --values .../values.npy --aggression .../aggression.npy \
      --pretrained checkpoints/general/model_best.pt \
      --style-weight 0.15 --lr 1e-4 --out checkpoints/attack --epochs 20
"""

import os
import sys
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ChessNet
from chess_dataset import ChessDataset


# ───────────────────────── helpers ─────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CosineWarmup:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']


# ───────────────────────── training ─────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    # ── dataset ──
    dataset = ChessDataset(
        args.boards, args.moves,
        values_path=args.values,
        aggression_path=args.aggression,
    )
    print(f'Loaded {len(dataset)} positions', flush=True)
    print(f'  has value targets:    {dataset.has_values}', flush=True)
    print(f'  has aggression scores: {dataset.has_aggression}', flush=True)

    torch.manual_seed(args.seed)

    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=(device.type == 'cuda'))

    # ── model ──
    model = ChessNet(
        num_res=args.num_res,
        filters=args.filters,
        se_ratio=args.se_ratio,
        p_drop=args.dropout,
    ).to(device)
    print(f'Model: {args.num_res} SE-ResBlocks x {args.filters} filters, '
          f'{count_parameters(model):,} parameters', flush=True)

    # ── load pretrained (Phase 2) ──
    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device,
                           weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f'Loaded pretrained weights from {args.pretrained}', flush=True)

    # ── optimiser ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # ── LR schedule ──
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(len(train_loader), total_steps // 10)
    scheduler = CosineWarmup(optimizer, warmup_steps, total_steps)

    # ── loss functions ──
    policy_loss_fn = nn.CrossEntropyLoss(reduction='none')
    value_loss_fn = nn.MSELoss()

    # ── bookkeeping ──
    os.makedirs(args.out, exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        pl_sum = 0.0
        vl_sum = 0.0
        sl_sum = 0.0
        acc_sum = 0.0
        nb = 0

        for bx, by, bv, ba in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            bv = bv.to(device)
            ba = ba.to(device)

            p_logits, v_pred = model(bx)

            # ── policy loss (per-sample, then weighted) ──
            pl_per_sample = policy_loss_fn(p_logits, by)  # (B,)

            if args.style_weight > 0 and dataset.has_aggression:
                # Amplify gradient on aggressive moves
                weight = 1.0 + args.style_weight * ba
                pl = (pl_per_sample * weight).mean()
                style_cost = (pl_per_sample * (args.style_weight * ba)).mean().item()
            else:
                pl = pl_per_sample.mean()
                style_cost = 0.0

            # ── value loss ──
            if dataset.has_values:
                vl = value_loss_fn(v_pred, bv)
            else:
                vl = value_loss_fn(v_pred, torch.zeros_like(v_pred))

            # ── total loss ──
            loss = pl + args.value_weight * vl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                acc = (p_logits.argmax(1) == by).float().mean().item()

            pl_sum += pl.item()
            vl_sum += vl.item()
            sl_sum += style_cost
            acc_sum += acc
            nb += 1
            global_step += 1

            if nb % args.log_interval == 0:
                lr_now = scheduler.lr
                print(f'  [ep{epoch} s{global_step}] '
                      f'pl={pl_sum/nb:.4f}  vl={vl_sum/nb:.4f}  '
                      f'style={sl_sum/nb:.4f}  acc={acc_sum/nb:.4f}  '
                      f'lr={lr_now:.2e}', flush=True)

        # ── validation ──
        val_pl, val_vl, val_acc = evaluate(model, val_loader, device,
                                           dataset.has_values)
        elapsed = time.time() - t0
        print(f'[epoch {epoch}  {elapsed:.0f}s]  '
              f'train_acc={acc_sum/nb:.4f}  val_acc={val_acc:.4f}  '
              f'val_pl={val_pl:.4f}  val_vl={val_vl:.4f}', flush=True)

        # ── save every-epoch checkpoint ──
        ckpt = os.path.join(args.out, f'model_ep{epoch}.pt')
        torch.save(model.state_dict(), ckpt)

        # ── save best model ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_path = os.path.join(args.out, 'model_best.pt')
            torch.save(model.state_dict(), best_path)
            print(f'  * New best val_acc={best_val_acc:.4f} -> {best_path}', flush=True)
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f'  Early stopping after {args.patience} epochs '
                      f'without improvement')
                break

    print(f'\nTraining complete. Best val_acc={best_val_acc:.4f}')


# ───────────────────────── evaluation ─────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, has_values=False):
    model.eval()
    pl_sum, vl_sum, acc_sum, n = 0.0, 0.0, 0.0, 0
    pl_fn = nn.CrossEntropyLoss()
    vl_fn = nn.MSELoss()

    for bx, by, bv, ba in loader:
        bx = bx.to(device)
        by = by.to(device)
        bv = bv.to(device)

        p_logits, v_pred = model(bx)
        pl_sum += pl_fn(p_logits, by).item()
        acc_sum += (p_logits.argmax(1) == by).float().mean().item()

        if has_values:
            vl_sum += vl_fn(v_pred, bv).item()

        n += 1

    return pl_sum / max(n, 1), vl_sum / max(n, 1), acc_sum / max(n, 1)


# ───────────────────────── CLI ─────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ChessNet (Phase 1: general, Phase 2: attack)')

    # data
    parser.add_argument('--boards', required=True,
                        help='Path to boards.npy')
    parser.add_argument('--moves', required=True,
                        help='Path to moves.npy')
    parser.add_argument('--values', default=None,
                        help='Path to values.npy (game results)')
    parser.add_argument('--aggression', default=None,
                        help='Path to aggression.npy (style scores)')

    # model
    parser.add_argument('--num-res', type=int, default=17,
                        help='Number of SE-ResBlocks (default 17)')
    parser.add_argument('--filters', type=int, default=256,
                        help='Channel width (default 256)')
    parser.add_argument('--se-ratio', type=int, default=4,
                        help='SE squeeze ratio (default 4)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout in FC heads (default 0.2)')

    # training
    parser.add_argument('--pretrained', default=None,
                        help='Path to pretrained .pt (for Phase 2)')
    parser.add_argument('--out', default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default 128, lower for CPU)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--value-weight', type=float, default=0.1,
                        help='Weight of value loss (default 0.1)')
    parser.add_argument('--style-weight', type=float, default=0.0,
                        help='Aggression style weight (0 = off, '
                             '0.1-0.2 for Phase 2)')
    parser.add_argument('--clip-norm', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--val-frac', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    train(args)