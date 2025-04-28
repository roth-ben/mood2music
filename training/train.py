#!/usr/bin/env python3
import os
import argparse
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from transformers import get_linear_schedule_with_warmup

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─── Dataset Definition ────────────────────────────────────────────────────────
class MoodDataset(Dataset):
    """Contrastive pairs of (description, description) with 1.0/0.0 labels."""
    def __init__(self, df: pd.DataFrame):
        self.examples = []
        if len(df) < 10:
            raise ValueError("Need at least 10 tracks for contrastive sampling")

        for idx, row in df.iterrows():
            # Positive sampling
            pos_cands = df[
                (df.mood == row.mood) &
                (abs(df.danceability - row.danceability) < 0.1) &
                (abs(df.valence      - row.valence     ) < 0.1) &
                (abs(df.energy       - row.energy      ) < 0.1) &
                (df.index != idx)
            ]
            if pos_cands.empty:
                pos_row = row
            else:
                pos_row = pos_cands.sample(1).iloc[0]

            self.examples.append(
                InputExample(
                    texts=[row.description, pos_row.description],
                    label=1.0
                )
            )

            # Negative sampling (up to two hard negatives per anchor)
            neg_cands = df[
                (df.mood != row.mood) &
                (abs(df.danceability - row.danceability) < 0.25) &
                (abs(df.valence      - row.valence     ) < 0.25) &
                (abs(df.energy       - row.energy      ) < 0.25)
            ]
            if len(neg_cands) < 2:
                neg_cands = df[df.mood != row.mood]
            if len(neg_cands) < 2:
                neg_cands = df.sample(frac=0.1)

            for _, neg_row in neg_cands.sample(n=min(2, len(neg_cands))).iterrows():
                self.examples.append(
                    InputExample(
                        texts=[row.description, neg_row.description],
                        label=0.0
                    )
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ─── Main Training Script ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("Mood‐Aware Contrastive Trainer")
    parser.add_argument('--data',       type=str, required=True,
                        help='CSV file with columns: description, mood, danceability, valence, energy')
    parser.add_argument('--base_model', type=str,
                        default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--lr',         type=float, default=3e-5)
    parser.add_argument('--warmup',     type=int,   default=500)
    parser.add_argument('--output_dir', type=str,   default='models/mood-encoder')
    args = parser.parse_args()

    # ─── GPU Check ───────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ─── Load & Validate Data ────────────────────────────────────────────────────
    df = pd.read_csv(args.data)
    required = {'description', 'mood', 'danceability', 'valence', 'energy'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}")
    logger.info(f"Loaded {len(df)} tracks from {args.data}")

    # ─── Model & DataLoader ─────────────────────────────────────────────────────
    model = SentenceTransformer(args.base_model, device='cuda')
    model.max_seq_length = 128

    train_dataset = MoodDataset(df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ─── Loss & Evaluator ────────────────────────────────────────────────────────
    train_loss = losses.OnlineContrastiveLoss(model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        train_dataset.examples[:1000],
        name='mood-val',
        show_progress_bar=True
    )

    # ─── Fit (no custom callback!) ──────────────────────────────────────────────
    logger.info("Starting training…")
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={
            "lr": args.lr,
            "eps": 1e-8
        },
        scheduler="warmuplinear",
        warmup_steps=args.warmup,
        output_path=args.output_dir,
        use_amp=True,
        checkpoint_path=os.path.join(args.output_dir, 'checkpoints'),
        checkpoint_save_total_limit=1,
        show_progress_bar=True
    )

    # ─── Final Save ─────────────────────────────────────────────────────────────
    model.save(args.output_dir)
    logger.info(f"Training complete. Model saved to {args.output_dir}")


if __name__ == '__main__':
    main()