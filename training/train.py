#!/usr/bin/env python3
import os
import argparse
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.model_selection import train_test_split


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

# ─── Text Augmentation ─────────────────────────────────────────────────────────
""" from nlpaug.augmenter.word import BackTranslationAug

aug = BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',  # English → German → English
    to_model_name='facebook/wmt19-de-en'
)

def augment_text(text):
    return aug.augment(text) """

# ─── Dataset Definition ────────────────────────────────────────────────────────
class MoodDataset(Dataset):
    """Generates contrastive pairs with hard negative mining"""
    def __init__(self, df: pd.DataFrame):
        self.examples = []
        if len(df) < 10:
            raise ValueError("Need at least 10 tracks for contrastive learning")
            
        for idx, row in df.iterrows():
            # Positive sampling
            pos_candidates = df[
                (df.mood == row.mood) & 
                (df.index != idx) &
                (abs(df.danceability - row.danceability) < 0.1) &
                (abs(df.valence - row.valence) < 0.1)
            ]
            pos_sample = pos_candidates.sample(1).iloc[0] if not pos_candidates.empty else row
            
            # Hard negative sampling
            neg_candidates = df[
                (df.mood != row.mood) &
                (abs(df.danceability - row.danceability) < 0.25) &
                (abs(df.valence - row.valence) < 0.25)
            ]
            if len(neg_candidates) < 2:
                neg_candidates = df[df.mood != row.mood]
            if len(neg_candidates) < 2:
                neg_candidates = df.sample(frac=0.1)
                
            for _, neg_row in neg_candidates.sample(min(2, len(neg_candidates))).iterrows():
                self.examples.append(InputExample(
                    texts=[row['description'], neg_row['description']],
                    label=0.0  # Negative pair
                ))
            self.examples.append(InputExample(
                texts=[row['description'], pos_sample['description']],
                label=1.0  # Positive pair
            ))

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

    train_df, val_df = train_test_split(
        df, 
        test_size=0.1, 
        random_state=42, 
        stratify=df['mood']
    )
    logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = MoodDataset(train_df)
    val_dataset = MoodDataset(val_df)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ─── Loss & Evaluator ────────────────────────────────────────────────────────
    train_loss = losses.ContrastiveLoss(
        model=model,
        margin=0.3,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE
    )

    val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_dataset.examples[:1000],  # Cap validation size
        name='mood-val',
        show_progress_bar=True
    )

    # ─── Fit (no custom callback!) ──────────────────────────────────────────────
    logger.info("Starting training…")
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=val_evaluator,
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