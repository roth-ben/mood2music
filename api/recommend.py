import logging
import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

router = APIRouter()

class RecommendationRequest(BaseModel):
    text: str
    top_k: int = 5

# Model and data initialization
try:
    # 1. Load model with CUDA
    model = SentenceTransformer("models/mood-encoder", device="cuda")
    
    # 2. Optimize for RTX 3080 Ti
    model.to(torch.float16)
    model.max_seq_length = 128
    
    logger.info(f"Model loaded on {model.device} with dtype {model[0].auto_model.dtype}")

    # 2. Load and validate track data
    tracks = pd.read_csv("data/raw/fake_tracks.csv")
    required_cols = {'id', 'title', 'description', 'mood', 'danceability', 'valence', 'energy'}
    assert required_cols.issubset(tracks.columns), "Missing required columns"
    logger.info(f"Loaded {len(tracks)} tracks")

    # 3. GPU-accelerated embedding precomputation
    with torch.inference_mode(), torch.amp.autocast(device_type='cuda'):
        song_embeddings = model.encode(
            tracks['description'].tolist(),
            batch_size=128,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True
        ).cpu().numpy()
        
    logger.info(f"Precomputed embeddings shape: {song_embeddings.shape}")

except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise RuntimeError(f"System initialization failed: {str(e)}")

@router.post("/recommend", response_model=dict)
async def match_playlist(request: RecommendationRequest):
    """GPU-accelerated mood-to-playlist matching"""
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(400, "Empty mood description")
        
        if request.top_k < 1 or request.top_k > 100:
            raise HTTPException(400, "top_k must be between 1-100")

        # GPU encoding with mixed precision
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            query_embed = model.encode(
                request.text,
                convert_to_tensor=True,
                device="cuda",
                normalize_embeddings=True
            ).cpu().numpy()

        # Similarity search
        scores = cosine_similarity([query_embed], song_embeddings)[0]
        top_indices = np.argsort(scores)[-request.top_k:][::-1]

        # Format response
        results = []
        for idx in top_indices:
            track = tracks.iloc[idx]
            results.append({
                "id": track['id'],
                "title": track['title'],
                "score": float(scores[idx]),
                "mood": track['mood'],
                #"danceability": track['danceability'],
                #"spotify_url": f"https://open.spotify.com/track/{track['id'].split(':')[-1]}"
            })

        return {"tracks": results}

    except torch.cuda.OutOfMemoryError:
        logger.error("GPU OOM during inference")
        raise HTTPException(500, "GPU memory exhausted")
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        raise HTTPException(500, f"Recommendation error: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "tracks_loaded": len(tracks) > 0
    }