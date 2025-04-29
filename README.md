# Music Recommendation System 🎵🤖

A lightweight, Dockerized music recommendation engine designed for experimentation and small-scale deployment. The system generates synthetic mood-based training data, fine-tunes a sentence-transformers model for semantic song-mood matching, and serves personalized recommendations via a REST API. Optimized for efficiency, it supports GPU acceleration (NVIDIA RTX 30-series or newer with ≥2GB VRAM) to enable fast training and inference on local hardware, ideal for prototyping, research, or hobbyist use.

![Docker](https://img.shields.io/badge/Docker-✔️-success) ![GPU Support](https://img.shields.io/badge/NVIDIA_GPU-✔️-76B900) ![REST API](https://img.shields.io/badge/REST_API-✔️-blue)

## Features ✨

- **Automated Pipeline**: Single-command data generation and model training
- **GPU Accelerated**: Optimized for NVIDIA GPUs with CUDA support
- **REST API**: Simple HTTP endpoint for recommendations
- **Customizable**: Easily modify recommendation count per request
- **Synthetic Data Generation**: Built-in fake data creation for testing

## Prerequisites 📋

- Docker Engine ≥ 20.10
- NVIDIA CUDA-Supported GPU (RTX 30's+ series)
- NVIDIA Docker Toolkit (for GPU acceleration)
- Python 3.8+ (for local development)
- curl (for API testing)

## Getting Started 🚀

### 1. Build Docker Image
```bash
docker build -t music-recommender .
```

### 2. Run Container (with GPU Support)
```bash
docker run -d --gpus all -p 8080:8080 --name recommender music-recommender
```

### 3. Generate Fake Music Data
```bash
docker exec -it --workdir /app recommender python3 ./scripts/generate_fake_spotify.py
```

### 4. Train Model on Mood Awareness
```bash
docker exec -it --workdir /app recommender python3 ./training/train.py \
    --data data/raw/fake_tracks.csv \
    --batch_size 64 \
    --epochs 15 \
    --lr 3e-5 \
    --warmup 500
```

**⏳ First Run Notes:**  
Initial execution may take longer due to:  
- Synthetic data generation (~2 minutes)  
- Model training (~5 minutes)  
- API server initialization (~30 seconds)

## API Documentation 📖

### Recommendation Endpoint
`POST /recommend`

**Request Format:**
```json
{
  "text": "Description of musical preferences",
  "top_k": 5
}
```

**Parameters:**
- `text`: (Required) Text description of desired music characteristics
- `top_k`: (Optional) Number of songs to return [1-10], default=5

**Response Format:**
```json
{
  "tracks": [
    {
        "id": "Song ID",
        "artist": "Artist Name",
        "title": "Song Title",
        "score": 0.95,
        "mood": "calm"
    },
    ...
  ]
}
```

## Usage Examples 💡

### Basic Recommendation
```bash
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "Uplifting EDM track with hype vocals and a build-up structure"}'
```

### Custom Recommendation Count
```bash
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "Jazzy lo-fi beats for studying", "top_k": 3}'
```

### Benchmark
This visualization compares the scoring distributions of the fine-tuned recommendation model against the base model. The refined model demonstrates a stronger alignment between songs and mood descriptors, with higher matching scores and tighter clustering, indicating more precise and emotionally relevant recommendations.

![Model Benchmark](https://github.com/roth-ben/mood2music/blob/main/similarity_distribution.png)

## Project Structure 📂
```
├── Dockerfile             # Container configuration
├── api/                   # Application code
├── data/                  # Synthetic data creation
├── models/                # Recommendation model
├── test/                  # Debug and analysis
└── training/              # Training scripts
```

## Troubleshooting 🔧

### Common Issues
- **Port Conflict**: Ensure port 8080 is free (`lsof -i :8080`)
- **GPU Availability**: Verify NVIDIA drivers with `nvidia-smi`
- **Container Logs**: Check logs with `docker logs recommender`

**Note**: This system uses synthetic data for demonstration purposes. For production use, replace with real music metadata and listening history data.

# Data Quality Tester
**Purpose**: Validates core integrity of music track metadata.

## Quick Start
```bash
python3 ./test/debug_data.py
```

## What It Checks
✅ **Basic Structure**: File loading and required columns  
✅ **Value Ranges**: Validates audio features (0-1 scale)  
✅ **Moods**: Checks expected categories exist  
✅ **Completeness**: Identifies missing values  

# Embedding Debugger  

**Purpose**: Validates embedding model quality, data integrity, and similarity distributions before deployment.  

## Quick Start  
```bash  
python3 ./test/debug_embeddings.py  
```  

## What It Checks  
✅ **Data**: Loads CSV, checks for duplicates/sample size  
✅ **Model**: Validates loading, GPU compatibility, output dimensions  
✅ **Embeddings**: Tests generation, normalization, and numerical stability  
✅ **Similarity**: Computes all-vs-all scores, analyzes distribution 

# Model Performance Tester

**Purpose**: Benchmarks embedding model quality by comparing tuned vs. base model performance on music recommendations.

## Quick Start
```bash
python3 ./test/debug_model.py
```

## Key Features
- 🏷️ Tests 100 `energetic` tracks against `Uplifting EDM` track query 
- ⚡ Measures GPU-accelerated processing speed
- 📊 Generates similarity distribution histogram
- 🔍 Compares trained vs. base model (all-mpnet-base-v2)

## Interpretation Guide
| Score Range  | Match Quality          | Expected Action        |
|--------------|------------------------|------------------------|
| >0.7         | Strong recommendation  | Prioritize in results  |
| 0.5-0.7      | Moderate match         | Secondary candidates   |
| <0.4         | Weak match             | Filter out             |

## Key Metrics
✅ **Performance Delta**: Tuned vs. base model improvement  
✅ **Strong Matches**: Count of scores >0.7  
✅ **Processing Speed**: Embedding generation time