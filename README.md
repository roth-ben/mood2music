# Music Recommendation System 🎵🤖

A Dockerized music recommendation system that generates synthetic data, trains a model, and serves personalized song recommendations via a REST API. Supports GPU acceleration for efficient training and inference.

![Docker](https://img.shields.io/badge/Docker-✔️-success) ![GPU Support](https://img.shields.io/badge/NVIDIA_GPU-✔️-76B900) ![REST API](https://img.shields.io/badge/REST_API-✔️-blue)

## Table of Contents

- [Music Recommendation System 🎵🤖](#music-recommendation-system-)
  - [Table of Contents](#table-of-contents)
  - [Features ✨](#features-)
  - [Prerequisites 📋](#prerequisites-)
  - [Getting Started 🚀](#getting-started-)
    - [1. Build Docker Image](#1-build-docker-image)
    - [2. Run Container (with GPU Support)](#2-run-container-with-gpu-support)
    - [3. Generate Fake Music Data](#3-generate-fake-music-data)
    - [4. Train Model on Mood Awareness](#4-train-model-on-mood-awareness)
  - [API Documentation 📖](#api-documentation-)
    - [Recommendation Endpoint](#recommendation-endpoint)
  - [Usage Examples 💡](#usage-examples-)
    - [Basic Recommendation](#basic-recommendation)
    - [Custom Recommendation Count](#custom-recommendation-count)
  - [Project Structure 📂](#project-structure-)
  - [Troubleshooting 🔧](#troubleshooting-)
    - [Common Issues](#common-issues)
    - [Without GPU Support](#without-gpu-support)
- [Data Quality Tester](#data-quality-tester)
  - [Quick Start](#quick-start)
  - [What It Checks](#what-it-checks)
- [Embedding Debugger](#embedding-debugger)
  - [Quick Start](#quick-start-1)
  - [What It Checks](#what-it-checks-1)
- [Model Performance Tester](#model-performance-tester)
  - [Quick Start](#quick-start-2)
  - [Key Features](#key-features)
  - [Interpretation Guide](#interpretation-guide)
  - [Key Metrics](#key-metrics)
- [Product-Scale Deployment](#product-scale-deployment)
    - [**1. System Goals \& Non-Functional Requirements**](#1-system-goals--non-functional-requirements)
    - [**2. Technical Architecture Components**](#2-technical-architecture-components)

## Features ✨

- **Automated Pipeline**: Single-command data generation and model training
- **GPU Accelerated**: Optimized for NVIDIA GPUs with CUDA support
- **REST API**: Simple HTTP endpoint for recommendations
- **Customizable**: Easily modify recommendation count per request
- **Synthetic Data Generation**: Built-in fake data creation for testing

## Prerequisites 📋

- Docker Engine ≥ 20.10
- NVIDIA CUDA-Supported GPU (30's+ series)
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
        "title": "Artist Name",
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
  -d '{"text": "Jazzy lo-fi beats for studying", "num_recommendations": 3}'
```

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

### Without GPU Support
```bash
docker run -d -p 8080:8080 --name recommender-cpu music-recommender
```

**Note**: This system uses synthetic data for demonstration purposes. For production use, replace with real music metadata and listening history data.

# Data Quality Tester
**Purpose**: Validates core integrity of music track metadata.

## Quick Start
```bash
python3 ./test/test_data_quality.py
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
python3 ./test/embedding_debug.py  
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
python3 ./test/test_model_performance.py
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

# Product-Scale Deployment

In order to serve 20 million active users, here's a future-looking technical roadmap to achieve production-quality at scale.

### **1. System Goals & Non-Functional Requirements**  
**Key Objectives:**  
- Serve 20M users (~2,315 RPS at 1 req/user/day)  
- Guarantee ≤250ms P99 latency for recommendation requests  
- Maintain 99.99% API availability (4 9's SLA)  
- Support zero-downtime deployments  

**Quantified Requirements:**  
| Metric | Target | Calculation Basis |  
|--------|--------|-------------------|  
| Throughput | 5,000 RPS | 2x peak traffic buffer |  
| Model Latency | ≤80ms | 30% latency budget allocation |  
| API Latency | ≤150ms | 60% budget for network + preprocessing |  
| Error Rate | <0.1% | SLA-compliant error budget |  
| Cold Start Time | <30s | Kubernetes pod initialization |  

**Critical Constraints:**  
- GPU memory optimization for batched inference  
- Stateless API design for horizontal scaling  
- Regional affinity routing to minimize latency  
- Model version rollback capability within 2 minutes  

---

### **2. Technical Architecture Components**  
**Core Architecture Diagram:**  
```
                            ┌───────────────────────┐
                            │ Global Load Balancer  │
                            │ (AWS ALB/GCP GLB)     │
                            └──────────┬───────────┘
                                       ▼
┌─────────────┐  API Calls  ┌───────────────────────┐  gRPC  ┌─────────────────┐
│  API Layer  │◄────────────┤   Inference Service   │───────►│ Model Registry   │
│ (FastAPI)   │             │ (NVIDIA Triton)       │        │ (MLflow/DVC)    │
└──────┬──────┘             └───────────────────────┘        └─────────────────┘
       │                              ▲                                ▲
       │ HTTP/2                       │ gRPC                          │ Model Pull
       ▼                              │                                │
┌──────────────┐             ┌────────┴─────────┐             ┌────────┴─────────┐
│  Edge Cache  │             │ Batch Inferencer │             │ Training Cluster │
│ (Varnish)    │             │ (Ray/KubeFlow)   │             │ (GPU Nodes)      │
└──────────────┘             └──────────────────┘             └──────────────────┘
```

**Key Components Breakdown:**  

1. **Model Serving Tier**  
   - **Triton Inference Server**: Deploy 8xA100 GPU nodes with:  
     - Dynamic batching (max batch size=256)  
     - Model ensembles for feature preprocessing  
     - FP16 quantization for latency reduction  
   - **Horizontal Scaling**: Auto-scaling group triggered by:  
     ```math
     \text{Scale Up If: } \frac{\text{GPU Memory Used}}{\text{Total Memory}} > 0.7 \text{ for 2m}
     ```

2. **API Tier**  
   - **Stateless FastAPI** pods (1,000 replicas):  
     - 1s timeout for Triton calls  
     - JWT validation via sidecar (Istio)  
   - **Redis Cache**:  
     - 3-layer cache strategy:  
       ```python
       # Pseudocode
       def get_recommendation(text):
           if in_local_LRU: return cached  
           elif in_redis: return redis.get()  
           else: triton_call().then(redis.set(ttl=60s))  
       ```

3. **Data Pipeline**  
   - **Real-Time Feature Store** (RedisTimeSeries):  
     - Track trending tracks/session frequency  
   - **Asynchronous Logging**:  
     ```bash
     fluentd → Kafka → Spark → S3 (hourly partitions)
     ```

4. **Network Optimization**  
   - **Protocol Choices**:  
     | Component       | Protocol   | Rationale               |
     |-----------------|------------|-------------------------|
     | Client → API    | HTTP/2     | Multiplexed connections |
     | API → Triton    | gRPC       | Binary payloads        |
     | Cache Updates   | UDP        | Tolerant to packet loss |

