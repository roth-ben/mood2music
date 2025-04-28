import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

def embedding_debug():
    # 1. Data Loading
    try:
        df = pd.read_csv("data/raw/fake_tracks.csv")
        print(f"✅ Data: Loaded {len(df)} tracks | {len(df['description'].unique())} unique descriptions")
        if len(df) < 10:
            print("⚠️ Warning: Very small dataset (<10 samples)")
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return

    # 2. Model Initialization
    try:
        model = SentenceTransformer("models/mood-encoder", device="cuda")
        # Safer way to get model info
        model_info = {
            'device': model.device,
            'max_seq_length': model.max_seq_length,
            'embedding_dim': model.get_sentence_embedding_dimension()
        }
        print(f"✅ Model loaded successfully")
        print(f"   - Device: {model_info['device']}")
        print(f"   - Embedding dim: {model_info['embedding_dim']}")
        print(f"   - Max seq length: {model_info['max_seq_length']}")
        
        # Test embedding
        test_embed = model.encode("test sentence", convert_to_tensor=True)
        print(f"🔍 Test embedding norm: {torch.norm(test_embed):.2f}")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        if "No such file or directory" in str(e):
            print("   → Hint: Check if 'models/mood-encoder' exists and contains:")
            print("     1. config.json")
            print("     2. pytorch_model.bin")
            print("     3. tokenizer files")
        return

    # 3. Embedding Generation
    try:
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            descriptions = df["description"].tolist()
            print(f"\nGenerating embeddings for {len(descriptions)} descriptions...")
            
            song_embeddings = model.encode(
                descriptions,
                convert_to_tensor=True,
                device="cuda",
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            # Validation checks
            print("\n🔬 Embedding Validation:")
            print(f"- Shape: {song_embeddings.shape} (samples × embedding_dim)")
            print(f"- Mean norm: {torch.norm(song_embeddings, dim=1).mean():.2f}")
            
            if torch.isnan(song_embeddings).any():
                print("❌ Found NaN values in embeddings!")
            if torch.isinf(song_embeddings).any():
                print("❌ Found infinite values in embeddings!")
                
    except Exception as e:
        print(f"❌ Embedding generation failed: {str(e)}")
        return

    # 4. Similarity Analysis
    try:
        print("\n📊 Running similarity analysis...")
        cos_sim = util.cos_sim(song_embeddings, song_embeddings)
        
        # Exclude self-similarity (diagonal)
        mask = ~torch.eye(len(cos_sim), dtype=torch.bool, device=cos_sim.device)
        filtered_scores = cos_sim[mask]
        
        stats = {
            'unique_scores': torch.unique(cos_sim).cpu().numpy(),
            'mean': filtered_scores.mean().item(),
            'std': filtered_scores.std().item(),
            'min': filtered_scores.min().item(),
            'max': filtered_scores.max().item()
        }
        
        print("\nSimilarity Matrix Results:")
        print(f"• Score range: {stats['min']:.3f} to {stats['max']:.3f}")
        print(f"• Mean similarity: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"• Unique values: {len(stats['unique_scores'])}")
        
        # Interpretation guide
        print("\n💡 Interpretation Guide:")
        print("0.9+ : Very similar (possible duplicates)")
        print("0.7-0.9: Strong similarity")
        print("0.5-0.7: Moderate similarity")
        print("<0.5: Low similarity")
        
    except Exception as e:
        print(f"❌ Similarity analysis failed: {str(e)}")

if __name__ == "__main__":
    print("===== Embedding Quality Debugger =====")
    print("Checking model, data, and embedding quality...\n")
    embedding_debug()