import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from time import time

print("===== Music Track Similarity Analysis =====")
print("Initializing...\n")

# Configuration
SAMPLE_SIZE = 100
TEST_QUERY = "Uplifting EDM track with hype vocals and a build-up structure"
TEST_MATCHING_MOOD = "energetic"

# Load data
print("[1/6] 📂 Loading dataset...")
df = pd.read_csv("data/raw/fake_tracks.csv")
energetic_tracks = df[df.mood == TEST_MATCHING_MOOD].head(SAMPLE_SIZE)
print(f"✅ Dataset loaded with {len(energetic_tracks)} energetic tracks")

def analyze_model(model, model_name, descriptions, query):
    print(f"\n🔍 Analyzing with {model_name}...")
    
    # Encoding with progress bar
    print("📥 Encoding text...")
    start = time()
    query_embed = model.encode(query, convert_to_tensor=True, device="cuda")
    track_embeds = model.encode(descriptions, convert_to_tensor=True, device="cuda", 
                              show_progress_bar=True)
    
    # Calculate similarities
    print("🧮 Calculating similarities...")
    cosine_scores = util.cos_sim(query_embed, track_embeds).cpu().numpy().flatten()
    
    # Statistics
    stats = {
        'mean': np.mean(cosine_scores),
        'std': np.std(cosine_scores),
        'max': np.max(cosine_scores),
        'min': np.min(cosine_scores),
        'median': np.median(cosine_scores),
        'time': time() - start
    }
    
    return cosine_scores, stats

# Initialize models
print("\n[2/6] 🚀 Loading models...")
models = {
    "Tuned Model": SentenceTransformer("models/mood-encoder", device="cuda"),
    "Base Model": SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
}

# Run analysis
results = {}
for name, model in models.items():
    scores, stats = analyze_model(model, name, energetic_tracks['description'].tolist(), TEST_QUERY)
    results[name] = {'scores': scores, 'stats': stats}
    print(f"\n📊 {name} Statistics:")
    for k, v in stats.items():
        print(f"{k.upper():<8}: {v:.4f}" if isinstance(v, float) else f"{k.upper():<8}: {v}")

# Visualization
print("\n[3/6] 📈 Generating visualization...")
plt.figure(figsize=(10, 6))
for name, result in results.items():
    plt.hist(result['scores'], alpha=0.5, label=name, bins=20)

plt.axvline(x=0.7, color='r', linestyle='--', label='Strong Match Threshold')
plt.axvline(x=0.4, color='y', linestyle='--', label='Weak Match Threshold')
plt.title("Cosine Similarity Distribution")
plt.xlabel("Similarity Score")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("similarity_distribution.png")
print("✅ Saved visualization to similarity_distribution.png")

# Comparative analysis
print("\n[4/6] 📊 Comparative Analysis:")
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Mean Score': [x['stats']['mean'] for x in results.values()],
    'Score Range': [f"{x['stats']['min']:.2f}-{x['stats']['max']:.2f}" for x in results.values()],
    'Processing Time (s)': [x['stats']['time'] for x in results.values()]
})
print(comparison)

print("\n[5/6] 💡 Interpretation Guidelines:")
print("• Scores >0.7: Strong matches (likely good recommendations)")
print("• Scores 0.5-0.7: Moderate matches")
print("• Scores <0.4: Weak/irrelevant matches")

print("\n[6/6] 🎉 Analysis complete!")
print(f"• Tuned model {'outperformed' if results['Tuned Model']['stats']['mean'] > results['Base Model']['stats']['mean'] else 'underperformed'} base model")
print(f"• Found {sum(results['Tuned Model']['scores'] > 0.7)} strong matches (tuned) vs {sum(results['Base Model']['scores'] > 0.7)} (base)")