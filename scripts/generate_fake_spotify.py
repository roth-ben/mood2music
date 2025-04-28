import argparse
import pandas as pd
import random
from faker import Faker
from typing import List, Dict

# Initialize Faker with custom music providers
fake = Faker()
fake.add_provider("music")
random.seed(42)

MOOD_PROFILES = {
    "energetic": {
        "genres": ["EDM", "techno", "drum & bass", "hardstyle", "electro house"],
        "instruments": ["synth", "drums", "electric guitar", "bass", "brass"],
        "vocals": ["powerful female", "chanting male", "choir", "rap", "hype vocals"],
        "bpm_range": (120, 180),
        "descriptors": ["high-energy", "pumping", "festival-ready", "explosive", "uplifting"],
        "structure": ["build-up", "drop", "breakdown", "outro"]
    },
    "calm": {
        "genres": ["lofi hip-hop", "ambient", "new age", "classical", "jazz"],
        "instruments": ["piano", "acoustic guitar", "flute", "strings", "harp"],
        "vocals": ["soft female", "whispered male", "nature sounds", "no vocals"],
        "bpm_range": (60, 100),
        "descriptors": ["soothing", "meditative", "peaceful", "gentle", "relaxing"],
        "structure": ["loop-based", "freeform", "gradual progression"]
    },
    "melancholic": {
        "genres": ["dark wave", "post-rock", "neoclassical", "blues", "soul"],
        "instruments": ["cello", "piano", "violin", "theremin", "music box"],
        "vocals": ["haunting female", "raspy male", "whispered", "choir"],
        "bpm_range": (40, 80),
        "descriptors": ["brooding", "nostalgic", "cinematic", "emotional", "atmospheric"],
        "structure": ["verse/chorus", "freeform", "crescendo"]
    }
}

def generate_track_description(mood: str) -> str:
    """Generate musically coherent track description with mood-specific elements"""
    profile = MOOD_PROFILES[mood]
    return (
        f"{random.choice(profile['descriptors']).title()} "
        f"{random.choice(profile['genres'])} track. "
        f"Features {random.choice(profile['instruments'])} with "
        f"{random.choice(profile['vocals'])} vocals. "
        f"BPM: {random.randint(*profile['bpm_range'])}. "
        f"Structure: {random.choice(profile['structure'])}. "
        f"Key: {fake.random_element(['A minor', 'C# minor', 'D major', 'F minor'])}."
    )

def generate_tracks(tracks_per_mood: int) -> List[Dict]:
    """Generate synthetic tracks with professional music metadata"""
    tracks = []
    for mood in MOOD_PROFILES.keys():
        for _ in range(tracks_per_mood):
            track = {
                "id": f"spotify:track:{fake.uuid4()}",
                "title": fake.catch_phrase().title(),
                "artist": fake.name(),
                "mood": mood,
                "mood_score": random.uniform(0.8 if mood == "energetic" else 0.5, 1.0),
                "danceability": random.uniform(
                    0.7 if mood == "energetic" else 0.3,
                    0.95 if mood == "energetic" else 0.6
                ),
                "valence": random.uniform(
                    0.8 if mood == "energetic" else 0.2,
                    1.0 if mood == "energetic" else 0.4
                ),
                "energy": random.uniform(0.75, 0.98) if mood == "energetic" else random.uniform(0.2, 0.6),
                "description": generate_track_description(mood),
                "duration_ms": random.randint(180000, 360000),  # 3-6 minutes
                "explicit": random.choice([True, False]),
                "release_date": fake.date_between(start_date="-5y", end_date="today").isoformat()
            }
            tracks.append(track)
    return tracks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate professional-grade synthetic music tracks")
    parser.add_argument("--tracks-per-mood", type=int, default=300,
                       help="Number of tracks to generate per mood category")
    parser.add_argument("--output", type=str, default="data/raw/fake_tracks.csv",
                       help="Output path for generated tracks")
    
    args = parser.parse_args()
    
    tracks = generate_tracks(args.tracks_per_mood)
    df = pd.DataFrame(tracks)
    
    # Ensure professional quality control
    df = df.drop_duplicates(subset=["description", "title"])
    df = df.sort_values(by=["mood", "energy"], ascending=[True, False])
    
    df.to_csv(args.output, index=False)
    print(f"Successfully generated {len(df)} professional tracks at {args.output}")
    print("\nExample Track:")
    print(df.iloc[0].to_string())