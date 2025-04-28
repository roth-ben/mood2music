import pandas as pd

def test_data_quality():
    """Validate core data quality metrics for music tracks dataset"""
    print("=== Music Data Quality Test ===")
    
    try:
        # 1. Load Data
        df = pd.read_csv("data/raw/fake_tracks.csv")
        print(f"✅ Loaded {len(df)} tracks")
        
        # 2. Basic Validation
        required_columns = {'title', 'mood', 'danceability', 'valence', 'energy'}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # 3. Sample Inspection
        print("\n🎧 Random Samples:")
        print(df[["title", "mood", "danceability"]].sample(5))
        
        # 4. Value Ranges
        print("\n📊 Value Ranges:")
        for col in ['danceability', 'valence', 'energy']:
            print(f"{col}: {df[col].min():.2f}-{df[col].max():.2f}")
        
        # 5. Mood Distribution
        print("\n😊 Mood Distribution:")
        print(df['mood'].value_counts())
        
        # 6. Null Checks
        print("\n🔍 Missing Values:")
        print(df.isnull().sum())
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_data_quality()