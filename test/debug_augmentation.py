from nlpaug.augmenter.word import SynonymAug, BackTranslationAug, RandomWordAug
from nlpaug.flow import Sequential
import torch
import nltk
from transformers import logging

# Initial setup
logging.set_verbosity_error()
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.empty_cache()

# Valid BackTranslationAug parameters only
aug = Sequential([
    RandomWordAug(action="swap", aug_p=0.1),  # Word shuffling
    SynonymAug(
        aug_max=2,
        aug_p=0.5,
        stopwords=["no", "not", "without"]
    ),
    BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-de',
        to_model_name='Helsinki-NLP/opus-mt-de-en',
        device=device,
        batch_size=1,
        max_length=128,  # Maximum tokens for translation
        force_reload=False  # Use cached models
    )
])

def safe_augment(text):
    try:
        result = aug.augment(text, n=3)
        return result[0] if isinstance(result, list) else result
    except Exception as e:
        print(f"Augmentation warning: {str(e)}")
        return text

if __name__ == "__main__":
    sample = "Soothing ambient track. Features strings with no vocals."
    print(f"Original: {sample}")
    print(f"Augmented: {safe_augment(sample)}")