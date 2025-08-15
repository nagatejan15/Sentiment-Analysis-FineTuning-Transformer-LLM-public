import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH")

print(f"Loading model from: {MODEL_PATH}")
try:
    sentiment_classifier = pipeline("sentiment-analysis", model=MODEL_PATH)
except OSError:
    print(f" Error: Model not found at {MODEL_PATH}.")
    print("Please run the training script `src/train.py` first.")
    exit()

print(" Model loaded successfully.")

test_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the plot was gripping.",
    "I was really disappointed with this film. It was boring and the ending made no sense.",
    "The film was okay, not great but not terrible either. Just average.",
    "A masterpiece of modern cinema. I was on the edge of my seat the entire time.",
    "I want to say film is not soo good and not thrilling."
]

print("\n--- Running Inference ---")
results = sentiment_classifier(test_reviews)

for review, result in zip(test_reviews, results):
    label = "POSITIVE" if result['label'] == 'LABEL_1' else "NEGATIVE"
    score = result['score']
    print(f"Review: \"{review}\"")
    print(f"Predicted Sentiment: {label} (Confidence: {score:.4f})\n")
