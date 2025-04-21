from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the FinBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Function to analyze sentiment of a news headline or paragraph
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()
    return ["Negative", "Neutral", "Positive"][sentiment]

# Example usage:
if __name__ == "__main__":
    sample_text = "Bitcoin hits an all-time high amid growing investor interest."
    sentiment = analyze_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")
