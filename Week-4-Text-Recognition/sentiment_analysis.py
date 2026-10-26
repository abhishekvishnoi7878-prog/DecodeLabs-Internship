"""
Project 4: Text Recognition (Basic) - Sentiment Analysis
--------------------------------------------------------
Uses the pre-trained TextBlob library to classify text as
Positive, Negative or Neutral, with a polarity score and a
confidence estimate.

Install the dependency first:
    pip install textblob
"""

# ---------------------------------------------------------------------------
# STEP 0: Import the library
# ---------------------------------------------------------------------------
from textblob import TextBlob

# Polarity range for Neutral: scores between -0.1 and +0.1 count as Neutral
NEUTRAL_THRESHOLD = 0.1

# Sample sentences for the demo mode
SAMPLE_TEXTS = [
    "I absolutely love this product, it works perfectly!",
    "This is terrible, I am very disappointed with the awful service.",
    "The meeting is scheduled for 3 PM on Monday.",
    "The food was okay, but the waiting time was terrible.",
    "What a fantastic and inspiring day!",
]


# ---------------------------------------------------------------------------
# STEP 1: Core analysis function
# ---------------------------------------------------------------------------
def analyze_sentiment(text):
    """Return (label, polarity, subjectivity, confidence) for the given text."""
    # TextBlob processes the text with a pre-trained, lexicon-based model
    blob = TextBlob(text)

    # Polarity: -1.0 (very negative) to +1.0 (very positive)
    polarity = blob.sentiment.polarity
    # Subjectivity: 0.0 (factual) to 1.0 (opinionated)
    subjectivity = blob.sentiment.subjectivity

    # Classify the polarity score into a category
    if polarity > NEUTRAL_THRESHOLD:
        label = "Positive"
    elif polarity < -NEUTRAL_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    # Confidence: how far the score is from zero, shown as a percentage.
    # This is a simple estimate, not a statistical probability.
    confidence = abs(polarity) * 100

    return label, polarity, subjectivity, confidence


# ---------------------------------------------------------------------------
# STEP 2: Display function
# ---------------------------------------------------------------------------
def display_result(text):
    """Analyze a text and print the result in a clean format."""
    label, polarity, subjectivity, confidence = analyze_sentiment(text)

    print("-" * 55)
    print(f"Text         : {text}")
    print(f"Sentiment    : {label}")
    print(f"Polarity     : {polarity:+.2f}  (-1 negative ... +1 positive)")
    print(f"Subjectivity : {subjectivity:.2f}  (0 factual ... 1 opinion)")
    print(f"Confidence   : {confidence:.0f}%")
    print("-" * 55)


# ---------------------------------------------------------------------------
# STEP 3: Run the program
# ---------------------------------------------------------------------------
def main():
    print("=== Sentiment Analysis (TextBlob) ===")

    # Part A: Process the built-in sample texts
    print("\n[Part A] Sample text analysis\n")
    for sample in SAMPLE_TEXTS:
        display_result(sample)

    # Part B: Interactive mode for user input
    print("\n[Part B] Try your own text. Type 'exit' to quit.\n")
    while True:
        # Sanitize input by trimming surrounding whitespace
        user_text = input("Enter text: ").strip()

        # Exit condition
        if user_text.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        # Reject empty input
        if not user_text:
            print("Please enter some text.\n")
            continue

        display_result(user_text)


if __name__ == "__main__":
    main()
