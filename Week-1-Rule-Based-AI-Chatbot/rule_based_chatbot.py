"""
Rule-Based AI Chatbot
---------------------
A simple chatbot that matches user input against keywords stored in a
dictionary (the Knowledge Base) and replies with a predefined response.
No machine learning involved: pure rule-based pattern matching.
"""

import random
import string
from datetime import datetime

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE
# Each intent has:
#   "keywords"  -> words/phrases that trigger this intent
#   "responses" -> possible replies (one is picked at random)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = {
    "greeting": {
        "keywords": ["hello", "hi", "hey", "good morning", "good evening"],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! Nice to see you.",
        ],
    },
    "status": {
        "keywords": ["how are you", "how are you doing", "whats up", "status"],
        "responses": [
            "I'm running smoothly, thanks for asking!",
            "All systems operational. How about you?",
        ],
    },
    "identity": {
        "keywords": ["who are you", "your name", "what are you", "introduce yourself"],
        "responses": [
            "I'm a rule-based chatbot built in Python.",
            "I'm PyBot, a simple AI assistant that answers using predefined rules.",
        ],
    },
    "time": {
        "keywords": ["time", "date", "what time is it", "today"],
        "responses": ["Here's the current date and time:"],  # time is appended in code
    },
    "help": {
        "keywords": ["help", "assist", "support", "what can i ask"],
        "responses": [
            "You can greet me, ask my name, ask for the time, "
            "check how I'm doing, or ask about my features. Type 'exit' to quit.",
        ],
    },
    "features": {
        "keywords": ["features", "what can you do", "abilities", "capabilities"],
        "responses": [
            "I can chat, tell you the time, introduce myself, "
            "and respond to common questions using keyword rules.",
        ],
    },
    "thanks": {
        "keywords": ["thanks", "thank you", "thx"],
        "responses": ["You're welcome!", "Happy to help!"],
    },
    "exit": {
        "keywords": ["exit", "bye", "quit", "goodbye"],
        "responses": ["Goodbye! Have a great day.", "Bye! Talk to you soon."],
    },
}

# Polite reply when no intent matches
FALLBACK_RESPONSES = [
    "Sorry, I didn't understand that. Could you rephrase it?",
    "Hmm, I don't have an answer for that yet. Type 'help' to see what I can do.",
]


def clean_input(text):
    """Sanitize input: lowercase, trim whitespace, remove punctuation."""
    text = text.lower().strip()
    # Remove punctuation so "Hello!" matches "hello"
    return text.translate(str.maketrans("", "", string.punctuation))


def find_intent(user_text):
    """
    Return the name of the first intent whose keyword appears in the input,
    or None if nothing matches.
    """
    # Pad with spaces so we match whole words only
    # (prevents "hi" from matching inside "this")
    padded_text = f" {user_text} "

    for intent, data in KNOWLEDGE_BASE.items():
        for keyword in data["keywords"]:
            if f" {keyword} " in padded_text:
                return intent
    return None


def get_response(intent):
    """Pick a response for the matched intent (or the fallback)."""
    if intent is None:
        return random.choice(FALLBACK_RESPONSES)

    reply = random.choice(KNOWLEDGE_BASE[intent]["responses"])

    # Special case: the time intent needs live data
    if intent == "time":
        reply += " " + datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    return reply


def main():
    print("PyBot: Hi! I'm a rule-based chatbot. Type 'help' for options, 'exit' to quit.")

    # Main loop: runs until the user triggers the exit intent
    while True:
        try:
            raw_input_text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            # Handles Ctrl+C / Ctrl+D gracefully
            print("\nPyBot: Session ended. Goodbye!")
            break

        user_text = clean_input(raw_input_text)

        # Ignore empty messages
        if not user_text:
            print("PyBot: Please type something.")
            continue

        intent = find_intent(user_text)
        print("PyBot:", get_response(intent))

        # Exit condition: stop the loop after saying goodbye
        if intent == "exit":
            break


if __name__ == "__main__":
    main()
