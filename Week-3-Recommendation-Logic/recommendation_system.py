"""
Project 3: AI Recommendation Logic
----------------------------------
A content-based movie recommender. It compares the user's interests against
each movie's genre and description using TF-IDF text vectors and cosine
similarity, then displays the best matches.
"""

# ---------------------------------------------------------------------------
# STEP 0: Import libraries
# ---------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# STEP 1: Create the dataset
# Each item has a title, a genre and a short description.
# ---------------------------------------------------------------------------
MOVIES = [
    {"title": "Interstellar", "genre": "sci-fi",
     "description": "astronauts travel through a wormhole in space to save humanity"},
    {"title": "The Martian", "genre": "sci-fi",
     "description": "an astronaut stranded on mars uses science to survive"},
    {"title": "Inception", "genre": "sci-fi thriller",
     "description": "a thief enters dreams to plant an idea in a target's mind"},
    {"title": "The Dark Knight", "genre": "action crime",
     "description": "a vigilante hero fights a chaotic criminal mastermind in the city"},
    {"title": "Mad Max: Fury Road", "genre": "action adventure",
     "description": "a desert chase across a wasteland with fast cars and rebels"},
    {"title": "The Notebook", "genre": "romance drama",
     "description": "a young couple falls in love and is separated by class and time"},
    {"title": "La La Land", "genre": "romance musical",
     "description": "a musician and an actress chase their dreams and fall in love"},
    {"title": "The Hangover", "genre": "comedy",
     "description": "friends wake up after a wild bachelor party in Las Vegas and lose the groom"},
    {"title": "Superbad", "genre": "comedy",
     "description": "two high school friends try to enjoy one last party before graduation"},
    {"title": "The Conjuring", "genre": "horror",
     "description": "paranormal investigators help a family haunted by a dark presence"},
    {"title": "A Quiet Place", "genre": "horror thriller",
     "description": "a family survives in silence while creatures hunt by sound"},
    {"title": "The Shawshank Redemption", "genre": "drama",
     "description": "a banker wrongly imprisoned finds hope and friendship in prison"},
    {"title": "Spirited Away", "genre": "animation fantasy",
     "description": "a girl enters a magical spirit world to rescue her parents"},
    {"title": "Toy Story", "genre": "animation family",
     "description": "toys come to life and go on an adventure to find their way home"},
    {"title": "The Imitation Game", "genre": "biography drama",
     "description": "a mathematician cracks a secret enemy code during world war two"},
]

# Minimum similarity score for an item to count as a match (0 to 1)
MIN_SCORE = 0.05
# Maximum number of recommendations to show
TOP_N = 5


# ---------------------------------------------------------------------------
# STEP 2: Prepare the text and build the TF-IDF model
# For each movie we combine genre + description into one searchable string.
# The genre is repeated so that genre matches carry more weight.
# ---------------------------------------------------------------------------
movie_texts = [f"{m['genre']} {m['genre']} {m['description']}" for m in MOVIES]

# TF-IDF converts text into numeric vectors; common words get lower weight
vectorizer = TfidfVectorizer(stop_words="english")
movie_vectors = vectorizer.fit_transform(movie_texts)


# ---------------------------------------------------------------------------
# STEP 3: Recommendation logic
# ---------------------------------------------------------------------------
def recommend(user_input):
    """Return a list of (movie, score) pairs sorted by best match first."""
    # Convert the user's text into the same vector space as the movies
    user_vector = vectorizer.transform([user_input.lower().strip()])

    # Cosine similarity: 1.0 = identical direction, 0.0 = nothing in common
    scores = cosine_similarity(user_vector, movie_vectors).flatten()

    # Pair each movie with its score
    results = list(zip(MOVIES, scores))

    # Filter: keep only movies above the minimum similarity threshold
    results = [(movie, score) for movie, score in results if score >= MIN_SCORE]

    # Rank: highest score first, then keep the top N
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results[:TOP_N]


def show_recommendations(results):
    """Print the recommendations as a clean numbered list."""
    print("\nRecommended for you:")
    print("-" * 50)
    for rank, (movie, score) in enumerate(results, start=1):
        print(f"{rank}. {movie['title']}  [{movie['genre']}]")
        print(f"   Match: {score * 100:.0f}%")
        desc = movie["description"]
        print(f"   {desc[0].upper() + desc[1:]}.")
    print("-" * 50)


def available_genres():
    """Collect every unique genre word in the dataset (for help messages)."""
    genres = set()
    for movie in MOVIES:
        genres.update(movie["genre"].split())
    return sorted(genres)


# ---------------------------------------------------------------------------
# STEP 4: Interact with the user
# ---------------------------------------------------------------------------
def main():
    print("=== Movie Recommendation System ===")
    print("Available genres:", ", ".join(available_genres()))
    print("Type a genre or interest (e.g. 'space adventure', 'funny', "
          "'love story'). Type 'exit' to quit.")

    while True:
        user_input = input("\nWhat are you in the mood for? ").strip()

        # Exit condition
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Enjoy your movie. Goodbye!")
            break

        # Reject empty input
        if not user_input:
            print("Please enter a genre or interest.")
            continue

        results = recommend(user_input)

        # Fallback when nothing matches
        if not results:
            print("No matches found. Try one of these genres:",
                  ", ".join(available_genres()))
        else:
            show_recommendations(results)


if __name__ == "__main__":
    main()
