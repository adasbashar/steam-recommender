import pandas as pd
import numpy as np
import time
import copy
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


# =======================
# Data Loading & Cleaning
# =======================

class DataLoader:
    """Handles dataset loading and initial preprocessing."""

    def __init__(self, file_path: str):
        """
        Initializes the DataLoader.

        :param file_path: Path to the CSV dataset file.
        """
        self.file_path = file_path

    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Loads and preprocesses the dataset.

        Performs:
        - Price normalization (free-to-play -> 0.0)
        - Combines genres and tags into a single feature column

        :return: Preprocessed DataFrame
        """
        start = time.time()
        df = pd.read_csv(self.file_path)
        df['price'] = df['price'].replace(
            to_replace=r'(?i)free\s*to\s*play|free',
            value=0,
            regex=True
        ).astype(float)
        df['genres_tags'] = df['genres'] + ' ' + df['tags']
        print(f"Data loaded in {time.time() - start:.2f}s")
        return df


# ===================
# User Profile Module
# ===================

class UserAnalyzer:
    """Analyzes user behavior and generates gaming profiles."""

    def __init__(self, df: pd.DataFrame):
        """
        Initializes with game interaction data.

        :param df: DataFrame containing user-game interactions
        """
        self.df = df

    def analyze_user(self, user_id: int) -> dict:
        """
        Generates a comprehensive user profile.

        :param user_id: Target user ID to analyze
        :return: Dictionary containing playtime, price stats, preferred devs,
                 played games, and weighted genre/tag profiles
        """
        start = time.time()
        user_data = self.df[self.df['user_id'] == user_id]
        print(f"User profile generated in {time.time() - start:.2f}s")
        return {
            'playtime': user_data['hours'].sum(),
            'avg_price': user_data['price'].mean(),
            'price_std': user_data['price'].std(),
            'preferred_devs': self._weighted_counter(user_data, 'developer'),
            'played_games': set(user_data['game_id']),
            'genre_profile': self._weighted_counter(user_data, 'genres'),
            'tag_profile': self._weighted_counter(user_data, 'tags')
        }

    def _weighted_counter(self, data: pd.DataFrame, column: str) -> Counter:
        """
        Creates a playtime-weighted counter for categorical features.

        :param data: Subset of user interaction data
        :param column: Feature column to analyze (developer/genres/tags)
        :return: Counter object with weighted counts
        """
        counter = Counter()
        for _, row in data.iterrows():
            items = row[column].split('|') if column in ['genres', 'tags'] else [row[column]]
            for item in items:
                counter[item] += row['hours']
        return counter

    def top_played_games(self, user_id: int, top_n: int = 3) -> list:
        """
        Identifies the most played games by playtime.

        :param user_id: Target user ID
        :param top_n: Number of top games to return
        :return: List of (game_title, hours) tuples
        """
        user_data = self.df[self.df['user_id'] == user_id]
        top_games = (
            user_data.groupby(['game_title'])['hours']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        return list(top_games.items())


# ==========================
# Recommender System Engine
# ==========================

class GameRecommender:
    """Hybrid recommender system combining multiple scoring strategies."""

    def __init__(self, df: pd.DataFrame):
        """
        Initializes recommendation models and pre-computes similarity matrices.

        :param df: DataFrame containing game metadata and interactions
        """
        self.df = df
        self.game_map = df.set_index('game_id')

        start = time.time()
        self.user_item = df.pivot_table(index='user_id', columns='game_id', values='hours', fill_value=0)
        self.game_id_to_col = {game_id: idx for idx, game_id in enumerate(self.user_item.columns)}
        print(f"User-item matrix built in {time.time() - start:.2f}s")

        start = time.time()
        self.content_sim = self._create_content_model()
        self.item_sim = self._create_collab_model()
        print(f"Similarity matrices computed in {time.time() - start:.2f}s")

    def _create_content_model(self) -> np.ndarray:
        """
        Creates a content-based similarity matrix using TF-IDF on genres/tags.

        :return: Dense cosine similarity matrix between games
        """
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'), stop_words='english')
        return cosine_similarity(tfidf.fit_transform(self.df['genres_tags']))

    def _create_collab_model(self):
        """
        Creates a collaborative filtering similarity matrix.

        :return: Sparse cosine similarity matrix between games
        """
        sparse_matrix = csr_matrix(self.user_item.values)
        return cosine_similarity(sparse_matrix.T, dense_output=False)

    def recommend(self, user_profile: dict, n: int = 5, weights: tuple = (0.4, 0.3, 0.2, 0.1), exclude_games: set = None) -> list:
        """
        Generates personalized game recommendations.

        :param user_profile: Dictionary from UserAnalyzer.analyze_user()
        :param n: Number of recommendations to return
        :param weights: Scoring weights for (content, collab, price, developer)
        :param exclude_games: Set of game IDs to exclude from results
        :return: List of recommendation dictionaries with explanations
        """
        start = time.time()
        content_scores = self._content_based_scores(user_profile)
        collab_scores = self._collab_based_scores(user_profile)
        price_scores = self._price_scores(user_profile)
        dev_scores = self._dev_scores(user_profile)
        combined = self._combine_scores(content_scores, collab_scores, price_scores, dev_scores, weights=weights)
        results = self._format_results(combined, user_profile, n, exclude_games)
        print(f"Recommendations generated in {time.time() - start:.2f}s")
        return results

    def _content_based_scores(self, profile: dict) -> Counter:
        """
        Scores games based on TF-IDF content similarity to the user's played games.

        :param profile: User profile dictionary
        :return: Counter of content-based scores
        """
        scores = Counter()
        indices = self.df[self.df['game_id'].isin(profile['played_games'])].index
        for idx in tqdm(indices, desc="Content-based scoring"):
            for game_idx, score in enumerate(self.content_sim[idx]):
                scores[game_idx] += score
        return scores

    def _collab_based_scores(self, profile: dict) -> Counter:
        """
        Scores games based on item-based collaborative filtering (cosine similarity).

        :param profile: User profile dictionary
        :return: Counter of collaborative filtering scores
        """
        scores = Counter()
        played_ids = [self.game_id_to_col[gid] for gid in profile['played_games'] if gid in self.game_id_to_col]
        for idx in tqdm(played_ids, desc="Collaborative filtering"):
            row = self.item_sim[idx]
            _, sim_indices = row.nonzero()
            for sim_idx in sim_indices:
                if sim_idx not in played_ids:
                    scores[sim_idx] += row[0, sim_idx]
        return scores

    def _price_scores(self, profile: dict) -> np.ndarray:
        """
        Scores games based on price compatibility with the user's spending patterns.

        :param profile: User profile dictionary
        :return: Array of price compatibility scores
        """
        price_diff = np.abs(self.df['price'] - profile['avg_price'])
        std = profile['price_std'] or 0.1
        return 1 / (1 + price_diff / std)

    def _dev_scores(self, profile: dict) -> pd.Series:
        """
        Scores games based on preferred developers (4x multiplier for top-3 devs).

        :param profile: User profile dictionary
        :return: Series of developer preference scores
        """
        top_devs = [d[0] for d in profile['preferred_devs'].most_common(3)]
        return self.df['developer'].apply(lambda x: 2 if x in top_devs else 0.5)

    def _combine_scores(self, *scores, weights: tuple) -> Counter:
        """
        Combines multiple scoring components with specified weights.

        :param scores: Variable number of score components
        :param weights: Corresponding weights for each component
        :return: Combined score Counter
        """
        combined = Counter()
        for score, weight in zip(scores, weights):
            for idx, val in (score.items() if isinstance(score, Counter) else enumerate(score)):
                combined[idx] += val * weight
        return combined

    def _format_results(self, scores: Counter, profile: dict, n: int, exclude_games: set = None) -> list:
        """
        Formats raw scores into readable recommendation dictionaries.

        :param scores: Combined score Counter
        :param profile: User profile dictionary
        :param n: Number of results to return
        :param exclude_games: Additional game IDs to exclude
        :return: List of formatted recommendation dictionaries
        """
        seen_games = profile['played_games'].union(exclude_games if exclude_games else set())
        recommended_ids = set()
        results = []

        for idx, _ in scores.most_common():
            game = self.df.iloc[idx]
            game_id = game['game_id']
            if game_id in seen_games or game_id in recommended_ids:
                continue
            results.append({
                'game_id': game_id,
                'title': game['game_title'],
                'price': game['price'],
                'developer': game['developer'],
                'genres': game['genres'],
                'reasons': self._generate_reasons(game, profile)
            })
            recommended_ids.add(game_id)
            if len(results) == n:
                break
        return results

    def _generate_reasons(self, game: pd.Series, profile: dict) -> list:
        """
        Generates human-readable justification for a recommendation.

        :param game: Game data row from DataFrame
        :param profile: User profile dictionary
        :return: List of explanation strings
        """
        reasons = []
        price_diff = game['price'] - profile['avg_price']
        if price_diff < -profile['price_std']:
            reasons.append(f"Budget pick (${game['price']} < avg ${profile['avg_price']:.2f})")
        elif price_diff > profile['price_std']:
            reasons.append(f"Premium title (${game['price']})")
        if game['developer'] in [d[0] for d in profile['preferred_devs'].most_common(3)]:
            reasons.append(f"From preferred developer: {game['developer']}")
        game_genres = set(game['genres'].split('|'))
        user_top = {g[0] for g in profile['genre_profile'].most_common(3)}
        if game_genres & user_top:
            reasons.append(f"Matches genres: {', '.join(game_genres & user_top)}")
        return reasons


# ================
# Main Program Run
# ================

def run(df: pd.DataFrame) -> None:
    """
    Main recommendation workflow for existing users with feedback loop.

    :param df: Preprocessed DataFrame
    """
    analyzer = UserAnalyzer(df)
    recommender = GameRecommender(df)

    try:
        user_ids = df['user_id'].unique()
        user_id = int(input(f"Enter your User ID (available IDs: {min(user_ids)} to {max(user_ids)}): "))

        if user_id not in user_ids:
            raise ValueError("User ID not found in the dataset.")

        base_profile = analyzer.analyze_user(user_id)
        session_profile = copy.deepcopy(base_profile)
        session_recommended = set()

        print(f"\n🧑‍💻 User {user_id} Profile Summary:")
        print(f"- Total Playtime:        {session_profile['playtime']:.2f} hours")
        print(f"- Average Spending:      ${session_profile['avg_price']:.2f}")
        print(f"- Price Sensitivity:     ${session_profile['price_std']:.2f} std")
        print(f"- Top Developers:        {[dev for dev, _ in session_profile['preferred_devs'].most_common(3)]}")
        print(f"- Top Genres:            {[genre for genre, _ in session_profile['genre_profile'].most_common(3)]}")
        print("- Top Played Games:")
        for title, hrs in analyzer.top_played_games(user_id):
            print(f"  • {title} ({hrs:.1f} hrs)")

        while True:
            recommendations = recommender.recommend(session_profile, exclude_games=session_recommended)

            print(f"\n🎮 Recommendations for User {user_id}:")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['title']}")
                print(f"   Price: ${rec['price']:.2f}  |  Developer: {rec['developer']}")
                print("   Why:")
                for reason in rec['reasons']:
                    print(f"   - {reason}")
                session_recommended.add(rec['game_id'])

            print("\n🔁 Feedback Options:")
            print("1. Like a recommendation")
            print("2. Dislike a recommendation")
            print("3. Show more recommendations")
            print("4. Exit")
            choice = input("Enter your choice (1-4): ").strip()

            if choice == '1':
                rec_num = int(input("Enter recommendation number to like: ")) - 1
                if 0 <= rec_num < len(recommendations):
                    liked = recommendations[rec_num]
                    for genre in liked['genres'].split('|'):
                        session_profile['genre_profile'][genre] += 20
                    session_profile['preferred_devs'][liked['developer']] += 20
                    print(f"👍 Updated preferences based on {liked['title']}")
                else:
                    print("❌ Invalid recommendation number")

            elif choice == '2':
                rec_num = int(input("Enter recommendation number to dislike: ")) - 1
                if 0 <= rec_num < len(recommendations):
                    disliked = recommendations[rec_num]
                    for genre in disliked['genres'].split('|'):
                        session_profile['genre_profile'][genre] = max(0, session_profile['genre_profile'][genre] - 20)
                    session_profile['preferred_devs'][disliked['developer']] = max(
                        0, session_profile['preferred_devs'][disliked['developer']] - 20
                    )
                    print(f"👎 Reduced preference for {disliked['title']} features")
                else:
                    print("❌ Invalid recommendation number")

            elif choice == '3':
                continue

            elif choice == '4':
                break

            else:
                print("❌ Invalid choice. Please enter 1-4.")

    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")
