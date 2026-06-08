from scipy.stats import norm
import pandas as pd
from typing import Set, List, Tuple


# ===========================
# Core Scoring Utilities
# ===========================

def wilson_score(pos: int, total: int, confidence: float = 0.80) -> float:
    """
    Calculate Wilson lower bound score for binomial proportion confidence interval.

    Parameters:
        pos (int): Number of positive interactions
        total (int): Total number of interactions
        confidence (float): Confidence level (default 0.80)

    Returns:
        float: Wilson score between 0 and 1
    """
    if total == 0:
        return 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = pos / total
    numerator = phat + z**2 / (2 * total) - z * ((phat * (1 - phat) + z**2 / (4 * total)) / total) ** 0.5
    denominator = 1 + z**2 / total
    return numerator / denominator


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Parameters:
        set1 (Set[str]): First set of elements
        set2 (Set[str]): Second set of elements

    Returns:
        float: Jaccard similarity score between 0 and 1
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0


# ===========================
# Data Processing Functions
# ===========================

def extract_all_genres(df: pd.DataFrame) -> List[str]:
    """
    Extract unique genres from DataFrame's 'genres' column.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'genres' column

    Returns:
        List[str]: Sorted list of unique lowercase genres
    """
    genre_set = set()
    for genre_str in df['genres'].dropna():
        genre_set.update([g.strip().lower() for g in genre_str.split('|')])
    return sorted(genre_set)


def prompt_user_selected_genres(available_genres: List[str]) -> List[str]:
    """
    Prompt user to select genres and validate input.

    Parameters:
        available_genres (List[str]): List of valid genre strings

    Returns:
        List[str]: Validated user-selected genres (lowercase)
    """
    print("🎮 Available Genres:")
    print(", ".join(available_genres))
    user_input = input("\n🧠 What genres are you interested in? (comma-separated): ")
    selected = [g.strip().lower() for g in user_input.split(',')]
    valid_selected = [g for g in selected if g in available_genres]

    if not valid_selected:
        print("⚠️  No valid genres selected.")
    return valid_selected


# ===========================
# Recommendation Engine
# ===========================

def compute_hybrid_scored_games(
    df: pd.DataFrame,
    selected_genres: List[str],
    w1: float = 0.1,
    w2: float = 0.9
) -> pd.DataFrame:
    """
    Calculate hybrid scores for games based on genre similarity and Wilson score.

    Parameters:
        df (pd.DataFrame): Source DataFrame with game data
        selected_genres (List[str]): User's preferred genres
        w1 (float): Weight for Jaccard similarity (default 0.1)
        w2 (float): Weight for Wilson score (default 0.9)

    Returns:
        pd.DataFrame: Games sorted by hybrid score descending
    """
    selected_set = set(selected_genres)
    results = []

    for game_id, group in df.groupby('game_id'):
        game = group.iloc[0]
        game_genres = set(str(game['genres']).lower().split('|'))

        jacc = jaccard_similarity(selected_set, game_genres)
        if jacc == 0:
            continue

        pos = group['normalized_binary'].sum()
        total = group['normalized_binary'].count()
        ws = wilson_score(pos, total)

        results.append({
            'game_id': game_id,
            'game_title': game['game_title'],
            'genres': game['genres'],
            'price': game['price'],
            'metascore': game['metascore'],
            'total_players': total,
            'jaccard_similarity': jacc,
            'wilson_score': ws,
            'final_score': w1 * jacc + w2 * ws
        })

    return pd.DataFrame(results).sort_values('final_score', ascending=False)


def get_top_diverse_games(df: pd.DataFrame, exclude_ids: Set[str], top_n: int = 5) -> pd.DataFrame:
    """
    Get top diverse games based on Wilson score, excluding specified IDs.

    Parameters:
        df (pd.DataFrame): Source DataFrame
        exclude_ids (Set[str]): Game IDs to exclude
        top_n (int): Number of results to return (default 5)

    Returns:
        pd.DataFrame: Top N games sorted by Wilson score
    """
    game_stats = df.groupby('game_id').agg(
        pos=('normalized_binary', 'sum'),
        total=('normalized_binary', 'count')
    ).reset_index()

    game_stats['wilson_score'] = game_stats.apply(
        lambda row: wilson_score(row['pos'], row['total']), axis=1
    )

    game_info = df[['game_id', 'game_title', 'genres', 'price', 'metascore']].drop_duplicates()
    merged = game_stats.merge(game_info, on='game_id')

    return (
        merged[~merged['game_id'].isin(exclude_ids)]
        .sort_values('wilson_score', ascending=False)
        .head(top_n)
    )


# ===========================
# Interface & Feedback
# ===========================

def display_recommendations(main_df: pd.DataFrame, extra_df: pd.DataFrame) -> None:
    """
    Display recommendation results in formatted output.

    Parameters:
        main_df (pd.DataFrame): Primary recommendations
        extra_df (pd.DataFrame): Bonus recommendations
    """
    print("\n📌 Top Game Recommendations Based on Your Favorite Genres:\n" + "=" * 80)
    for i, (_, row) in enumerate(main_df.iterrows(), 1):
        print(f"{i}. 🎮 {row['game_title']}")
        print(f"   Genres:               {row['genres']}")
        print(f"   Jaccard Similarity:   {row['jaccard_similarity']:.2f}")
        print(f"   Wilson Score:         {row['wilson_score']:.4f}")
        print(f"   Price:                {row['price']}")
        print("-" * 80)

    print("\n🎁 Bonus Recommendations You Might Like:\n" + "=" * 80)
    start_num = len(main_df) + 1
    for i, (_, row) in enumerate(extra_df.iterrows(), start_num):
        print(f"{i}. 🎮 {row['game_title']}")
        print(f"   Genres:             {row['genres']}")
        print(f"   Wilson Score:       {row['wilson_score']:.4f}")
        print(f"   Price:              {row['price']}")
        print("-" * 80)


def process_feedback(recommendations: pd.DataFrame, current_genres: Set[str]) -> Tuple[Set[str], bool]:
    """
    Process user feedback and update genre preferences.

    Parameters:
        recommendations (pd.DataFrame): Shown recommendations
        current_genres (Set[str]): Current preferred genres

    Returns:
        Tuple[Set[str], bool]: Updated genres and exit flag
    """
    liked_genres = set()
    disliked_genres = set()

    while True:
        print("\n🔁 Feedback Options:")
        print("1. Like a recommendation")
        print("2. Dislike a recommendation")
        print("3. Show more recommendations")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()

        if choice in ('1', '2'):
            try:
                rec_num = int(input("Enter recommendation number: ")) - 1
                if 0 <= rec_num < len(recommendations):
                    game = recommendations.iloc[rec_num]
                    genres = set(g.strip().lower() for g in game['genres'].split('|'))
                    if choice == '1':
                        liked_genres.update(genres)
                        print(f"👍 Added {', '.join(genres)} to preferences")
                    else:
                        disliked_genres.update(genres)
                        print(f"👎 Removed {', '.join(genres)} from preferences")
                else:
                    print("Invalid recommendation number.")
            except ValueError:
                print("Please enter a valid number.")
        elif choice == '3':
            return (current_genres.union(liked_genres) - disliked_genres, False)
        elif choice == '4':
            return (current_genres, True)
        else:
            print("Invalid choice. Please enter 1-4.")


# ===========================
# Main Recommendation Flow
# ===========================

def run(df: pd.DataFrame, top_n: int = 5) -> None:
    """
    Main recommendation workflow for new users with feedback loop.

    Parameters:
        df (pd.DataFrame): Source game data
        top_n (int): Number of main recommendations (default 5)
    """
    available_genres = extract_all_genres(df)
    selected_genres = prompt_user_selected_genres(available_genres)

    if not selected_genres:
        return

    current_genres = set(selected_genres)
    shown_ids = set()
    exit_flag = False

    while not exit_flag:
        hybrid_df = compute_hybrid_scored_games(df, list(current_genres))
        hybrid_df = hybrid_df[~hybrid_df['game_id'].isin(shown_ids)]

        if hybrid_df.empty:
            print("❌ No more games matching your preferences.")
            break

        main_recommendations = hybrid_df.head(top_n)
        shown_ids.update(main_recommendations['game_id'])

        bonus_recommendations = get_top_diverse_games(df, shown_ids)
        shown_ids.update(bonus_recommendations['game_id'])

        display_recommendations(main_recommendations, bonus_recommendations)
        current_genres, exit_flag = process_feedback(
            pd.concat([main_recommendations, bonus_recommendations]),
            current_genres
        )
