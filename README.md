# Steam Game Recommender System

A hybrid recommendation system built on a dataset of 2,000 Steam users. The system handles both new and existing users through separate pipelines, combining content-based filtering, collaborative filtering, and behavioral signals into a single weighted score.

---

## Features

- **New user (cold-start)**: Genre-based recommendations using Jaccard similarity and Wilson score confidence intervals — no prior history needed
- **Existing user**: Hybrid scoring across four signals — TF-IDF content similarity, item-based collaborative filtering, price compatibility, and developer preference
- **Feedback loop**: Both flows update preferences in real time based on likes/dislikes within the session
- **Explainable results**: Every recommendation includes a human-readable reason (e.g. "Matches genres: action, RPG" or "From preferred developer: Valve")

---

## Architecture

```
main.py                    ← Entry point — run with --user new or --user existing
├── new_user.py            ← Cold-start pipeline
│   ├── Genre selection (user input)
│   ├── Jaccard similarity (genre match)
│   ├── Wilson score (community confidence)
│   └── Feedback loop (like/dislike updates genre preferences)
│
└── existing_user.py       ← Hybrid pipeline
        ├── DataLoader          (CSV loading, price normalization, feature concat)
    ├── UserAnalyzer        (playtime-weighted genre/tag/developer profiling)
    └── GameRecommender     (TF-IDF + item-based CF + price + dev scoring)
        └── Feedback loop   (virtual playtime injection on like/dislike)
```

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3 |
| Data | pandas, numpy |
| ML / Similarity | scikit-learn (TF-IDF, cosine similarity), scipy (Wilson score, sparse matrices) |
| Stats | scipy.stats (norm), Wilson lower bound confidence intervals |
| UX | tqdm (progress bars), nbformat (notebook execution) |

---

## Dataset

The system uses `steam_dataset_2000users.csv` — a filtered and merged dataset built from Steam user interaction logs and game metadata. It contains **2,000 users**, **1,951 games**, and **19,472 user–game interaction records**.

| Column | Type | Description |
|---|---|---|
| `user_id` | Integer | Unique user identifier |
| `game_id` | Integer | Unique game identifier |
| `game_title` | String | Official game title (e.g. "Portal 2") |
| `hours` | Integer | Total hours the user played the game |
| `normalized_binary` | Binary | Implicit like label — 1 if the user "liked" the game based on playtime, 0 otherwise |
| `developer` | String | Studio that developed the game (e.g. "Valve") |
| `genres` | Categorical | Pipe-separated genre list (e.g. "Action\|Adventure") |
| `tags` | Categorical | Pipe-separated user-assigned tags (e.g. "Co-op\|Singleplayer\|Puzzle") |
| `specs` | Categorical | Supported features (e.g. "Steam Achievements\|Full controller support") |
| `metascore` | Integer | Metacritic score, range 0–100 |
| `price` | Float | Game price in USD; free-to-play titles normalized to 0.0 |

**Key statistics from exploratory analysis:**
- Average playtime per game: **66.1 hours** (90th percentile: 133 hours) — a small cohort of "power users" drives most playtime
- Only **3.6%** of titles are free or free-to-play; most paid games fall in the **$10–$20** range (25th–75th percentile: $9.99–$19.99)
- Implicit feedback skews heavily negative — only **26.6%** of interactions carry a positive label, reflecting that most users try many games but genuinely enjoy few
- Top genres: Action (870 games), Indie (819), Adventure (607)
- Top developers by total playtime: Valve (86,488 hrs), Paradox Development Studio (55,140 hrs), Bethesda Game Studios (54,451 hrs)

>  Place `steam_dataset_2000users.csv` in the root directory before running.

---

## Data Preprocessing

Before any modeling, the following preprocessing steps were applied:

**Price normalization** — all "free", "Free to Play", and similar string variants in the price column were replaced with `0.0` and the column was cast to `float`. This ensures the price scoring component operates on a clean numeric field.

**Feature concatenation** — the `genres` and `tags` columns were merged into a single `genres_tags` string per game. This combined field is what the TF-IDF vectorizer operates on in the existing-user pipeline, capturing both high-level genre signals and granular community-assigned tags in one representation.

**Missing value check** — all key attributes (playtime, feedback label, metadata fields) were verified to have no missing values before model construction.

**User–item matrix** — a sparse matrix indexed by `user_id` (rows) and `game_id` (columns), valued by `hours`, was built for collaborative filtering. Using a sparse representation (via `scipy.sparse.csr_matrix`) keeps memory usage manageable given the high sparsity of the interaction space.

**Similarity matrices** — two similarity matrices were pre-computed at load time:
- A **dense TF-IDF cosine-similarity matrix** over `genres_tags` for content-based scoring
- A **sparse item–item cosine-similarity matrix** over game columns of the user–item matrix for collaborative scoring

**Cold-start bypass** — for new users with no history, the preprocessing pipeline is skipped entirely. Instead, raw `genres` sets are used directly for Jaccard similarity computations, and global `normalized_binary` interaction counts feed the Wilson score estimator.

---

## How It Works

### New User Flow
1. Displays all available genres and prompts the user to select preferences
2. Scores all games using a weighted hybrid: Jaccard genre similarity (10%) + Wilson score community confidence (90%)
3. Returns top 5 main recommendations + 5 bonus picks based on global popularity
4. Accepts feedback — liked games expand genre preferences, disliked games narrow them
5. Loops until the user exits

### Existing User Flow
1. Loads the user's full interaction history and builds a playtime-weighted profile covering genres, tags, and developers
2. Computes four independent score vectors:
   - **Content score** — TF-IDF cosine similarity between the user's played games and all candidates
   - **Collaborative score** — item-based CF using sparse cosine similarity on the user–item matrix
   - **Price score** — inverse absolute deviation from the user's historical average spend
   - **Developer score** — 4× multiplier for games from the user's top-3 preferred developers
3. Combines scores: `final_score = 0.4·content + 0.3·collab + 0.2·price + 0.1·developer`
4. Feedback adjusts the session profile by injecting virtual playtime hours (+20 for likes, −20 for dislikes), shifting future recommendations without permanently altering the base profile

### Evaluation
The model was evaluated on 1,000 held-out users not seen during training:

| Metric | Score |
|---|---|
| Precision@5 | 0.0112 |
| Recall@5 | 0.0336 |
| F1@5 | 0.0168 |
| Hit Rate@5 | 0.052 |

These values are low but expected — each user averages only 2.25 positively labeled games, making it statistically unlikely to land one in a top-5 list. Qualitatively, the model correctly surfaces action-heavy picks for action players and RPG picks for RPG fans, confirming the scoring signals are meaningful despite the sparse label space.

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/steam-recommender.git
cd steam-recommender
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**

Place `steam_dataset_2000users.csv` in the root folder.

**4. Launch**
```bash
python main.py
```

# or specify directly:
python main.py --user new
python main.py --user existing

---

## Project Structure

```
steam-recommender/
├── main.py                 # Entry point
├── new_user.py             # Cold-start pipeline
├── existing_user.py        # Hybrid pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Authors

| Name | Role |
|---|---|
| Bashar Adas | Team Leader — data cleaning, evaluation, Jaccard/Wilson scoring, terminal interface |
| Taher El Taher | EDA, data description, collaborative filtering |
| Kinan Khaskieh | Content model, hybrid fusion logic, diversity experiments |

**Bashar Adas** — B.Sc. Computer Science, American University of Sharjah  
[LinkedIn](https://www.linkedin.com/in/bashar-adas-213196192/)
