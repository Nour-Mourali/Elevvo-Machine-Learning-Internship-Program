import kagglehub
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import os

# 1. Download and load dataset
path = kagglehub.dataset_download("prajitdatta/movielens-100k-dataset")
print("Dataset folder:", path)

data_dir = os.path.join(path, "ml-100k")
ratings_path = os.path.join(data_dir, "u.data")
movies_path = os.path.join(data_dir, "u.item")

# Load ratings
ratings = pd.read_csv(ratings_path, sep="\t", names=["user_id","item_id","rating","timestamp"])

# Load movies
movie_cols = ["movie_id","title","release_date","video_release","imdb_url"]
movies = pd.read_csv(movies_path, sep="|", header=None, encoding="latin-1")
movies = movies.iloc[:, :5]
movies.columns = movie_cols

# 2. Train-test split
ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)

# Leave-one-out per user
test_indices = ratings.groupby('user_id').tail(1).index
test_df = ratings.loc[test_indices]
train_df = ratings.drop(test_indices)

# Map raw IDs to indices
user_ids = sorted(train_df['user_id'].unique())
item_ids = sorted(train_df['item_id'].unique())
user_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_map = {iid: idx for idx, iid in enumerate(item_ids)}
rev_item_map = {idx: iid for iid, idx in item_map.items()}

# 3. Build user-item matrix
rows = train_df['user_id'].map(user_map)
cols = train_df['item_id'].map(item_map)
data = train_df['rating']

R = csr_matrix((data, (rows, cols)), shape=(len(user_map), len(item_map)))
R_dense = R.toarray()

# Compute movie popularity 
movie_popularity = train_df.groupby('item_id').size()  # count of ratings per movie
popularity_scores = np.array([movie_popularity.get(iid, 0) for iid in item_ids], dtype=float)
popularity_scores /= popularity_scores.max()  # normalize between 0 and 1

# 4. Recommendation + evaluation functions
def recommend_movies_svd(user_idx, R_pred_svd, R_dense, popularity_scores, alpha=0.6, n_rec=5):
    """
    Recommend top-N movies using SVD prediction + popularity weighting.
    alpha: weight for SVD prediction (0.0-1.0)
    """
    pred_ratings = R_pred_svd[user_idx].copy()
    final_scores = alpha * pred_ratings + (1 - alpha) * popularity_scores
    final_scores[R_dense[user_idx] > 0] = -1  # mask already rated
    rec_indices = np.argsort(final_scores)[::-1][:n_rec]
    return rec_indices

def precision_at_k(recommended_indices, true_item_indices, k=5):
    recommended_topk = recommended_indices[:k]
    hits = sum([1 for item in recommended_topk if item in true_item_indices])
    return hits / k

# 5. Matrix Factorization (SVD)
n_factors = 20
svd = TruncatedSVD(n_components=n_factors, random_state=42)
user_factors = svd.fit_transform(R_dense)
item_factors = svd.components_.T
R_pred_svd = np.dot(user_factors, item_factors.T)

# Example: recommend movies for user 0
rec_movies = recommend_movies_svd(0, R_pred_svd, R_dense, popularity_scores, alpha=0.6, n_rec=5)
print("\nRecommended movies for user 0:")
for idx in rec_movies:
    movie_id = rev_item_map[idx]
    title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print("-", title)

# 6. Evaluate with Precision@5
precisions = []
for uid in test_df['user_id'].unique():
    if uid not in user_map:
        continue
    user_idx = user_map[uid]
    true_items = test_df[test_df['user_id'] == uid]['item_id']
    true_indices = [item_map[iid] for iid in true_items if iid in item_map]
    if len(true_indices) == 0:
        continue
    rec_indices = recommend_movies_svd(user_idx, R_pred_svd, R_dense, popularity_scores, alpha=0.6, n_rec=5)
    precisions.append(precision_at_k(rec_indices, true_indices, k=5))

avg_precision_svd = sum(precisions) / len(precisions)
print(f"\nOptimized SVD CF + Popularity Precision@5: {avg_precision_svd:.4f}")
