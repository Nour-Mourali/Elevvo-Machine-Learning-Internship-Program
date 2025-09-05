import kagglehub
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import os

# 1️ Download and load dataset
path = kagglehub.dataset_download("prajitdatta/movielens-100k-dataset")
print("Dataset folder:", path)

data_dir = os.path.join(path, "ml-100k")
print("Data directory:", data_dir)
print("Files inside:", os.listdir(data_dir))

# Ratings
ratings_path = os.path.join(data_dir, "u.data")
ratings = pd.read_csv(ratings_path, sep="\t", names=["user_id","item_id","rating","timestamp"])
print("\nRatings sample:")
print(ratings.head())

# Movies
movie_cols = ["movie_id","title","release_date","video_release","imdb_url"]
movies_path = os.path.join(data_dir, "u.item")
movies = pd.read_csv(movies_path, sep="|", header=None, encoding="latin-1")
movies = movies.iloc[:, :5]
movies.columns = movie_cols
print("\nMovies sample:")
print(movies.head())

# 2️ Shuffle and split into train/test
ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)
test_indices = ratings.groupby('user_id').tail(1).index
test_df = ratings.loc[test_indices]
train_df = ratings.drop(test_indices)

print("Train size:", len(train_df))
print("Test size:", len(test_df))

# 3️ Map IDs to indices using train only
user_ids = sorted(train_df['user_id'].unique())
item_ids = sorted(train_df['item_id'].unique())
user_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_map = {iid: idx for idx, iid in enumerate(item_ids)}
rev_item_map = {idx: iid for iid, idx in item_map.items()}

print(f"Total users: {len(user_map)}, Total movies: {len(item_map)}")

# 4️ Build user-item matrix (CSR + dense)
rows = train_df['user_id'].map(user_map)
cols = train_df['item_id'].map(item_map)
data = train_df['rating']

R = csr_matrix((data, (rows, cols)), shape=(len(user_map), len(item_map)))
R_dense = R.toarray()

print("User-Item matrix shape:", R.shape)
print("Number of ratings (non-zero):", R.nnz)


# Model 1: User-CF

print("\n Model 1: User-CF \n")

# 5️ Compute user similarity
user_sim = cosine_similarity(R_dense)
print("User similarity matrix shape:", user_sim.shape)

# User-CF functions
def predict_ratings(user_idx, R_dense, user_sim, top_k=5):
    sim_scores = user_sim[user_idx]
    similar_users = np.argsort(sim_scores)[::-1]
    similar_users = similar_users[similar_users != user_idx]
    top_users = similar_users[:top_k]

    ratings_pred = np.zeros(R_dense.shape[1])
    sim_sum = np.sum(user_sim[user_idx, top_users])
    if sim_sum == 0:
        sim_sum = 1e-8

    for i in range(R_dense.shape[1]):
        ratings_pred[i] = np.dot(user_sim[user_idx, top_users], R_dense[top_users, i]) / sim_sum

    return ratings_pred

def recommend_movies(user_idx, R_dense, user_sim, top_k=5, n_rec=5):
    ratings_pred = predict_ratings(user_idx, R_dense, user_sim, top_k=top_k)
    rated = R_dense[user_idx] > 0
    ratings_pred[rated] = -1
    rec_indices = np.argsort(ratings_pred)[::-1][:n_rec]
    return rec_indices

# Example recommendation
rec_movies = recommend_movies(0, R_dense, user_sim, top_k=5, n_rec=5)
print("\nRecommended movies for user 0 (User-CF):")
for idx in rec_movies:
    movie_id = rev_item_map[idx]
    title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print("-", title)

# Evaluate Precision@5
def precision_at_k(recommended_indices, true_item_indices, k=5):
    recommended_topk = recommended_indices[:k]
    hits = sum([1 for item in recommended_topk if item in true_item_indices])
    return hits / k

precisions = []
for uid in test_df['user_id'].unique():
    if uid not in user_map:
        continue
    user_idx = user_map[uid]
    true_items = test_df[test_df['user_id'] == uid]['item_id']
    true_indices = [item_map[iid] for iid in true_items if iid in item_map]
    if len(true_indices) == 0:
        continue
    rec_indices = recommend_movies(user_idx, R_dense, user_sim, top_k=5, n_rec=5)
    precisions.append(precision_at_k(rec_indices, true_indices, k=5))

avg_precision = sum(precisions) / len(precisions)
print(f"User-CF Precision@5: {avg_precision:.4f}")

# Model 2: Item-CF

print("\n Model 2: Item-CF \n")

item_sim = cosine_similarity(R_dense.T)
print("Item similarity matrix shape:", item_sim.shape)

def predict_ratings_itemcf(user_idx, R_dense, item_sim, top_k=5):
    ratings_pred = np.zeros(R_dense.shape[1])
    for i in range(R_dense.shape[1]):
        if R_dense[user_idx, i] > 0:
            ratings_pred[i] = -1
            continue
        sim_scores = item_sim[i]
        rated_items = np.where(R_dense[user_idx] > 0)[0]
        top_items = rated_items[np.argsort(sim_scores[rated_items])[::-1][:top_k]]
        sim_sum = np.sum(sim_scores[top_items])
        if sim_sum == 0:
            sim_sum = 1e-8
        ratings_pred[i] = np.dot(sim_scores[top_items], R_dense[user_idx, top_items]) / sim_sum
    return ratings_pred

def recommend_movies_itemcf(user_idx, R_dense, item_sim, top_k=5, n_rec=5):
    ratings_pred = predict_ratings_itemcf(user_idx, R_dense, item_sim, top_k=top_k)
    rec_indices = np.argsort(ratings_pred)[::-1][:n_rec]
    return rec_indices

rec_movies_item = recommend_movies_itemcf(0, R_dense, item_sim, top_k=5, n_rec=5)
print("Recommended movies for user 0 (Item-CF):")
for idx in rec_movies_item:
    movie_id = rev_item_map[idx]
    title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print("-", title)

precisions_item = []
for uid in test_df['user_id'].unique():
    if uid not in user_map:
        continue
    user_idx = user_map[uid]
    true_items = test_df[test_df['user_id'] == uid]['item_id']
    true_indices = [item_map[iid] for iid in true_items if iid in item_map]
    if len(true_indices) == 0:
        continue
    rec_indices = recommend_movies_itemcf(user_idx, R_dense, item_sim, top_k=5, n_rec=5)
    precisions_item.append(precision_at_k(rec_indices, true_indices, k=5))

avg_precision_item = sum(precisions_item) / len(precisions_item)
print(f"Item-CF Precision@5: {avg_precision_item:.4f}")

# Model 3: Hybrid CF (User + Item)

print("\n Model 3: Hybrid CF \n")

# Hybrid functions
def predict_ratings_hybrid(user_idx, R_dense, user_sim, item_sim, top_k_user=5, top_k_item=5, alpha=0.5):
    user_pred = predict_ratings(user_idx, R_dense, user_sim, top_k=top_k_user)
    item_pred = predict_ratings_itemcf(user_idx, R_dense, item_sim, top_k=top_k_item)
    hybrid_pred = alpha * user_pred + (1 - alpha) * item_pred
    hybrid_pred[R_dense[user_idx] > 0] = -1
    return hybrid_pred

def recommend_movies_hybrid(user_idx, R_dense, user_sim, item_sim, top_k_user=5, top_k_item=5, n_rec=5, alpha=0.5):
    hybrid_pred = predict_ratings_hybrid(user_idx, R_dense, user_sim, item_sim, top_k_user, top_k_item, alpha)
    rec_indices = np.argsort(hybrid_pred)[::-1][:n_rec]
    return rec_indices

rec_movies_hybrid = recommend_movies_hybrid(0, R_dense, user_sim, item_sim, top_k_user=5, top_k_item=5, n_rec=5, alpha=0.5)
print("Recommended movies for user 0 (Hybrid CF):")
for idx in rec_movies_hybrid:
    movie_id = rev_item_map[idx]
    title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print("-", title)

precisions_hybrid = []
for uid in test_df['user_id'].unique():
    if uid not in user_map:
        continue
    user_idx = user_map[uid]
    true_items = test_df[test_df['user_id'] == uid]['item_id']
    true_indices = [item_map[iid] for iid in true_items if iid in item_map]
    if len(true_indices) == 0:
        continue
    rec_indices = recommend_movies_hybrid(user_idx, R_dense, user_sim, item_sim, top_k_user=5, top_k_item=5, n_rec=5, alpha=0.5)
    precisions_hybrid.append(precision_at_k(rec_indices, true_indices, k=5))

avg_precision_hybrid = sum(precisions_hybrid) / len(precisions_hybrid)
print(f"Hybrid CF Precision@5: {avg_precision_hybrid:.4f}")


# Model 4: Matrix Factorization (SVD)

print("\n Model 4: Matrix Factorization (SVD) \n")

n_factors = 20
svd = TruncatedSVD(n_components=n_factors, random_state=42)
user_factors = svd.fit_transform(R_dense)
item_factors = svd.components_.T
R_pred_svd = np.dot(user_factors, item_factors.T)

def recommend_movies_svd(user_idx, R_pred_svd, R_dense, n_rec=5):
    pred_ratings = R_pred_svd[user_idx].copy()
    pred_ratings[R_dense[user_idx] > 0] = -1
    rec_indices = np.argsort(pred_ratings)[::-1][:n_rec]
    return rec_indices

rec_movies_svd = recommend_movies_svd(0, R_pred_svd, R_dense, n_rec=5)
print("Recommended movies for user 0 (SVD):")
for idx in rec_movies_svd:
    movie_id = rev_item_map[idx]
    title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print("-", title)

precisions_svd = []
for uid in test_df['user_id'].unique():
    if uid not in user_map:
        continue
    user_idx = user_map[uid]
    true_items = test_df[test_df['user_id'] == uid]['item_id']
    true_indices = [item_map[iid] for iid in true_items if iid in item_map]
    if len(true_indices) == 0:
        continue
    rec_indices = recommend_movies_svd(user_idx, R_pred_svd, R_dense, n_rec=5)
    precisions_svd.append(precision_at_k(rec_indices, true_indices, k=5))

avg_precision_svd = sum(precisions_svd) / len(precisions_svd)
print(f"SVD CF Precision@5: {avg_precision_svd:.4f}")

# Model 5: Hybrid CF + SVD
print("\n Model 5: Hybrid CF + SVD \n")

def predict_ratings_hybrid_svd(user_idx, R_dense, user_sim, item_sim, R_pred_svd, top_k_user=5, top_k_item=5, alpha=0.4, beta=0.3):
    user_pred = predict_ratings(user_idx, R_dense, user_sim, top_k=top_k_user)
    item_pred = predict_ratings_itemcf(user_idx, R_dense, item_sim, top_k=top_k_item)
    svd_pred = R_pred_svd[user_idx]
    hybrid_pred = alpha * user_pred + beta * item_pred + (1 - alpha - beta) * svd_pred
    hybrid_pred[R_dense[user_idx] > 0] = -1
    return hybrid_pred

def recommend_movies_hybrid_svd(user_idx, R_dense, user_sim, item_sim, R_pred_svd, top_k_user=5, top_k_item=5, n_rec=5, alpha=0.4, beta=0.3):
    hybrid_pred = predict_ratings_hybrid_svd(user_idx, R_dense, user_sim, item_sim, R_pred_svd, top_k_user, top_k_item, alpha, beta)
    rec_indices = np.argsort(hybrid_pred)[::-1][:n_rec]
    return rec_indices

rec_movies_hybrid_svd = recommend_movies_hybrid_svd(0, R_dense, user_sim, item_sim, R_pred_svd, top_k_user=5, top_k_item=5, n_rec=5)
print("Recommended movies for user 0 (Hybrid CF+SVD):")
for idx in rec_movies_hybrid_svd:
    movie_id = rev_item_map[idx]
    title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print("-", title)

precisions_hybrid_svd = []
for uid in test_df['user_id'].unique():
    if uid not in user_map:
        continue
    user_idx = user_map[uid]
    true_items = test_df[test_df['user_id'] == uid]['item_id']
    true_indices = [item_map[iid] for iid in true_items if iid in item_map]
    if len(true_indices) == 0:
        continue
    rec_indices = recommend_movies_hybrid_svd(user_idx, R_dense, user_sim, item_sim, R_pred_svd)
    precisions_hybrid_svd.append(precision_at_k(rec_indices, true_indices, k=5))

avg_precision_hybrid_svd = sum(precisions_hybrid_svd) / len(precisions_hybrid_svd)

# Summary of all models
print("\n Summary of Precision@5\n")
print(f"User-CF Precision@5: {avg_precision:.4f}")
print(f"Item-CF Precision@5: {avg_precision_item:.4f}")
print(f"Hybrid CF Precision@5: {avg_precision_hybrid:.4f}")
print(f"SVD CF Precision@5: {avg_precision_svd:.4f}")
print(f"Hybrid CF+SVD Precision@5: {avg_precision_hybrid_svd:.4f}")
