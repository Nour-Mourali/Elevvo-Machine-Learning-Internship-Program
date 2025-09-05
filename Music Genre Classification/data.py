import kagglehub
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedGroupKFold

def download_and_split(random_state=42):
    # Download dataset
    path = Path(kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification"))
    genres_root = path / "Data" / "genres_original"

    # Index files
    rows = []
    for genre_dir in genres_root.iterdir():
        if genre_dir.is_dir():
            genre = genre_dir.name
            for wav in genre_dir.glob("*.wav"):
                rows.append({
                    "filepath": str(wav),
                    "genre": genre,
                    "track_id": wav.stem
                })
    df = pd.DataFrame(rows)

    # Prepare folders
    SPLIT_DIR = Path("splits"); SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR = Path("metadata"); META_DIR.mkdir(parents=True, exist_ok=True)

    # StratifiedGroupKFold split
    y = df["genre"].values
    groups = df["track_id"].values
    idx_all = np.arange(len(df))

    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf_test.split(idx_all, y=y, groups=groups))

    remain_mask = np.zeros(len(df), dtype=bool)
    remain_mask[train_idx] = True
    df_remain = df[remain_mask].reset_index(drop=True)

    y_rem = df_remain["genre"].values
    groups_rem = df_remain["track_id"].values
    idx_rem = np.arange(len(df_remain))

    sgkf_val = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state+1)
    train2_idx, val2_idx = next(sgkf_val.split(idx_rem, y=y_rem, groups=groups_rem))

    abs_remain_positions = np.where(remain_mask)[0]
    train_abs = abs_remain_positions[train2_idx]
    val_abs = abs_remain_positions[val2_idx]
    test_abs = test_idx

    df.iloc[train_abs].to_csv(SPLIT_DIR / "train.csv", index=False)
    df.iloc[val_abs].to_csv(SPLIT_DIR / "val.csv", index=False)
    df.iloc[test_abs].to_csv(SPLIT_DIR / "test.csv", index=False)

    # Save class mapping
    classes = sorted(df["genre"].unique())
    with open(META_DIR / "class_mapping.json", "w") as f:
        json.dump({"classes": classes, "class_to_id": {c: i for i, c in enumerate(classes)}}, f, indent=2)

    print("Splits created in 'splits/' and metadata saved.")

if __name__ == "__main__":
    download_and_split()
