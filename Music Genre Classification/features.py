import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings

SR = 22050
N_MFCC = 20
SPLIT_DIR = Path("splits")
FEATURE_DIR = Path("features")
FEATURE_DIR.mkdir(exist_ok=True, parents=True)

def extract_features(file_path, sr=SR):
    """Extract audio features for one track as a dict of aggregated statistics."""
    feats = {}
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        warnings.warn(f"⚠️ Could not read {file_path}: {e}")
        return None  # skip this file

    # Spectral features
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    feats["rms_mean"] = librosa.feature.rms(S=S).mean()
    feats["zcr_mean"] = librosa.feature.zero_crossing_rate(y).mean()
    feats["centroid_mean"] = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    feats["bandwidth_mean"] = librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()
    feats["rolloff_mean"] = librosa.feature.spectral_rolloff(S=S, sr=sr).mean()
    feats["flatness_mean"] = librosa.feature.spectral_flatness(S=S).mean()

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats["tempo"] = float(tempo)  


    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(mfccs[i].mean())
        feats[f"mfcc{i+1}_std"] = float(mfccs[i].std())

    # Chroma
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    for i in range(chroma.shape[0]):
        feats[f"chroma{i+1}_mean"] = chroma[i].mean()

    return feats


def process_split(split_csv, out_csv):
    """Extract features for one CSV split and save to output CSV."""
    df_split = pd.read_csv(split_csv)
    rows = []
    skipped = 0
    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Extracting {split_csv.stem}"):
        feats = extract_features(row["filepath"])
        if feats is None:
            skipped += 1
            continue
        feats["track_id"] = row["track_id"]
        feats["genre"] = row["genre"]
        rows.append(feats)

    df_feats = pd.DataFrame(rows)
    df_feats.to_csv(out_csv, index=False)
    print(f"✅ Saved {out_csv} with shape {df_feats.shape}. Skipped {skipped} files.")


def extract_all_features():
    """Extract features for train, val, and test splits."""
    for split in ["train", "val", "test"]:
        split_csv = SPLIT_DIR / f"{split}.csv"
        out_csv = FEATURE_DIR / f"{split}_features.csv"
        process_split(split_csv, out_csv)


if __name__ == "__main__":
    extract_all_features()
