import sys
import joblib
import numpy as np
from pathlib import Path
import warnings
import tempfile
from features import extract_features  # reuse existing function

# For MP3 to WAV conversion
from pydub import AudioSegment  

# Paths
RESULTS_DIR = Path("results")
MODEL_PATH = max(RESULTS_DIR.glob("best_model_*.pkl"), key=lambda p: p.stat().st_mtime)  # latest saved model


def convert_mp3_to_wav(mp3_path):
    """Convert MP3 to WAV in a temporary file and return its path."""
    sound = AudioSegment.from_mp3(mp3_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sound.export(tmp.name, format="wav")
    return tmp.name


def predict_genre(file_path):
    """Predict genre of a given audio file (wav or mp3)."""
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    scaler = saved["scaler"]
    label_encoder = saved["label_encoder"]

    # Convert mp3 to wav if needed
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".mp3":
        file_path = Path(convert_mp3_to_wav(file_path))

    feats = extract_features(file_path)
    if feats is None:
        warnings.warn("Could not extract features from this file.")
        return None

    X = np.array(list(feats.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return label_encoder.inverse_transform(y_pred)[0]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Usage: python predict.py <audio_file.wav|audio_file.mp3>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"⚠️ File not found: {file_path}")
        sys.exit(1)

    genre = predict_genre(file_path)
    print(f"\n🎵 Predicted Genre: {genre}")
