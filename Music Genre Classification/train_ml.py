import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Directories
FEATURE_DIR = Path("features")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    """Load features from CSVs and split into train/val/test sets."""
    train_df = pd.read_csv(FEATURE_DIR / "train_features.csv")
    val_df = pd.read_csv(FEATURE_DIR / "val_features.csv")
    test_df = pd.read_csv(FEATURE_DIR / "test_features.csv")

    X_train = train_df.drop(columns=["track_id", "genre"])
    y_train = train_df["genre"]
    X_val = val_df.drop(columns=["track_id", "genre"])
    y_val = val_df["genre"]
    X_test = test_df.drop(columns=["track_id", "genre"])
    y_test = test_df["genre"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_features(X_train, X_val, X_test):
    """Standardize features across train/val/test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def encode_labels(y_train, y_val, y_test):
    """Convert string labels into numeric IDs."""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_val_enc, y_test_enc, le


def get_models():
    """Define ML models to train."""
    return {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, use_label_encoder=False, eval_metric="mlogloss", random_state=42
        ),
    }


def train_and_evaluate(models, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder):
    """Train all models, evaluate, and collect results."""
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        # Decode labels for reports
        y_val_decoded = label_encoder.inverse_transform(y_val)
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_val_decoded = label_encoder.inverse_transform(y_pred_val)
        y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)

        val_acc = accuracy_score(y_val_decoded, y_pred_val_decoded)
        test_acc = accuracy_score(y_test_decoded, y_pred_test_decoded)

        print(f"{name} — Validation Accuracy: {val_acc:.4f}")
        print(f"{name} — Test Accuracy: {test_acc:.4f}")
        print(f"{name} — Training Time: {train_time:.2f}s")
        print("\nClassification Report (Test Set):\n")
        print(classification_report(y_test_decoded, y_pred_test_decoded, digits=4))

        results.append({
            "Model": name,
            "Validation Accuracy": val_acc,
            "Test Accuracy": test_acc,
            "Training Time (s)": train_time,
        })

        trained_models[name] = model

    return pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False), trained_models


def plot_results(results_df):
    """Visualize model comparison."""
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="Model", y="Test Accuracy", palette="viridis")
    plt.title("Model Performance on Test Set")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png")
    plt.show()


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    # Encode labels
    y_train_enc, y_val_enc, y_test_enc, le = encode_labels(y_train, y_val, y_test)

    models = get_models()
    results_df, trained_models = train_and_evaluate(models, X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, le)

    # Save comparison results
    results_csv = RESULTS_DIR / "ml_model_comparison.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Results saved to {results_csv}")

    print("\n=== Model Comparison ===")
    print(results_df)

    # Save best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    model_path = RESULTS_DIR / f"best_model_{best_model_name.replace(' ', '_')}.pkl"
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "label_encoder": le
    }, model_path)
    print(f"✅ Best model ({best_model_name}) saved to {model_path}")

    plot_results(results_df)


if __name__ == "__main__":
    main()
