
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse import save_npz

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import ConfusionMatrixDisplay


# --------------------------------------------------
# 1. Load the dataset and official TDC splits
# --------------------------------------------------

from tdc.single_pred import ADME

data = ADME(name="CYP2C19_Veith")
split = data.get_split()

train = split["train"].copy()
valid = split["valid"].copy()
test = split["test"].copy()

print("--- Dataset overview ---")
for name, subset in [
    ("Train", train),
    ("Validation", valid),
    ("Test", test),
]:
    print(f"{name}: {len(subset)} molecules")


# --------------------------------------------------
# 2. Basic SMILES analysis
# --------------------------------------------------

output_dir = Path(__file__).resolve().parent.parent / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

train["smiles_length"] = train["Drug"].str.len()

print("\n--- SMILES length statistics (train) ---")
print(train["smiles_length"].describe())

plt.figure(figsize=(8, 5))

for label in sorted(train["Y"].unique()):
    lengths = train.loc[
        train["Y"] == label, "smiles_length"
    ]
    plt.hist(
        lengths,
        bins=40,
        alpha=0.5,
        label=f"Y = {label}",
    )

plt.xlabel("SMILES length (characters)")
plt.ylabel("Number of molecules")
plt.title("SMILES length distribution by class")
plt.legend()
plt.tight_layout()
plt.savefig(
    output_dir / "text_smiles_length_distribution.png",
    dpi=200,
)
plt.close()


# --------------------------------------------------
# 3. Convert SMILES into character n-gram features
# --------------------------------------------------

# Use character sequences of length 2, 3 and 4.
# Fit ONLY on training data to prevent information leakage.

vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    min_df=2,
    max_features=30000,
    binary=True,
    dtype=np.float32,
)

X_train = vectorizer.fit_transform(train["Drug"])
X_valid = vectorizer.transform(valid["Drug"])
X_test = vectorizer.transform(test["Drug"])

y_train = train["Y"].astype(int).to_numpy()
y_valid = valid["Y"].astype(int).to_numpy()
y_test = test["Y"].astype(int).to_numpy()

print("\n--- Text representation ---")
print("Training matrix:", X_train.shape)
print("Validation matrix:", X_valid.shape)
print("Test matrix:", X_test.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))

for name, matrix in [
    ("Train", X_train),
    ("Validation", X_valid),
    ("Test", X_test),
]:
    density = matrix.nnz / (
        matrix.shape[0] * matrix.shape[1]
    )
    print(f"{name} matrix density: {density:.4%}")

# Save sparse matrices without converting them to dense arrays.
# They can be reused without rebuilding the vocabulary.
save_npz(output_dir / "text_X_train.npz", X_train)
save_npz(output_dir / "text_X_valid.npz", X_valid)
save_npz(output_dir / "text_X_test.npz", X_test)


# --------------------------------------------------
# 4. Number of distinct n-grams per molecule
# --------------------------------------------------

active_ngrams = np.asarray(X_train.getnnz(axis=1)).ravel()

plt.figure(figsize=(8, 5))

for label in sorted(np.unique(y_train)):
    plt.hist(
        active_ngrams[y_train == label],
        bins=40,
        alpha=0.5,
        label=f"Y = {label}",
    )

plt.xlabel("Number of distinct character n-grams")
plt.ylabel("Number of molecules")
plt.title("Character n-grams per molecule")
plt.legend()
plt.tight_layout()
plt.savefig(
    output_dir / "text_active_ngrams_distribution.png",
    dpi=200,
)
plt.close()


# --------------------------------------------------
# 5. Most frequent character n-grams
# --------------------------------------------------

feature_names = vectorizer.get_feature_names_out()
frequencies = np.asarray(X_train.sum(axis=0)).ravel()

top_indices = np.argsort(frequencies)[-25:][::-1]

top_features = pd.DataFrame({
    "ngram": feature_names[top_indices],
    "molecule_count": frequencies[top_indices],
})

print("\n--- Most frequent character n-grams (train) ---")
print(top_features.to_string(index=False))

plt.figure(figsize=(11, 6))
plt.bar(
    top_features["ngram"],
    top_features["molecule_count"],
)
plt.xlabel("Character n-gram")
plt.ylabel("Number of molecules containing the pattern")
plt.title("25 most frequent character n-grams")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(
    output_dir / "text_frequent_ngrams.png",
    dpi=200,
)
plt.close()


# --------------------------------------------------
# 6. Two-dimensional projection with TruncatedSVD
# --------------------------------------------------

# Fit the projection on training data only.
# Then display a sample from all three partitions.

rng = np.random.default_rng(42)

sample_size = min(5000, len(train) + len(valid) + len(test))

all_matrices = [X_train, X_valid, X_test]
all_targets = [y_train, y_valid, y_test]
all_names = ["Train", "Validation", "Test"]

all_X = __import__("scipy").sparse.vstack(all_matrices)
all_y = np.concatenate(all_targets)
all_partitions = np.concatenate([
    np.repeat(name, len(y))
    for name, y in zip(all_names, all_targets)
])

sample_idx = rng.choice(
    all_X.shape[0],
    size=sample_size,
    replace=False,
)

X_sample = all_X[sample_idx]
y_sample = all_y[sample_idx]

svd = TruncatedSVD(
    n_components=2,
    random_state=42,
)

svd.fit(X_train)
projection = svd.transform(X_sample)

plt.figure(figsize=(8, 6))

for label in sorted(np.unique(y_sample)):
    mask = y_sample == label
    plt.scatter(
        projection[mask, 0],
        projection[mask, 1],
        s=12,
        alpha=0.5,
        label=f"Y = {label}",
    )

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("SMILES character n-gram projection (TruncatedSVD)")
plt.legend()
plt.tight_layout()
plt.savefig(
    output_dir / "text_fingerprint_projection.png",
    dpi=200,
)
plt.close()

print(
    "SVD explained variance ratio sum:",
    round(svd.explained_variance_ratio_.sum(), 4),
)


# --------------------------------------------------
# 7. Baseline classification models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for name, model in models.items():
    print(f"\n--- {name} ---")

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_valid, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_valid, predictions
        ),
        "precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_valid, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_valid, probabilities
        ),
    }

    results.append(metrics)

    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric}: {value:.3f}")

    ConfusionMatrixDisplay.from_predictions(
        y_valid,
        predictions,
        values_format="d",
    )
    plt.title(f"{name} — validation")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    plt.savefig(
        output_dir / f"text_{safe_name}_confusion_matrix.png",
        dpi=200,
    )
    plt.close()


# Save validation metrics for later comparison with Morgan.
results_df = pd.DataFrame(results)
results_df.to_csv(
    output_dir / "text_baseline_metrics.csv",
    index=False,
)

print("\n--- Validation comparison ---")
print(results_df.round(3).to_string(index=False))

print(f"\nFigures and results saved in: {output_dir.resolve()}")
