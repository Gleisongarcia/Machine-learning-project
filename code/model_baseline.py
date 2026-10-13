
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tdc.single_pred import ADME

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


# 1. Load the original dataset split
data = ADME(name="CYP2C19_Veith")
split = data.get_split()


# 2. Convert SMILES into Morgan fingerprints
def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits
    )

    array = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, array)

    return array


def prepare_split(subset):
    X = np.stack([
        smiles_to_morgan(smiles)
        for smiles in subset["Drug"]
    ])
    y = subset["Y"].to_numpy()

    return X, y


X_train, y_train = prepare_split(split["train"])
X_valid, y_valid = prepare_split(split["valid"])
X_test, y_test = prepare_split(split["test"])


# 3. Define baseline models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
}


# 4. Train on train and evaluate on validation
for name, model in models.items():
    print(f"\n--- {name} ---")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_score = model.predict_proba(X_valid)[:, 1]

    print(f"Accuracy:          {accuracy_score(y_valid, y_pred):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_valid, y_pred):.3f}")
    print(f"Precision:         {precision_score(y_valid, y_pred, zero_division=0):.3f}")
    print(f"Recall:            {recall_score(y_valid, y_pred, zero_division=0):.3f}")
    print(f"F1-score:          {f1_score(y_valid, y_pred, zero_division=0):.3f}")
    print(f"ROC-AUC:           {roc_auc_score(y_valid, y_score):.3f}")
