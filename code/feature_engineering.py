
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from tdc.single_pred import ADME
from sklearn.decomposition import TruncatedSVD


# 1. Load the dataset
data = ADME(name="CYP2C19_Veith")
split = data.get_split()

train = split["train"].copy()
valid = split["valid"].copy()
test = split["test"].copy()


# 2. Convert SMILES into Morgan fingerprints
def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits
    )

    array = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return array


# 3. Transform each split independently
fingerprints_by_split = {}
targets_by_split = {}

for name, subset in [
    ("train", train),
    ("valid", valid),
    ("test", test)
]:
    fingerprints = [
        smiles_to_morgan(smiles)
        for smiles in subset["Drug"]
    ]

    invalid_count = sum(fp is None for fp in fingerprints)

    print(f"\n{name.upper()}")
    print("Number of molecules:", len(subset))
    print("Invalid SMILES:", invalid_count)

    valid_indices = [
        i for i, fp in enumerate(fingerprints)
        if fp is not None
    ]

    X = pd.DataFrame(
        [fingerprints[i] for i in valid_indices],
        index=valid_indices
    )

    y = subset.iloc[valid_indices]["Y"].reset_index(drop=True)

    fingerprints_by_split[name] = X
    targets_by_split[name] = y

    print("Fingerprint matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    if len(X) > 0:
        print("First fingerprint:", X.iloc[0, :10].tolist())
        print("First target:", y.iloc[0])


# 4. Create output folder for figures
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)


# 5. Combine fingerprints for exploratory visualization
X_all = pd.concat(
    fingerprints_by_split.values(),
    ignore_index=True
)

y_all = pd.concat(
    targets_by_split.values(),
    ignore_index=True
)

print("\n--- Fingerprint analysis ---")
print("Combined fingerprint matrix:", X_all.shape)


# 6. Distribution of active bits per molecule
active_bits = X_all.sum(axis=1)

plt.figure(figsize=(8, 5))
plt.hist(active_bits, bins=40, edgecolor="black")
plt.xlabel("Number of active bits")
plt.ylabel("Number of molecules")
plt.title("Distribution of active Morgan bits")
plt.tight_layout()
plt.savefig(output_dir / "active_bits_distribution.png", dpi=200)
plt.show()


# 7. Most frequent bits
bit_frequencies = X_all.sum(axis=0).to_numpy()
top_bits = np.argsort(bit_frequencies)[-20:][::-1]

plt.figure(figsize=(10, 5))
plt.bar(
    [str(bit) for bit in top_bits],
    bit_frequencies[top_bits]
)
plt.xlabel("Morgan bit ID")
plt.ylabel("Number of molecules containing the bit")
plt.title("20 most frequent Morgan bits")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(output_dir / "frequent_bits.png", dpi=200)
plt.show()


# 8. Two-dimensional projection using TruncatedSVD
rng = np.random.default_rng(42)
sample_size = min(5000, len(X_all))
sample_idx = rng.choice(len(X_all), size=sample_size, replace=False)

X_sample = X_all.iloc[sample_idx].to_numpy()
y_sample = y_all.iloc[sample_idx].to_numpy()

projection = TruncatedSVD(
    n_components=2,
    random_state=42
).fit_transform(X_sample)

plt.figure(figsize=(8, 6))

for label in np.unique(y_sample):
    mask = y_sample == label
    plt.scatter(
        projection[mask, 0],
        projection[mask, 1],
        s=12,
        alpha=0.5,
        label=f"Y = {label}"
    )

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Morgan fingerprint projection (TruncatedSVD)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "fingerprint_projection.png", dpi=200)
plt.show()



# 9. Visualize molecular fragments associated with Morgan bits
first_smiles = train["Drug"].iloc[0]
mol = Chem.MolFromSmiles(first_smiles)

if mol is not None:
    bit_info = {}

    AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048,
        bitInfo=bit_info
    )

    # Select up to six active bits
    active_bit_ids = list(bit_info.keys())[:6]

    for bit_id in active_bit_ids:
        atom_id, radius = bit_info[bit_id][0]

        # Highlight the atoms and bonds corresponding to this bit
        env = Chem.FindAtomEnvironmentOfRadiusN(
            mol, radius, atom_id
        )

        atom_ids = {atom_id}
        for bond_id in env:
            bond = mol.GetBondWithIdx(bond_id)
            atom_ids.add(bond.GetBeginAtomIdx())
            atom_ids.add(bond.GetEndAtomIdx())

        Draw.MolToFile(
            mol,
            str(output_dir / f"morgan_bit_{bit_id}.png"),
            size=(600, 400),
            highlightAtoms=list(atom_ids),
            highlightBonds=list(env),
            legend=f"Morgan bit {bit_id}, radius {radius}"
        )

print(f"\nFigures saved in: {output_dir.resolve()}")

