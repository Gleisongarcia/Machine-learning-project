from tdc.single_pred import ADME

# 1. Load the dataset independently
data = ADME(name="CYP2C19_Veith")
df = data.get_data()

# 2. Explore the dataset
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())


import matplotlib.pyplot as plt

# 3. Distribution of the target variable
print("\nTarget distribution:")
print(df["Y"].value_counts().sort_index())

print("\nTarget distribution (%):")
print(
    (df["Y"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

# 4. Plot the target distribution
df["Y"].value_counts().sort_index().plot(
    kind="bar",
    rot=0
)

plt.title("Distribution of CYP2C19 inhibition classes")
plt.xlabel("Target class (Y)")
plt.ylabel("Number of molecules")
plt.tight_layout()

# Save the figure instead of displaying it
plt.savefig("target_distribution.png", dpi=150)
print("\nGraph saved as target_distribution.png")

plt.close()


# 5. Explore SMILES strings
df["smiles_length"] = df["Drug"].astype(str).str.len()

print("\nSMILES length statistics:")
print(df["smiles_length"].describe())

print("\nNumber of unique SMILES:", df["Drug"].nunique())
print("Repeated SMILES:", df["Drug"].duplicated().sum())

# 6. Plot SMILES length distribution
plt.figure(figsize=(8, 5))
plt.hist(df["smiles_length"], bins=40, edgecolor="black")

plt.title("Distribution of SMILES lengths")
plt.xlabel("SMILES length (characters)")
plt.ylabel("Number of molecules")
plt.tight_layout()

plt.savefig("smiles_length_distribution.png", dpi=150)
print("\nGraph saved as smiles_length_distribution.png")

plt.close()
