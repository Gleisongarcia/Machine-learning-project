from tdc.single_pred import ADME

data = ADME(name="CYP2C19_Veith")
split = data.get_split()

print(split.keys())

for name, subset in split.items():
    print(name, subset.shape)


print("\n--- Target distribution by split ---")

for name, subset in split.items():
    print(f"\n{name.upper()} — {len(subset)} molecules")

    counts = subset["Y"].value_counts().sort_index()
    percentages = (
        subset["Y"].value_counts(normalize=True).sort_index() * 100
    )

    print("Counts:")
    print(counts)

    print("Percentages:")
    print(percentages.round(2))
