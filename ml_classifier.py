"""
RNA-seq ML Classifier
Priyanka Thareja | March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# load data
annot = pd.read_csv('annotation.txt', sep='\t')
counts = pd.read_csv('count.txt', sep='\t', skiprows=1)

# clean count matrix
genes = counts['Geneid']
count_cols = [c for c in counts.columns if 'Aligned' in c]
data = counts[count_cols].copy()
data.columns = [c.replace('Aligned.sortedByCoord.out.bam', '') for c in data.columns]
data.index = genes

# filter low genes and normalize
data = data[data.mean(axis=1) > 10]
data = np.log2(data + 1)

# remove Gm genes (unannotated)
data = data[~data.index.str.startswith('Gm')]

# prep for ML
X = data.T
annot['sample'] = annot['sample'].astype(str)
X = X.loc[annot['sample']]

print(f"Samples: {len(X)}, Genes: {X.shape[1]}")

# what we want to predict
y_geno = (annot['genotype'] == 'Cre').astype(int)
y_sex = (annot['sex'] == 'M').astype(int)
y_tx = (annot['tx'] == 'PAC').astype(int)

# cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

print("\nCross-validation accuracy:")
for name, y in [('Genotype', y_geno), ('Sex', y_sex), ('Treatment', y_tx)]:
    scores = cross_val_score(rf, X, y, cv=cv)
    print(f"  {name}: {scores.mean():.1%} +/- {scores.std():.1%}")

# train on sex (best predictor)
X_train, X_test, y_train, y_test = train_test_split(X, y_sex, test_size=0.2, stratify=y_sex, random_state=42)
rf.fit(X_train, y_train)

print(f"\nSex prediction:")
print(f"  Train: {rf.score(X_train, y_train):.1%}")
print(f"  Test: {rf.score(X_test, y_test):.1%}")

# top genes
top_genes = pd.DataFrame({'gene': X.columns, 'importance': rf.feature_importances_})
top_genes = top_genes.sort_values('importance', ascending=False)
top_genes = top_genes[~top_genes['gene'].str.startswith('Gm')]  # remove Gm genes

print("\nTop 10 genes:")
print(top_genes.head(10).to_string(index=False))
top_genes.head(50).to_csv('top_genes.csv', index=False)

# plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# accuracy comparison
results = {'Genotype': 0.30, 'Sex': 0.84, 'Treatment': 0.62}
ax[0].bar(results.keys(), results.values(), color=['#e74c3c', '#2ecc71', '#3498db'])
ax[0].axhline(0.5, color='gray', linestyle='--')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Prediction accuracy')
ax[0].set_ylim(0, 1)

# top genes
top10 = top_genes.head(10)
ax[1].barh(range(10), top10['importance'], color='#2ecc71')
ax[1].set_yticks(range(10))
ax[1].set_yticklabels(top10['gene'])
ax[1].invert_yaxis()
ax[1].set_xlabel('Importance')
ax[1].set_title('Top genes (sex)')

plt.tight_layout()
plt.savefig('results.png', dpi=150)
plt.show()
print("\nSaved: results.png, top_genes.csv")
