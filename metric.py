import ast
import json
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score
)

# Read and parse the first file
with open('merged_matches.txt', 'r', encoding='utf-8') as f:
    manual_annos_str = f.read()
    manual_annos = ast.literal_eval(manual_annos_str)  # Parse as a dictionary

# Read and parse the second file
with open('merged_results.txt', 'r', encoding='utf-8') as f:
    predicted_str = f.read()
    predicted = json.loads(predicted_str)  # Parse as a dictionary

# 1. Find matching indices
matches =[]
for k, true_label in manual_annos.items():
    # Ensure k is a string because the keys in 'predicted' are strings
    pred_label = predicted[str(k)]["predicted_cell_type"]

    # Compare after normalizing to lowercase
    if true_label.lower() == pred_label.lower():
        matches.append(k)

# 2. Output the results
print(f"Number of matched spots: {len(matches)} / {len(manual_annos)}")
#print("Matched spot indices:", matches)

# 1. Construct true_labels and pred_labels
keys = sorted(manual_annos.keys())
true_labels = np.array([manual_annos[k] for k in keys])
pred_labels = np.array([predicted[str(k)]["predicted_cell_type"] for k in keys])

# 2. Normalize to lowercase
true_norm = np.char.lower(true_labels)
pred_norm = np.char.lower(pred_labels)

# 3. Calculate clustering metrics
metrics = {
    "ARI": adjusted_rand_score(true_norm, pred_norm),
    "NMI": normalized_mutual_info_score(true_norm, pred_norm),
    "AMI": adjusted_mutual_info_score(true_norm, pred_norm),
    "V-measure": v_measure_score(true_norm, pred_norm),
    "Homogeneity": homogeneity_score(true_norm, pred_norm)
}

# 4. Calculate classification metrics
metrics["Micro-F1"] = f1_score(true_norm, pred_norm, average='micro', zero_division=0)
metrics["Macro-F1"] = f1_score(true_norm, pred_norm, average='macro', zero_division=0)
metrics["MCC"] = matthews_corrcoef(true_norm, pred_norm)
metrics["Cohen's κ"] = cohen_kappa_score(true_norm, pred_norm)

# 5. Output metrics
for name, val in metrics.items():
    print(f"{name}: {val:.4f}")