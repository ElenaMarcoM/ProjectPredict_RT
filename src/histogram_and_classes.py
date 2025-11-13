import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def get_class_dict_and_histogram(top_n):
    # Files
    tsv_file = "tests_for_smrt/resources/smrt_classified.tsv"
    hist_file = "tests_for_smrt/results/histogram.png"

    # Read the TSV file line by line
    data = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 0:
                inchi = parts[0]
                classes = parts[1:] if len(parts) > 1 else []
                data.append((inchi, classes))

    # Collect compounds per class
    class_to_compounds = defaultdict(list)
    for inchi, classes in data:
        for cls in classes:
            class_to_compounds[cls].append(inchi)

    # Create the result dictionary with all classes
    full_result = {cls: (len(compounds), compounds) for cls, compounds in class_to_compounds.items()}

    # Select top_n classes by frequency
    sorted_classes = sorted(full_result.keys(), key=lambda k: full_result[k][0], reverse=True)[:top_n]
    # Create result dictionary with only top_n classes
    result = {cls: full_result[cls] for cls in sorted_classes}

    # Prepare data for histogram (bar plot)
    if result:
        sorted_counts = [result[k][0] for k in sorted_classes]

        # Create a wider figure to prevent overlap
        plt.figure(figsize=(30, 8))
        plt.bar(sorted_classes, sorted_counts)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.xlabel('Classes')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of Top {top_n} out of {len(full_result)} Chemical Classes')
        plt.tight_layout()
        plt.savefig(hist_file)
        plt.close()

    return result
