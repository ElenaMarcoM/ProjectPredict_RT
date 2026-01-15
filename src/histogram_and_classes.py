import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def get_class_dict_and_histogram(top_n):
    # Files
    tsv_file = "resources/smrt_classified.tsv"
    hist_file = "results/histogram.png"

    data = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 0:
                inchi = parts[0]
                classes = parts[1:] if len(parts) > 1 else []
                data.append((inchi, classes))

    class_to_compounds = defaultdict(list)
    for inchi, classes in data:
        for cls in classes:
            class_to_compounds[cls].append(inchi)

    full_result = {cls: (len(compounds), compounds) for cls, compounds in class_to_compounds.items()}

    # Select top_n classes by frequency
    sorted_classes = sorted(full_result.keys(), key=lambda k: full_result[k][0], reverse=True)[:top_n]
    result = {cls: full_result[cls] for cls in sorted_classes}

    if result:
        sorted_counts = [result[k][0] for k in sorted_classes]

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

# --- New Function ---
def plot_training_history(history, title="Model Loss (MAE)", figsize=(10, 6)):
    """
    Plots training and validation loss over epochs.

    Parameters:
    - history: the history object returned by model.fit()
    - title: title of the plot
    - figsize: size of the figure
    """
    plt.figure(figsize=figsize)

    # Plot training & validation loss
    plt.plot(history.history['loss'], label='Training Loss (MAE)', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss (MAE)', linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Optional: add a marker for the best validation loss
    best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7,
                label=f'Best Val Loss (Epoch {best_epoch})')

    plt.legend()
    plt.tight_layout()
    plt.show()
# --------------------

