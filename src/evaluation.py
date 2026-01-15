import matplotlib.pyplot as plt


def save_results_histogram(results_df):
    output_file = "results1/mae_vs_frequency.png"

    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['Frequency'], results_df['Test_MAE'], alpha=0.6, edgecolors='w', s=50)

    plt.xlabel('Frequency (Number of Compounds)')
    plt.ylabel('Test MAE (Mean Absolute Error)')
    plt.title('Test MAE vs. Frequency for Chemical Classes')

    if results_df['Frequency'].max() / results_df['Frequency'].min() > 10:
        plt.xscale('log')

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()
    print("Histogram saved to:", output_file)



    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['Frequency'], results_df['Normalized_MAE'], alpha=0.6, edgecolors='w', s=50)

    plt.xlabel('Frequency (Number of Compounds)')
    plt.ylabel('Test normalized MAE (Mean Absolute Error)')
    plt.title('Test normalized MAE vs. Frequency for Chemical Classes')

    if results_df['Frequency'].max() / results_df['Frequency'].min() > 10:
        plt.xscale('log')

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the plot
    output_file = "results1/normalized_mae_vs_frequency.png"
    plt.savefig(output_file)
    plt.close()

    print("Histogram saved to:", output_file)