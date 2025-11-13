import matplotlib.pyplot as plt


def save_results_histogram(results_df):
    output_file = "tests_for_smrt/results/mae_vs_frequency.png"

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['Frequency'], results_df['Test_MAE'], alpha=0.6, edgecolors='w', s=50)

    # Add labels and title
    plt.xlabel('Frequency (Number of Compounds)')
    plt.ylabel('Test MAE (Mean Absolute Error)')
    plt.title('Test MAE vs. Frequency for Chemical Classes')

    # Use logarithmic scale for x-axis if frequencies vary widely
    if results_df['Frequency'].max() / results_df['Frequency'].min() > 10:
        plt.xscale('log')

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ensure layout is tight to prevent clipping
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file)
    plt.close()
    print("Histogram saved to:", output_file)



    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['Frequency'], results_df['Normalized_MAE'], alpha=0.6, edgecolors='w', s=50)

    # Add labels and title
    plt.xlabel('Frequency (Number of Compounds)')
    plt.ylabel('Test normalized MAE (Mean Absolute Error)')
    plt.title('Test normalized MAE vs. Frequency for Chemical Classes')

    # Use logarithmic scale for x-axis if frequencies vary widely
    if results_df['Frequency'].max() / results_df['Frequency'].min() > 10:
        plt.xscale('log')

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ensure layout is tight to prevent clipping
    plt.tight_layout()

    # Save the plot
    output_file = "tests_for_smrt/results/normalized_mae_vs_frequency.png"
    plt.savefig(output_file)
    plt.close()

    print("Histogram saved to:", output_file)