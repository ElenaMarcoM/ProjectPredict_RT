import pandas as pd


def add_inchi_to_alvadesc_smrt():
    # Files
    smrt_file = "resources/smrt.csv"
    fingerprints_file = "tests_for_smrt/resources/smrt_fingerprints.csv"  # or fingerprints_file = "tests_for_smrt/resources/smrt_descriptors.csv"
    output_file = "tests_for_smrt/resources/smrt_fingerprints.csv"  # or output_file = "tests_for_smrt/resources/smrt_descriptors.csv"

    # Read the input CSV files
    smrt_df = pd.read_csv(smrt_file, sep=';')
    fingerprints_df = pd.read_csv(fingerprints_file)

    # Merge the dataframes on pubchem and pid columns
    merged_df = fingerprints_df.merge(
        smrt_df[['pubchem', 'inchi']],
        left_on='pid',
        right_on='pubchem',
        how='left'
    )

    # Drop the redundant pubchem column
    merged_df = merged_df.drop('pubchem', axis=1)

    # Reorder columns to place 'inchi' after 'rt' and before 'V0'
    cols = merged_df.columns.tolist()
    rt_index = cols.index('rt')
    inchi_index = cols.index('inchi')

    # Move inchi to the position after rt
    cols.insert(rt_index + 1, cols.pop(inchi_index))
    merged_df = merged_df[cols]

    # Save the result to the output CSV file (overwriting the input fingerprints file)
    merged_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    add_inchi_to_alvadesc_smrt()