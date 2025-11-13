import pandas as pd
from rdkit import Chem


def compute_inchikey_prefix(inchi):
    # Directly use the InChI string for InchiToInchiKey
    if not inchi or not isinstance(inchi, str):
        return None
    try:
        inchikey = Chem.InchiToInchiKey(inchi)
        return inchikey[:14]
    except:
        return None

def big_file_classifier():
    # Files
    fingerprints_file = "tests_for_smrt/resources/smrt_fingerprints.csv"
    classifications_file = "tests_for_smrt/resources/classifications_sorted.tsv"
    output_file = "tests_for_smrt/resources/smrt_classified.tsv"

    # Load the fingerprints CSV
    df = pd.read_csv(fingerprints_file)

    # Compute InChIKey prefixes for each InChI
    df['prefix'] = df['inchi'].apply(compute_inchikey_prefix)

    # Get unique prefixes (in case of duplicates, though unlikely)
    prefixes_set = set(df['prefix'].dropna())

    # Dictionary to store classifications by prefix
    prefix_to_classes = {}

    # Read the large TSV line by line
    with open(classifications_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                prefix = parts[0]
                if prefix in prefixes_set:
                    prefix_to_classes[prefix] = parts[1:]

    # Prepare the output
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for _, row in df.iterrows():
            prefix = row['prefix']
            if prefix is not None and prefix in prefix_to_classes:
                classes = prefix_to_classes[prefix]
                out_f.write(row['inchi'] + '\t' + '\t'.join(classes) + '\n')
            else:
                # If no match, just write the InChI with no classes
                out_f.write(row['inchi'] + '\n')
