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
    fingerprints_file = "resources/smrt_fingerprints.csv"
    classifications_file = "resources/all_classified.tsv"
    output_file = "resources/smrt_classified.tsv"

    df = pd.read_csv(fingerprints_file)

    df['prefix'] = df['inchi'].apply(compute_inchikey_prefix)

    prefixes_set = set(df['prefix'].dropna())

    prefix_to_classes = {}

    with open(classifications_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                prefix = parts[0]
                if prefix in prefixes_set:
                    prefix_to_classes[prefix] = parts[1:]

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for _, row in df.iterrows():
            prefix = row['prefix']
            if prefix is not None and prefix in prefix_to_classes:
                classes = prefix_to_classes[prefix]
                out_f.write(row['inchi'] + '\t' + '\t'.join(classes) + '\n')
            else:
                out_f.write(row['inchi'] + '\n')
