import pandas as pd

# Function to process a CSV file and count entries, unique compounds, and find repeated compounds
def analyze_csv(file_path, id_column, inchi_column, delimiter=','):
    try:
        # Read the CSV file with specified delimiter, skip bad lines if necessary
        df = pd.read_csv(file_path, delimiter=delimiter, on_bad_lines='warn', quoting=0)

        # Total number of entries
        total_entries = len(df)

        # Number of unique compounds by ID column
        unique_by_id = df[id_column].nunique()

        # Number of unique compounds by InChI column
        unique_by_inchi = df[inchi_column].nunique()

        # Find repeated compounds by InChI (compounds with duplicate InChI values)
        inchi_counts = df[inchi_column].value_counts()
        repeated_inchi = inchi_counts[inchi_counts > 1].index
        repeated_rows = df[df[inchi_column].isin(repeated_inchi)]

        return {
            'file': file_path,
            'total_entries': total_entries,
            'unique_by_id': unique_by_id,
            'unique_by_inchi': unique_by_inchi,
            'repeated_rows': repeated_rows
        }
    except Exception as e:
        return {'file': file_path, 'error': str(e)}

# File paths
smrt_fingerprints_path = '/home/grf/PycharmProjects/metlin_foundation/tests_for_smrt/resources/smrt_fingerprints.csv'
smrt_path = '/home/grf/PycharmProjects/metlin_foundation/resources/smrt.csv'

# Analyze smrt_fingerprints.csv (pid, inchi, comma-separated)
result1 = analyze_csv(smrt_fingerprints_path, 'pid', 'inchi', delimiter=',')

# Analyze smrt.csv (pubchem, inchi, semicolon-separated)
result2 = analyze_csv(smrt_path, 'pubchem', 'inchi', delimiter=';')

# Print results for smrt_fingerprints.csv
print("Results for smrt_fingerprints.csv:")
if 'error' in result1:
    print(f"Error: {result1['error']}")
else:
    print(f"Total entries: {result1['total_entries']}")
    print(f"Unique compounds by pid: {result1['unique_by_id']}")
    print(f"Unique compounds by inchi: {result1['unique_by_inchi']}")
    # Print repeated compounds
    print("\n\nRepeated compounds by inchi:")
    if not result1['repeated_rows'].empty:
        print(result1['repeated_rows'].to_string(index=False))
    else:
        print("No repeated compounds found.")

# Print results for smrt.csv
print("\nResults for smrt.csv:")
if 'error' in result2:
    print(f"Error: {result2['error']}")
else:
    print(f"Total entries: {result2['total_entries']}")
    print(f"Unique compounds by pubchem: {result2['unique_by_id']}")
    print(f"Unique compounds by inchi: {result2['unique_by_inchi']}")
    # Print repeated compounds
    print("\n\nRepeated compounds by inchi:")
    if not result2['repeated_rows'].empty:
        print(result2['repeated_rows'].to_string(index=False))
    else:
        print("No repeated compounds found.")


# File paths
smrt_fingerprints_path = '/home/grf/PycharmProjects/metlin_foundation/tests_for_smrt/resources/smrt_fingerprints.csv'
smrt_path = '/home/grf/PycharmProjects/metlin_foundation/resources/smrt.csv'

# Analyze smrt_fingerprints.csv (pid, inchi, comma-separated)
result1 = analyze_csv(smrt_fingerprints_path, 'pid', 'inchi', delimiter=',')

# Analyze smrt.csv (pubchem, inchi, semicolon-separated)
result2 = analyze_csv(smrt_path, 'pubchem', 'inchi', delimiter=';')

# Print results
print("Results for smrt_fingerprints.csv:")
if 'error' in result1:
    print(f"Error: {result1['error']}")
else:
    print(f"Total entries: {result1['total_entries']}")
    print(f"Unique compounds by pid: {result1['unique_by_id']}")
    print(f"Unique compounds by inchi: {result1['unique_by_inchi']}")

print("\nResults for smrt.csv:")
if 'error' in result2:
    print(f"Error: {result2['error']}")
else:
    print(f"Total entries: {result2['total_entries']}")
    print(f"Unique compounds by pubchem: {result2['unique_by_id']}")
    print(f"Unique compounds by inchi: {result2['unique_by_inchi']}")