import os
import mysql.connector
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import requests
import pandas as pd
from urllib.parse import quote
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdFingerprintGenerator


# - - - SEGREGATION COMPOUNDS DEPENDING ON CLASS - - -
def only_smrt_compounds(path_all_classified, path_rtdata, path_salida, path_no_coincidencias, chunk_size=100000):
    """
    Filters `all_classified.tsv` comparing its partial InChIKeys with the full ones found
    in `0186_rtdata.tsv`. Saves the ones that share the same partial InChIKeys.

    :param path_all_classified: path to file with compound parents
    :param path_rtdata: path to file that contains rtdata
    :param path_salida: exit path
    :param chunk_size: size of chunks
    """
    contains_inchikey = []
    doesNotcontain_inchikey = []

    rtdata_df = pd.read_csv(path_rtdata, sep='\t', usecols=['canonical.inchikey.std'])
    inchikey_set = set(rtdata_df['canonical.inchikey.std'].dropna().apply(lambda x: x.split('-')[0]))

    column_name = [f"col_{i}" for i in range(1441)]
    chunk_iter = pd.read_csv(
        path_all_classified,
        sep='\t',
        header=None,
        names=column_name,
        chunksize=chunk_size,
        dtype=str,
        engine='python'
    )

    for i, chunk in enumerate(chunk_iter):
        mask = chunk.iloc[:, 0].isin(inchikey_set)
        contains_inchikey_chunk = chunk[mask]
        doesNotcontain_inchikey_chunk = chunk[~mask]

        if not contains_inchikey_chunk.empty:
            contains_inchikey.append(contains_inchikey_chunk)
        if not doesNotcontain_inchikey_chunk.empty:
            doesNotcontain_inchikey.append(doesNotcontain_inchikey_chunk)

        print(f"Chunk {i + 1}")

    if contains_inchikey:
        print(f"Filed saved correctly: {path_salida}...")
        pd.concat(contains_inchikey).to_csv(path_salida, sep='\t', index=False)

    if doesNotcontain_inchikey:
        print(f"Filed saved correctly: {path_no_coincidencias}...")
        pd.concat(doesNotcontain_inchikey).to_csv(path_no_coincidencias, sep='\t', index=False)

def repetition_parents(filepath, output_path):
    """
    Finds the parents that are repeated across the compounds in a given dataset.

    :param filepath: path to file with compound parents dataset
    :param output_path: path to exit file
    """
    df = pd.read_csv(filepath, sep='\t')
    df = df.iloc[:, 1:]

    repetitions = defaultdict(int)
    lines = defaultdict(set)

    for index, row in df.iterrows():
        for valor in row.dropna().unique():
            repetitions[valor] += (row == valor).sum()
            lines[valor].add(index)

    resultado = pd.DataFrame({
        'Parent_Name': repetitions.keys(),
        'Repetitions': repetitions.values(),
        'N_of_compounds': [len(lines[nombre]) for nombre in repetitions.keys()]
    })

    resultado.to_csv(output_path, sep='\t', index=False)
    print(f"Filed saved correctly: {output_path}")

def compounds_per_class(classified_rtdata, classesFile, group_crit, outputFile, histogram_path):
    """
        Computes from the file with all rt compounds which belong to a certain class in order to
        classify said compounds and arrange them in groups for a future histograph representation.

        :param classified_rtdata: input file with all the rt compounds and its direct and alternative parents
        :param classesFile: path to the file which contains which parents belong to which class
        :param outputFile: path to the output file
        :param histogram_path: path to where the histogram will be saved
    """
    df_compoundsrt = pd.read_csv(classified_rtdata, sep='\t')
    df_superclases = pd.read_csv(classesFile, sep='\t')

    collumns_components = [col for col in df_compoundsrt.columns if col != 'col_0']
    component_to_compound = defaultdict(set)
    superclass_compounds = defaultdict(set)
    output = []

    for _, row in df_compoundsrt.iterrows():
        compounds = row['col_0']
        for col in collumns_components:
            components = row[col]
            if pd.notna(components):
                key = components.strip().lower()
                component_to_compound[key].add(compounds)


    for _, row in df_superclases.iterrows():
        group_class = row[group_crit]
        if pd.isna(row['components']) or group_class.lower() == "not found":
            continue
        components = [c.strip().lower() for c in row['components'].split(',')]
        for comp in components:
            if comp in component_to_compound:
                superclass_compounds[group_class].update(component_to_compound[comp])

    for group_class, compound in superclass_compounds.items():
        output.append({
            group_crit: group_class,
            'compounds': ', '.join(sorted(compound)),
            'n_compounds': len(compound)
        })

    df_salida = pd.DataFrame(output)
    df_salida.to_csv(outputFile, sep='\t', index=False)
    plot_class_histogram(outputFile, histogram_path, group_crit, min_count=10 )
    print(f"Filed saved correctly: {outputFile}")


def subgroups_basedon_superclass(sqlHost, sqlUser, sqlPassword, sqlDatabase, group_crit, parentsFile, outputFile):
    """
        Access the database compounds.20231121.spl and matches the given parents in parentsFile with the ones
        that appear within the database. Then it associates the parents with the superclass they each descend from.

        :param sqlHost: Host of the MySQL database
        :param sqlUser: Username used in MySQL
        :param sqlPassword: Password used in MySQL
        :param sqlDatabase: Name of the database
        :param parentsFile: Path to the file which contains the parents from the compounds
        :param outputFile: Path where the output file will be stored
    """
    conn = mysql.connector.connect(
        host=sqlHost,
        user=sqlUser,
        password=sqlPassword,
        database=sqlDatabase
    )
    cursor = conn.cursor()
    super_class_componentes = {}
    chunk_iter = pd.read_csv(parentsFile, sep='\t', chunksize=1)

    for chunk in chunk_iter:
        nombre = chunk.iloc[0]['Parent_Name']
        try:
            cursor.execute(
                "SELECT node_id FROM classyfire_classification_dictionary WHERE node_name = %s",
                (nombre,)
            )
            res_node_id = cursor.fetchone()
            if not res_node_id:
                grouping_class = "not found"
            else:
                node_id = res_node_id[0]

                cursor.execute(
                    "SELECT main_class FROM classyfire_classification WHERE node_id = %s",
                    (node_id,)
                )
                res_group = cursor.fetchone()
                if not res_group:
                    grouping_class = "not found"
                else:
                    class_id = res_group[0]

                    # Obtener nombre del super_class
                    cursor.execute(
                        "SELECT node_name FROM classyfire_classification_dictionary WHERE node_id = %s",
                        (class_id,)
                    )
                    res_super_name = cursor.fetchone()
                    if not res_super_name:
                        grouping_class = "not found"
                    else:
                        grouping_class = res_super_name[0]
            if grouping_class not in super_class_componentes:
                super_class_componentes[grouping_class] = []

            super_class_componentes[grouping_class].append(nombre)
        except Exception as e:
            print(f"Error processing '{nombre}': {e}")
            continue

    cursor.close()
    conn.close()

    df_salida = pd.DataFrame([
        {
            group_crit: gc,
            'components': ', '.join(super_class_componentes[gc])
        }
        for gc in super_class_componentes
    ])

    df_salida.to_csv(outputFile, sep='\t', index=False)

def ids_subgroupCompounds(subgroups_file, rtdata_file, choosen_class, class_type):
    """
        Obtains the ids of the compounds that are associated with a specific superclass.

        :param subgroups_file: path to file with compound parents dataset
        :param rtdata_file: path to exit file
        :param choosen_class: choosen superclass
        :return List with all ids rom compounds in group
    """
    df_subgroups = pd.read_csv(subgroups_file, sep='\t')
    df_rtdata = pd.read_csv(rtdata_file, sep='\t')
    cids_found = []
    superclass_compounds = df_subgroups[df_subgroups[class_type] == choosen_class]
    compounds_str = superclass_compounds.iloc[0]['compounds']
    inchikeys_sc_compounds = [mol.strip() for mol in compounds_str.split(',') if mol.strip()]

    if superclass_compounds.empty:
        print(f"Could not  find: {choosen_class}")
        return []

    for _, row in df_rtdata.iterrows():
        inchikey_completo = row.get('canonical.inchikey.std')
        pubchem_cid = row.get('pubchem.cid')

        if pd.notna(inchikey_completo) and pd.notna(pubchem_cid):
            inchikey_parcial = inchikey_completo.split('-')[0]
            if inchikey_parcial in inchikeys_sc_compounds:
                cids_found.append(pubchem_cid)

    return cids_found

def obtain_fpSubgroup(subgroups, rtdata, choosen_class, class_type, all_fingerprints, subgroup_fingerprints):
    """
        Obtains the ids of the compounds that are associated with a specific superclass.

        :param subgroups_file: path to file with compound parents dataset
        :param rtdata_file: path to exit file
        :param choosen_superclass: choosen superclass
        :return List with all ids rom compounds in group
    """
    id_list = ids_subgroupCompounds(subgroups, rtdata, choosen_class, class_type)

    df = pd.read_csv(all_fingerprints)

    df_filtrado = df[df['pid'].isin(id_list)]

    df_filtrado.to_pickle(subgroup_fingerprints)
    print(f"Filed saved as: {subgroup_fingerprints} (Total lines: {len(df_filtrado)})")

def plot_class_histogram(file_path, save_path, class_crit, min_count=10000):
    """
        Generates a histogram of the compounds associated with a specific class.

        :param file_path: path with file used to obtain the data for the histogram plot
        :param save_path: path to exit file where the histogram is saved
        :param class_crit: criteria used for grouping the compounds (superclass, mainclass, subclass....)
        :param min_count: minimum number of compounds associated with a specific class to be considered
    """
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["n_compounds"] > min_count].sort_values(by="n_compounds", ascending=False)
    df = df.sort_values(by="n_compounds", ascending=False)

    plt.figure(figsize=(14, 7))
    plt.bar(df[class_crit], df["n_compounds"], color="steelblue")
    plt.xticks(rotation=75, ha="right")
    plt.xlabel("Mainclass")
    plt.ylabel("Number of compounds")
    plt.title("Molecules per " + class_crit)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Histogram with classes saved in: {save_path}")
# - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - CREATION FINGERPRINTS - - -
def respect_rate_limit(last_call_ts, delay_sec):
    """
     Gives a new timestamp after the rate limit.

     :param last_call_ts: Timestamp of the last call
     :param delay_sec: Min number of seconds between calls
     :return: New timestamp after waiting
     """
    now = time.monotonic()
    elapsed = now - last_call_ts
    if elapsed < delay_sec:
        time.sleep(delay_sec - elapsed)
    return time.monotonic()

def get_smiles_from_inchi(inchi, cache,last_call_ts,*,delay_sec = 0.5,max_retries = 3,timeout = 15):
    """
    Consults PubChem for CanonicalSMILES from the InChI

    :param inchi: InChI string of the compound to query
    :param cache: Dictionary for caching results {inchi: smiles}
    :param last_call_ts: Timestamp of the last API call
    :param delay_sec: Minimum delay between API calls in seconds
    :param max_retries: Maximum number of retries if the request fails
    :param timeout: Maximum time to wait for the HTTP response
    :return: tuple(smiles|None, cache, last_call_ts)
    """
    if not inchi:
        return None, cache, last_call_ts

    if inchi in cache:
        return cache[inchi], cache, last_call_ts

    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/{}/property/CanonicalSMILES/JSON"
    url = base_url.format(quote(inchi, safe=""))

    for attempt in range(1, max_retries + 1):
        try:
            last_call_ts = respect_rate_limit(last_call_ts, delay_sec)
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None, cache, last_call_ts
            resp.raise_for_status()

            data = resp.json()
            smiles = (
                data.get("PropertyTable", {})
                    .get("Properties", [{}])[0]
                    .get("CanonicalSMILES")
            )
            if smiles:
                cache[inchi] = smiles
                return smiles, cache, last_call_ts
            return None, cache, last_call_ts

        except requests.exceptions.RequestException:
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s...
                continue
            return None, cache, last_call_ts
        except (KeyError, IndexError, ValueError):
            return None, cache, last_call_ts

    return None, cache, last_call_ts

def compute_fingerprints(mol, morgan_gen, rdkit_gen):
    """
    Generates fingerprints using multiple fingerprint types and combining them.
    """
    morgan_1024 = morgan_gen.GetFingerprint(mol).ToBitString()
    rdkit_fp = rdkit_gen.GetFingerprint(mol).ToBitString()
    pattern_fp = rdmolops.PatternFingerprint(mol, fpSize=1024).ToBitString()
    full_fp = morgan_1024 + rdkit_fp + pattern_fp
    return [int(bit) for bit in full_fp]

def process_rtdata(file_path, out_pkl_path, out_err_tsv, delay_sec= 0.5, max_retries= 3,timeout= 15,):
    """
    Generates the fingerprint.pkl file and stores it.
    """
    df_input = pd.read_csv(file_path, sep="\t")
    rows = df_input[["pubchem.cid", "rt", "pubchem.inchi"]].values.tolist()

    data, errores = [], []

    cache = {}
    last_call_ts = 0.0

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=166)

    for pid, rt, inchi_str in rows:
        try:
            rt_s = float(rt) * 60

            smiles, cache, last_call_ts = get_smiles_from_inchi(
                inchi_str, cache, last_call_ts,
                delay_sec=delay_sec, max_retries=max_retries, timeout=timeout
            )
            mol = Chem.MolFromSmiles(smiles) if smiles else None

            if mol is None:
                mol = Chem.MolFromInchi(inchi_str)

            if mol is None:
                errores.append((pid, rt_s, inchi_str, "MolFromInchi/Smiles returned None"))
                continue

            bits = compute_fingerprints(mol, morgan_gen, rdkit_gen)
            data.append([pid, rt_s] + bits)

        except Exception as e:
            errores.append((pid, rt, inchi_str, str(e)))
            continue

    if data:
        num_bits = len(data[0]) - 2
        columns = ["pid", "rt"] + [f"V{i}" for i in range(num_bits)]
        df_fps = pd.DataFrame(data, columns=columns)
    else:
        df_fps = pd.DataFrame(columns=["pid", "rt"])

    df_fps.to_pickle(out_pkl_path)

    df_errores = pd.DataFrame(errores, columns=["pid", "rt", "inchi", "error"])
    df_errores.to_csv(out_err_tsv, sep="\t", index=False)

    print(f"Guardado: {out_pkl_path} | Registros: {len(df_fps)} | Errores: {len(df_errores)}")
# - - - - - - - - - - - - - - - - -