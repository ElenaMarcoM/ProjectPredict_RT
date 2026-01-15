import requests
import csv
import os
import time


def _submit_query(inchi):
    url = 'http://classyfire.wishartlab.com/queries'
    # Format as ID\tstructure for multiple, but for single, use empty ID
    query_data = f"ID\t{inchi}"
    payload = {
        'label': 'grok_script',
        'query': query_data,
        'query_type': 'STRUCTURE'
    }
    try:
        response = requests.post(url, data=payload)
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text}")
        if response.status_code == 201:
            return response.json()['id']
    except Exception as e:
        print(f"Exception: {e}")
    return None


def _get_query_status(query_id):
    url = f'http://classyfire.wishartlab.com/queries/{query_id}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def _get_classifications(inchi):
    query_id = _submit_query(inchi)
    if not query_id:
        return []
    start = time.time()
    while time.time() - start < 300:  # 5 min timeout
        status = _get_query_status(query_id)
        if status and status.get('classified_entities', 0) == status.get('number_of_entities', 0):
            if 'entities' in status and status['entities']:
                entity = status['entities'][0]
                ancestors = entity.get('ancestors', [])
                # Filter out terms containing "entity"
                filtered = [anc for anc in ancestors if 'entity' not in anc.lower()]
                return sorted(filtered)
        time.sleep(5)
    return []

def classify_with_classyfire():
    inchis = []
    with open('resources/all_datasets.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('dataset') == 'smrt':
                inchi = row.get('inchi')
                if inchi:
                    inchis.append(inchi)

    output_file = 'resources/all_smrt_classifications.csv'

    existing = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row:
                    inchi = row[0]
                    classes = row[1:] if len(row) > 1 else []
                    existing[inchi] = classes

    for inchi in inchis:
        if inchi not in existing:
            existing[inchi] = []

    for inchi, classes in list(existing.items()):
        if classes:
            continue

        print(f"Processing {inchi}")
        all_classes = _get_classifications(inchi)

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            if all_classes:
                writer.writerow([inchi] + all_classes)
                existing[inchi] = all_classes
            else:
                writer.writerow([inchi])
                existing[inchi] = []
        # Rate limiting between compounds
        time.sleep(1)