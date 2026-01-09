from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import requests
from urllib.parse import quote
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdFingerprintGenerator

@dataclass
class MolParseResult:
    mol: Optional["Chem.Mol"]
    input_type: str
    canonical_smiles: Optional[str]
    warnings: List[str]


def _compute_fingerprints(mol: "Chem.Mol"):
    """
    Project fingerprint: Morgan(1024) + RDKit(166) + Pattern(1024) -> 2214 bits.
    """
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=166)

    morgan_1024 = morgan_gen.GetFingerprint(mol).ToBitString()
    rdkit_fp = rdkit_gen.GetFingerprint(mol).ToBitString()
    pattern_fp = rdmolops.PatternFingerprint(mol, fpSize=1024).ToBitString()

    full_fp = morgan_1024 + rdkit_fp + pattern_fp
    arr = np.fromiter((1 if ch == "1" else 0 for ch in full_fp), dtype=np.float32, count=2214)

    if arr.shape[0] != 2214:
        raise ValueError(f"Fingerprint size mismatch. Got {arr.shape[0]}, expected {2214}.")

    return arr


def _pubchem_inchi_to_smiles(inchi: str, timeout: int = 15):
    """
    Fallback: convert InChI -> CanonicalSMILES using PubChem PUG REST.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/{}/property/CanonicalSMILES/JSON"
    url = base_url.format(quote(inchi, safe=""))
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None
        return props[0].get("CanonicalSMILES")
    except Exception:
        return None


def parse_input_to_mol(smiles: Optional[str], inchi: Optional[str]):
    """
    Parse a molecular input (SMILES or InChI) into an RDKit molecule.
    """
    warnings: List[str] = []
    mol = None

    # To be able to work with inchis and smiles
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return MolParseResult(None, "smiles", None, ["Invalid SMILES"])
        can = Chem.MolToSmiles(mol, canonical=True)
        return MolParseResult(mol, "smiles", can, warnings)
    if inchi:
        # with RDKit
        try:
            mol = Chem.MolFromInchi(inchi)
        except Exception:
            mol = None

        if mol is not None:
            can = Chem.MolToSmiles(mol, canonical=True)
            return MolParseResult(mol, "inchi", can, warnings)

        # Fallback to PubChem conversion if RDKit fails
        smiles_from_pubchem = _pubchem_inchi_to_smiles(inchi)
        if smiles_from_pubchem:
            warnings.append("InChI converted to SMILES using PubChem (external service).")
            mol2 = Chem.MolFromSmiles(smiles_from_pubchem)
            if mol2 is None:
                return MolParseResult(None, "inchi", None, ["Invalid structure after PubChem conversion"])
            can = Chem.MolToSmiles(mol2, canonical=True)
            return MolParseResult(mol2, "inchi", can, warnings)
        return MolParseResult(None, "inchi", None, ["InChI could not be parsed/converted"])
    return MolParseResult(None, "unknown", None, [])


def featurize(smiles: Optional[str], inchi: Optional[str]) -> Tuple[np.ndarray, MolParseResult]:
    """
    Return (fingerprint_vector, parse_result).
    """
    parse_res = parse_input_to_mol(smiles, inchi)

    if parse_res.mol is None:
        return np.array([], dtype=np.float32), parse_res
    fp = _compute_fingerprints(parse_res.mol)
    return fp, parse_res
