import os
from api.features import featurize
from api.inference import InferenceService
from api.schemas import PredictRequest

def test_smiles_ok():
    fp, info = featurize(smiles="CCO", inchi=None)  # etanol
    print("=== TEST SMILES OK ===")
    print("mol_ok:", info.mol is not None)
    print("fingerprint_len:", fp.shape[0], "expected:", 2214)
    print("canonical_smiles:", info.canonical_smiles)
    print("warnings:", info.warnings)
    print()

def test_smiles_invalid():
    fp, info = featurize(smiles="NOT_A_SMILES", inchi=None)
    print("=== TEST SMILES INVALID ===")
    print("mol:", info.mol)
    print("fp_size:", fp.size)
    print("warnings:", info.warnings)
    print()

def test_missing_input():
    fp, info = featurize(smiles=None, inchi=None)
    print("=== TEST MISSING INPUT ===")
    print("input_type:", info.input_type)
    print("mol:", info.mol)
    print("fp_size:", fp.size)
    print("warnings:", info.warnings)
    print()

def test_inchi_ok():
    fp, info = featurize(smiles=None, inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
    print("=== TEST INCHI OK ===")
    print("mol_ok:", info.mol is not None)
    print("fp_size:", fp.size)
    print("input_type:", getattr(info, "input_type", None))
    print("canonical_smiles:", getattr(info, "canonical_smiles", None))
    print("warnings:", info.warnings)
    print()

def test_inchi_invalid():
    fp, info = featurize(smiles=None, inchi="NOT_AN_INCHI")
    print("=== TEST INCHI INVALID ===")
    print("mol:", info.mol)
    print("fp_size:", fp.size)
    print("warnings:", info.warnings)
    print()

def test_inference_ok():
    print("=== TEST INFERENCE OK ===")
    root = os.getcwd()
    model_path = os.path.join(root, "resources", "model_simple.h5")
    scaler_path = os.path.join(root, "recources", "scaler.pkl")

    service = InferenceService(model_path=model_path, scaler_path=scaler_path, model_name="generic_v1")
    service.load()

    out = service.predict(smiles="CCO", inchi=None)
    print("loaded:", service.is_loaded())
    print("rt_pred_seconds:", out.rt_pred_seconds)
    print("model:", out.model_name)
    print("canonical_smiles:", out.canonical_smiles)
    print("warnings:", out.warnings)
    print()

def test_inference_inchi_ok():
    print("=== TEST INFERENCE InChI OK ===")

    # InChI de etanol
    inchi_ethanol = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"

    root = os.getcwd()
    model_path = os.path.join(root, "resources", "model_simple.h5")
    scaler_path = os.path.join(root, "resources", "scaler.pkl")

    service = InferenceService(model_path=model_path, scaler_path=scaler_path, model_name="generic_v1")
    service.load()

    out = service.predict(smiles=None, inchi=inchi_ethanol)
    print("loaded:", service.is_loaded())
    print("rt_pred_seconds:", out.rt_pred_seconds)
    print("model:", out.model_name)
    print("canonical_smiles:", out.canonical_smiles)
    print("warnings:", out.warnings)
    print()

def test_inference_missing_input():
    print("=== TEST INFERENCE MISSING INPUT ===")
    root = os.getcwd()
    model_path = os.path.join(root, "resources", "model_simple.h5")
    scaler_path = os.path.join(root, "resources", "scaler.pkl")

    service = InferenceService(model_path=model_path, scaler_path=scaler_path, model_name="generic_v1")
    service.load()

    try:
        service.predict(smiles=None, inchi=None)
    except Exception as e:
        print("error_type:", type(e).__name__)
        print("error_msg:", str(e))
    print()

def test_schemas_ok_smiles():
    print("=== TEST SCHEMAS OK (SMILES) ===")
    req = PredictRequest(smiles="CCO")
    print(req)
    print()

def test_schemas_ok_inchi():
    print("=== TEST SCHEMAS OK (InChI) ===")
    req = PredictRequest(inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
    print(req)
    print()

def test_schemas_missing():
    print("=== TEST SCHEMAS MISSING INPUT ===")
    try:
        PredictRequest()
    except Exception as e:
        print("error_type:", type(e).__name__)
        print("error_msg:", str(e))
    print()

if __name__ == "__main__":
    test_schemas_ok_smiles()
    test_schemas_ok_inchi()


