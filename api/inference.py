from __future__ import annotations
import pickle
from dataclasses import dataclass
import os
import joblib
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from .features import featurize

# /pred output
@dataclass
class PredictOutput:
    input_type: str
    input_value: str
    rt_pred_seconds: float
    model_name: str
    warnings: List[str]


class InferenceService:
    def __init__(self, model_path: str, scaler_path: str, model_name="generic"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model_name = model_name

        self.model = None
        self.scaler = None

    def load(self) -> None:
        # Import tensorflow
        from tensorflow.keras.models import load_model

        # Ensure the model/scaler are availeable
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")

        # Load model and scaler
        self.model = load_model(str(self.model_path), compile=False)
        with open(self.scaler_path, "rb") as f:
            self.scaler = joblib.load(f)

    # Check model and scaler are loaded
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    # Use model to predict rt from inchi/smile of a given compound
    def predict(self, smiles: Optional[str], inchi: Optional[str]) -> PredictOutput:
        if not self.is_loaded():
            raise RuntimeError("Model/scaler not loaded")

        if smiles is not None:
            input_value = smiles
        elif inchi is not None:
            input_value = inchi
        else:
            input_value = ""

        fp, parse_res = featurize(smiles, inchi)
        # Ensure valid fp
        if fp.size == 0:
            raise ValueError("SMILES/INCHI not found")
        if fp.shape[0] != 2214:
            raise ValueError(f"Fingerprint length {fp.shape[0]} != {2214}")

        x = fp.reshape(1, -1)
        # Predict (scaled)
        y_scaled = self.model.predict(x, verbose=0)
        y_scaled = np.array(y_scaled).reshape(1, -1)
        # Inverse transform to seconds
        y = self.scaler.inverse_transform(y_scaled).flatten()

        return PredictOutput(
            input_type=parse_res.input_type,
            input_value=input_value,
            rt_pred_seconds=y,
            model_name=self.model_name,
            warnings=parse_res.warnings + parse_res.warnings,
        )
