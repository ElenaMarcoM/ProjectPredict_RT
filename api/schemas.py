from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator

# Input of /predict for user interface
class PredictRequest(BaseModel):
    smiles: Optional[str] = Field(
        default=None,
        description="SMILES string of the molecule.",
        examples=["COC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=CC=C3)Br)OC"]
    )
    inchi: Optional[str] = Field(
        default=None,
        description="InChI string of the molecule.",
        examples=["InChI=1S/C16H14BrN3O2/c1-21-14-7-12-13(8-15(14)22-2)18-9-19-16(12)20-11-5-3-4-10(17)6-11/h3-9H,1-2H3,(H,18,19,20)"]
    )

    # Check the input given is valid
    @model_validator(mode="after")
    def check_input(self):
        if not self.smiles and not self.inchi:
            raise ValueError("SMILES/INCHI not found")
        return self

# Output of /predict for user interface
class PredictResponse(BaseModel):
    input_type: str = Field(
        description="Type of input provided.",
        examples=["smiles"]
    )
    input: str = Field(
        description="Original input string provided, can be SMILES or InChI.",
        examples=["COC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=CC=C3)Br)OC"]
    )
    rt_predicted: float = Field(
        description="Predicted retention time (RT).",
        examples=[790.749]
    )
    units: str = Field(
        default="s",
        description="Units of the predicted RT (seconds).",
        examples=["s"]
    )
    model: str = Field(
        description="Name/version of the model used for prediction.",
        examples=["generic_v1"]
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about input parsing or model validity domain.",
        examples=[[]]
    )
