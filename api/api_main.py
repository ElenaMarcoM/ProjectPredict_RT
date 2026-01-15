import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import PredictRequest, PredictResponse
from api.inference import InferenceService

app = FastAPI(title="Prediction_RT API", version="0.1.0")

# paths
current_file = os.path.abspath(__file__)
api_dir = os.path.dirname(current_file)
project_root = os.path.dirname(api_dir)

# For docker a different version of the model was used that does not use lambda was employed
# Docker gives an error when loading the original model "model.h5"
model_path = os.path.join(project_root, "resources", "model_baby.h5")
scaler_path = os.path.join(project_root, "resources", "scaler.pkl")

# Uncomment to load model for LOCAL API
# Use the following command in the project console:  python -m uvicorn api.api_main:app --port 8000
#model_path = os.path.join(project_root, "resources", "model.h5")

service = InferenceService(
    model_path=model_path,
    scaler_path=scaler_path,
    model_name="generic_v1"
)

# Loads model and scaler as the server starts
@app.on_event("startup")
def startup_event():
    try:
        service.load()
        print("[startup] Model & scaler loaded OK")
    except Exception as e:
        print(f"[startup] Could not load model/scaler: {e}")

# Turns error 422 from Pydantic into error 400 "SMILES/INCHI not found"
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    msg = str(exc)
    if "SMILES/INCHI not found" in msg:
        return JSONResponse(status_code=400, content={"detail": "SMILES/INCHI not found"})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# Returns to the user the state of the API. If the model and scaler are correctly loaded
@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description="Check whether the API and the ML model are correctly loaded."
)
def health():
    return {
        "status": "ok",
        "model_loaded": service.is_loaded(),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "model_version": "generic_v1",
    }

# Obtains the fp of the given compound and applies the model to obtain a prediction of its rt
@app.post(
"/predict",
    response_model=PredictResponse,
    tags=["Predictor"],
    summary="Predict retention time (RT) of a given compound.",
    description=(
        "**Provide ONE of the following inputs:**\n\n"
        "- `smiles`: SMILES representation of the compound\n"
        "- `inchi`: InChI representation of the compound\n\n"
        "This API predicts chromatographic retention time (RT) in **seconds**.\n\n"
        "⚠️ **Model validity domain**: trained with RT ≥ 400 s. "
    ),
    responses={
        400: {"description": "Bad Request (SMILES/INCHI not found or invalid)"},
        500: {"description": "Internal Server Error (model not loaded)"},
    },
          )
async def predict(req: PredictRequest, request: Request):
    # To check locally that the input given is the same as the one processed
    #raw = await request.body()
    #print("=== /predict RAW BODY ===", raw)
    #print("=== /predict PARSED ===", req.model_dump())

    if not req.smiles and not req.inchi:
        raise HTTPException(status_code=400, detail="SMILES/INCHI not found")
    if not service.is_loaded():
        raise HTTPException(status_code=500, detail="Model not loaded")

    out = service.predict(smiles=req.smiles, inchi=req.inchi)
    return PredictResponse(
        input_type=out.input_type,
        input=out.input_value,
        rt_predicted=out.rt_pred_seconds,
        units="s",
        model=out.model_name,
        warnings=out.warnings,
    )
