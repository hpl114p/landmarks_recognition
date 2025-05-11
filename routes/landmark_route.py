import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import File, UploadFile
from fastapi import APIRouter
from schemas.landmark_schema import LandmarkResponse
from config.landmark_cfg import ModelConfig
from models.landmark_predictor import Predictor

router = APIRouter()
predictor = Predictor(
    model_name=ModelConfig.MODEL_NAME, 
    model_weight=ModelConfig.MODEL_WEIGHT, 
    api_key=ModelConfig.API_KEY,
    device=ModelConfig.DEVICE
)

@router.post("/predict")
async def predict(file_upload: UploadFile = File(...)):
    #print(file_upload)
    response = await predictor.predict(file_upload.file)
    
    return LandmarkResponse(**response)

