from pydantic import BaseModel

class LandmarkResponse(BaseModel):
    best_prob: float = -1.0
    predicted_id: int = -1
    predicted_class: str = ""
    lable_name: str = ""
    landmark_description: str = """"""
