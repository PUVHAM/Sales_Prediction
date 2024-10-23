from pydantic import BaseModel
from typing import Any, Dict

class PredictRequest(BaseModel):
    model_type: str
    input_data: Dict[str, Any]