from pydantic import BaseModel
from typing import Optional, Literal

class ForecastRequest(BaseModel):
    date_column: str
    target_column: str
    filter_column: Optional[str] = None
    filter_value: Optional[str] = None
    frequency: Literal['daily', 'weekly', 'monthly']
    category: Literal['actual', 'forecast', 'rolling forecast', 'prediction']  # Include 'prediction' here
    train_upto: Optional[str] = None  # Specify up to which date to use for training
    predict_for: Optional[int] = 30   # Duration to predict (default 30 periods)
    user_role: Literal['public', 'private']
