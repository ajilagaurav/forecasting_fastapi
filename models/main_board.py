from pydantic import BaseModel

class MainBoard(BaseModel):
    id: int
    name: str
    description: str