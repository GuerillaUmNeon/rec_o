from operator import gt
from pydantic import BaseModel, Field

class PlaylistInput(BaseModel):
    ArtistName: str = Field(..., min_length=1, max_length=255)
    Genre: str = Field(..., min_length=1, max_length=255)

class PlaylistOutput(BaseModel):
    ArtistName: str = Field(..., min_length=1, max_length=255)
    Genre: str = Field(..., min_length=1, max_length=255)