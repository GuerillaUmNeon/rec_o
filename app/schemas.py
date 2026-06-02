from operator import gt
from pydantic import BaseModel, Field


class PlaylistInput(BaseModel):
    ArtistName: str = Field(..., min_length=1, max_length=255)
    Genre: str = Field(..., min_length=1, max_length=255)


class PlaylistOutput(BaseModel):
    ArtistName: str = Field(..., min_length=1, max_length=255)
    Genre: str = Field(..., min_length=1, max_length=255)


class AlbumSearchInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class AlbumSearchOutput(BaseModel):
    release_group_id: int
    title: str


class ArtistSearchInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class ArtistSearchOutput(BaseModel):
    id: int
    name: str
