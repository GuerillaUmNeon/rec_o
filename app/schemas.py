from pydantic import BaseModel, Field


class PlaylistInput(BaseModel):
    ArtistIds: list[int] = Field(..., min_length=1)
    TopN: int = Field(default=5, ge=1, le=50)


class PlaylistOutput(BaseModel):
    ArtistIds: list[int] = Field(..., min_length=1)


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
    disambiguation: str | None = None
