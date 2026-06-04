from pydantic import BaseModel, Field
from uuid import UUID

class PlaylistInput(BaseModel):
    ArtistIds: list[int] = Field(..., min_length=1)
    TopN: int = Field(default=5, ge=1, le=50)

class ArtistUrl(BaseModel):
    url: str
    type: int

class ArtistRow(BaseModel):
    gid: UUID
    name: str
    genre: list[str]
    urls: list[ArtistUrl]

class PlaylistOutput(BaseModel):
    artists: list[ArtistRow]

class AlbumSearchInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class AlbumSearchOutput(BaseModel):
    release_group_id: int
    title: str


class ArtistSearchInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class ArtistSearchOutput(BaseModel):
    artist_id: int
    name: str
    disambiguation: str | None = None


class GenreSearchInput(BaseModel):
    genre_name: str = Field(..., min_length=1, max_length=255)


class GenreSearchOutput(BaseModel):
    genre_id: int
    genre_name: str
