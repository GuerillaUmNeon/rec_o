from pydantic import BaseModel, Field
from uuid import UUID


class PlaylistInput(BaseModel):
    ArtistIds: list[int] = Field(..., min_length=1)
    TopN: int = Field(default=5, ge=1, le=50)
    BlacklistArtistIds: list[int] = Field(default_factory=list)


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
    title: str = Field(..., min_length=1, max_length=255, description="Release group title"),
    artist: str | None = Field(None, description="Artist name")


class AlbumSearchOutput(BaseModel):
    release_group_id: int
    title: str
    artist: str
    disambiguation: str | None = None


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


class AlbumPredictInput(BaseModel):
    release_group_id: list[int] = Field(..., min_length=1)
    genre_id: list[int] | None = None
    response_length: int = Field(default=10, ge=1, le=50)
    blacklist_release_group_id: list[int] | None = None

class AlbumPredictRow(BaseModel):
    gid: UUID
    url: list[str] = Field(default_factory=list)
    title: str
    genres: list[str]
    length: int | None = None
    tracks: int | None = None
    artist: str

class AlbumPredictOutput(BaseModel):
    albums: list[AlbumPredictRow]

class ListenbrainzInput(BaseModel):
    username: str
    range: str = Field(default="week")
    min_listen: int = Field(default=5, ge=0)
    blacklist: str | None = None
    blacklist_min: int = Field(default=5, ge=0)
    max_results: int = Field(default=10, ge=1, le=100)
    ntfy_url: str | None = None
    ntfy_topic: str | None = None