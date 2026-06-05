"""
FastAPI entry point for rec_o.

Exposes HTTP routes, API key authentication, rate limiting,
and MusicBrainz prediction / search endpoints.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.database import fetch_all, get_connection
from app.predictor import get_model_info, load_model, predict_artist, predict_playlist
from app.queries import (
    ALBUM_SEARCH_QUERY,
    ARTIST_SEARCH_QUERY,
    GENRE_SEARCH_QUERY,
)
from app.schemas import (
    AlbumSearchInput,
    AlbumSearchOutput,
    AlbumPredictInput,
    AlbumPredictOutput,
    AlbumPredictRow,
    ArtistSearchInput,
    ArtistSearchOutput,
    GenreSearchInput,
    GenreSearchOutput,
    PlaylistInput,
    PlaylistOutput,
)

# Load variables from .env (TOKEN_API_KEY, POSTGRES, etc.)
load_dotenv()

logger = logging.getLogger(__name__)

# Expected key in the X-API-Key header for protected routes
API_KEY = os.getenv("TOKEN_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the KNN artifact once at startup (local file or GCS), before serving traffic."""
    try:
        if load_model() is None:
            logger.warning(
                "Recommender model not loaded at startup — "
                "set MODEL_LOCAL_PATH or MODEL_BUCKET_NAME + MODEL_BLOB_NAME."
            )
        else:
            info = get_model_info()
            logger.info(
                "Recommender model loaded at startup (%s): %s",
                info.get("source"),
                info.get("path") or info.get("gcs_uri"),
            )
    except RuntimeError as exc:
        logger.error("Recommender model failed to load at startup: %s", exc)
    yield


def get_client_ip(request: Request) -> str:
    """
    Return the client IP address for rate limiting.

    Behind Cloud Run or a load balancer, uses X-Forwarded-For.
    Otherwise, uses the direct request IP.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return get_remote_address(request)


# Limit requests per IP (slowapi)
limiter = Limiter(key_func=get_client_ip)
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """
    Validate the X-API-Key header.

    Used via Depends() on protected routes.
    Returns the key if valid; raises HTTP 401 or 500 otherwise.
    """
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TOKEN_API_KEY is not configured",
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key


@app.get("/")
@limiter.limit("60/minute")
def read_root(request: Request):
    """Health / test route: no API key required."""
    return {"message": "Hello, World!"}


@app.get("/model")
@limiter.limit("60/minute")
def model_status(request: Request):
    """Which recommender artifact is loaded (local path or GCS URI). No API key required."""
    return get_model_info()


@app.post("/predict/artist", response_model=PlaylistOutput)
@limiter.limit("10/minute")
def predict(
    request: Request,
    input: PlaylistInput,
    _: str = Depends(verify_api_key),
):
    """
    Predict nearest artist IDs from one or more input artist IDs.

    JSON body: ArtistIds, TopN (see PlaylistInput).
    Requires the X-API-Key header.
    """
    try:
        artist_ids = predict_playlist(input.ArtistIds, input.TopN)

        with get_connection() as conn:
            artist_df = predict_artist(artist_ids, conn)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    artist_df = artist_df[["gid", "name", "genre", "urls"]]

    return PlaylistOutput(artists=artist_df.to_dict(orient="records"))


@app.post("/predict/album", response_model=AlbumPredictOutput)
def predict_album(
    input: AlbumPredictInput,
    _: str = Depends(verify_api_key),
):
    """
    Temporary mock endpoint for album recommendations.

    Will be connected to the real recommendation model later.
    """

    mock_albums = [
        AlbumPredictRow(
            gid="123e4567-e89b-12d3-a456-426614174000",
            title="Hybrid Theory",
            url=["https://example.com/hybrid-theory"],
            genres=["Nu metal", "Alternative rock"],
            length=12,
            tracks=[
                "Papercut",
                "One Step Closer",
                "Crawling"
            ]
        ),
        AlbumPredictRow(
            gid="123e4567-e89b-12d3-a456-426614174001",
            title="Meteora",
            url=["https://example.com/meteora"],
            genres=["Alternative rock"],
            length=13,
            tracks=[
                "Somewhere I Belong",
                "Numb",
                "Faint"
            ]
        )
    ]

    return AlbumPredictOutput(albums=mock_albums)



@app.post("/search/album", response_model=list[AlbumSearchOutput])
def search_album(
    input: AlbumSearchInput,
    _: str = Depends(verify_api_key),
):
    """
    Search release groups (albums) by partial title.

    Example: "abbey" matches titles containing "abbey".
    Returns at most 20 results.
    """
    rows = fetch_all(ALBUM_SEARCH_QUERY, (f"%{input.title}%",))

    return [
        AlbumSearchOutput(
            release_group_id=row[0],
            title=row[1],
            artist=row[2]
        )
        for row in rows
    ]


@app.post("/search/artist", response_model=list[ArtistSearchOutput])
def search_artist(
    input: ArtistSearchInput,
    _: str = Depends(verify_api_key),
):
    """
    Search artists by partial name.

    Example: "daft" matches names containing "daft".
    Returns at most 20 results.
    """
    rows = fetch_all(ARTIST_SEARCH_QUERY, (f"%{input.name}%",))

    return [
        ArtistSearchOutput(
            artist_id=row[0],
            name=row[1],
            disambiguation=row[2]
        )
        for row in rows
    ]


@app.post("/search/genre", response_model=list[GenreSearchOutput])
def search_genre(
    input: GenreSearchInput,
    _: str = Depends(verify_api_key),
):
    """
    Search genres by partial name.

    Example: "hip" matches genres containing "hip".
    Returns at most 20 results.
    """
    rows = fetch_all(
        GENRE_SEARCH_QUERY,
        (f"%{input.genre_name}%",)
    )

    return [
        GenreSearchOutput(
            genre_id=row[0],
            genre_name=row[1]
        )
        for row in rows
    ]
