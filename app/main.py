"""
FastAPI entry point for rec_o.

Exposes HTTP routes, API key authentication, rate limiting,
and MusicBrainz prediction / search endpoints.
"""

import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.database import fetch_all
from app.predictor import predict_playlist
from app.queries import (
    ALBUM_SEARCH_QUERY,
    ARTIST_SEARCH_QUERY,
    GENRE_SEARCH_QUERY,
)
from app.schemas import (
    AlbumSearchInput,
    AlbumSearchOutput,
    ArtistSearchInput,
    ArtistSearchOutput,
    GenreSearchInput,
    GenreSearchOutput,
    PlaylistInput,
    PlaylistOutput,
)

# Load variables from .env (TOKEN_API_KEY, POSTGRES, etc.)
load_dotenv()

# Expected key in the X-API-Key header for protected routes
API_KEY = os.getenv("TOKEN_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


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
app = FastAPI()
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


@app.post("/predict", response_model=PlaylistOutput)
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
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return PlaylistOutput(ArtistIds=artist_ids)


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
            title=row[1]
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
