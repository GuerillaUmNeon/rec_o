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

from app.artist.enrichment import get_top_artists
from app.database import fetch_all, get_connection
from app.artist import enrich_artists_from_db, recommend_artist_ids
from app.release_group import enrich_release_groups_from_db, recommend_release_group_ids
from app.models import get_models_info, load_models
from app.queries import (
    ALBUM_SEARCH_QUERY,
    ARTIST_SEARCH_QUERY,
    GENRE_SEARCH_QUERY, ARTIST_GID_SEARCH_QUERY,
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
    PlaylistOutput, ListenbrainzInput,
)
import pandas as pd

# Load variables from .env (TOKEN_API_KEY, POSTGRES, etc.)
load_dotenv()

logger = logging.getLogger(__name__)

# Expected key in the X-API-Key header for protected routes
API_KEY = os.getenv("TOKEN_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load recommender artifacts at startup (artist first; others later)."""
    try:
        load_models()
        models_info = get_models_info()
        for model_name in ("artist", "release_group"):
            info = models_info.get(model_name, {})
            if not info.get("loaded"):
                logger.warning(
                    "%s model not loaded at startup — check %s_MODEL_LOCAL_PATH "
                    "or MODEL_BUCKET_NAME + %s_MODEL_BLOB_NAME.",
                    model_name.replace("_", " ").title(),
                    model_name.upper(),
                    model_name.upper(),
                )
            else:
                logger.info(
                    "%s model loaded at startup (%s): %s",
                    model_name.replace("_", " ").title(),
                    info.get("source"),
                    info.get("path") or info.get("gcs_uri"),
                )
    except Exception as exc:
        logger.error("Model loading failed at startup: %s", exc, exc_info=True)
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
    """Load status per model type (artist, release_group, genre). No API key required."""
    return get_models_info()


@app.post("/predict/artist", response_model=PlaylistOutput)
@limiter.limit("10/minute")
def predict(
    request: Request,
    input: PlaylistInput,
    _: str = Depends(verify_api_key),
):
    """
    Predict nearest artist IDs from one or more input artist IDs.

    JSON body: ArtistIds, TopN, optional BlacklistArtistIds (see PlaylistInput).
    Requires the X-API-Key header.
    """
    try:
        artist_ids = recommend_artist_ids(
            input.ArtistIds,
            input.TopN,
            blacklist_artist_ids=input.BlacklistArtistIds,
        )

        with get_connection() as conn:
            artist_df = enrich_artists_from_db(artist_ids, conn)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    artist_df = artist_df[["gid", "name", "genre", "urls"]]

    return PlaylistOutput(artists=artist_df.to_dict(orient="records"))


@app.post("/predict/album", response_model=AlbumPredictOutput)
@limiter.limit("10/minute")
def predict_album(
    request: Request,
    input: AlbumPredictInput,
    _: str = Depends(verify_api_key),
):
    """
    Predict nearest release group IDs from one or more seed album IDs.

    JSON body: release_group_id, response_length, optional genre_id and blacklist.
    Requires the X-API-Key header.
    """
    try:
        release_group_ids = recommend_release_group_ids(
            input.release_group_id,
            input.response_length,
            blacklist=input.blacklist_release_group_id,
            genre_ids=input.genre_id,
        )

        with get_connection() as conn:
            albums_df = enrich_release_groups_from_db(release_group_ids, conn)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    albums = [
        AlbumPredictRow(
            artist=row["artist"],
            gid=row["gid"],
            title=row["title"],
            genres=row["genres"],
            length=None if pd.isna(row["length"]) else int(row["length"]),
            tracks=None if pd.isna(row["tracks"]) else int(row["tracks"])
        )
        for row in albums_df.to_dict(orient="records")
    ]

    return AlbumPredictOutput(albums=albums)



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

@app.post("/listenbrainz/artist", response_model=PlaylistOutput)
def lb_artist_predict(
    request: Request,
    input: ListenbrainzInput,
    _: str = Depends(verify_api_key),
):
    artists = get_top_artists(input.username, input.range, input.min_listen)
    artist_mbids = [artist["artist_mbid"] for artist in artists]

    rows = fetch_all(ARTIST_GID_SEARCH_QUERY, (artist_mbids,))
    artist_ids = [row[0] for row in rows]

    blacklist_ids = []
    if input.blacklist != "None":
        blacklist = get_top_artists(
            input.username,
            input.blacklist,
            input.blacklist_min
        )
        blacklist_mbids = [artist["artist_mbid"] for artist in blacklist]
        blacklist_rows = fetch_all(ARTIST_GID_SEARCH_QUERY, (blacklist_mbids,))
        blacklist_ids = [row[0] for row in blacklist_rows]

    try:
        predict_artist_ids = recommend_artist_ids(
            artist_ids,
            input.max_results,
            blacklist_artist_ids=blacklist_ids,
        )

        with get_connection() as conn:
            artist_df = enrich_artists_from_db(predict_artist_ids, conn)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    artist_df = artist_df[["gid", "name", "genre", "urls"]]

    return {"artists": artist_df.to_dict(orient="records")}