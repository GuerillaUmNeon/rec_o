import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.predictor import predict_playlist
from app.schemas import PlaylistInput, PlaylistOutput

load_dotenv()

API_KEY = os.getenv("TOKEN_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return get_remote_address(request)


limiter = Limiter(key_func=get_client_ip)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
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
    return {"message": "Hello, World!"}


@app.post("/predict", response_model=PlaylistOutput)
@limiter.limit("10/minute")
def predict(
    request: Request,
    input: PlaylistInput,
    _: str = Depends(verify_api_key),
):
    artist_name, artist_genre = predict_playlist(input.ArtistName, input.Genre)
    return PlaylistOutput(ArtistName=artist_name, Genre=artist_genre)