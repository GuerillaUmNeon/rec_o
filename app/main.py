import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.predictor import predict_playlist
from app.schemas import PlaylistInput, PlaylistOutput

load_dotenv()

API_KEY = os.getenv("TOKEN_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI()


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
def read_root():
    return {"message": "Hello, World!"}


@app.post("/predict", response_model=PlaylistOutput)
def predict(
    input: PlaylistInput,
    _: str = Depends(verify_api_key),
):
    artist_name, artist_genre = predict_playlist(input.ArtistName, input.Genre)
    return PlaylistOutput(ArtistName=artist_name, Genre=artist_genre)