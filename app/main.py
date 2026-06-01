from fastapi import FastAPI
from app.schemas import PlaylistInput, PlaylistOutput
from app.predictor import predict_playlist

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict", response_model=PlaylistOutput)
def predict(input: PlaylistInput):
    artist_name, artist_genre = predict_playlist(input.ArtistName, input.Genre)
    return PlaylistOutput(ArtistName=artist_name, Genre=artist_genre)