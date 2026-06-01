from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
model = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None

def predict_playlist(artist_name: str, artist_genre: str) -> tuple[str, str]:
    data_dict = {
        'artist_name': [artist_name],
        'artist_genre': [artist_genre],
    }
    data_df = pd.DataFrame(data_dict)
    #pred_result = model.predict(data_df)[0]
    pred_result = ('Celine Dion', 'Rock')
    return pred_result