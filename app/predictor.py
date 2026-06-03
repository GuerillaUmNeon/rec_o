from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
model = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None

# TODO: remove predict_playlist when predict_artist is implemented
def predict_playlist(artist_name: str, artist_genre: str) -> tuple[str, str]:
    data_dict = {
        'artist_name': [artist_name],
        'artist_genre': [artist_genre],
    }
    data_df = pd.DataFrame(data_dict)
    #pred_result = model.predict(data_df)[0]
    pred_result = ('Celine Dion', 'Rock')
    return pred_result

def predict_artist(artist_ids, conn):
    """
    Return one row per artist with genres aggregated as a list.

    Parameters
    ----------
    artist_ids : list[int]
        List of MusicBrainz artist.id values.
    conn :
        DB connection compatible with pandas.read_sql_query.

    Returns
    -------
    pd.DataFrame
        Columns: id, gid, name, genre
        genre is a list of unique non-null genres for each artist.
    """
    if not artist_ids:
        return pd.DataFrame(columns=["id", "gid", "name", "genre"])

    placeholders = ",".join(["%s"] * len(artist_ids))

    query = f"""
            SELECT
                artist.id AS id,
                artist.gid AS gid,
                artist.name AS name,
                genre.name AS genre
            FROM artist
            LEFT JOIN artist_tag
                ON artist_tag.artist = artist.id
            LEFT JOIN tag
                ON artist_tag.tag = tag.id
            LEFT JOIN genre
                ON tag.name = genre.name
            WHERE artist.id IN ({placeholders})
              AND artist_tag.count > 0
            """

    params = list(artist_ids)
    result = pd.read_sql_query(query, conn, params=params)

    if result.empty:
        return pd.DataFrame(columns=["id", "gid", "name", "genre"])

    grouped = (
        result.groupby(["id", "gid", "name"], as_index=False)
              .agg({
                    "genre": lambda x: [g.capitalize() for g in x.dropna().unique().tolist()]
              })
    )

    return grouped