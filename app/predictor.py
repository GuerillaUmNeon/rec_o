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
        return pd.DataFrame(columns=["id", "gid", "name", "genre", "url"])

    placeholders = ",".join(["%s"] * len(artist_ids))

    query = f"""
        SELECT
            artist.id AS id,
            artist.gid AS gid,
            artist.name AS name,
            genre.name AS genre,
            url.url AS url,
            link_type.id AS link_type_id
        FROM artist
        LEFT JOIN artist_tag
            ON artist_tag.artist = artist.id
        LEFT JOIN tag
            ON artist_tag.tag = tag.id
        LEFT JOIN genre
            ON tag.name = genre.name
        LEFT JOIN l_artist_url
            ON l_artist_url.entity0 = artist.id
        LEFT JOIN url
            ON url.id = l_artist_url.entity1
        LEFT JOIN link
            ON link.id = l_artist_url.link
        LEFT JOIN link_type
            ON link_type.id = link.link_type
        WHERE artist.id IN ({placeholders})
          AND artist_tag.count > 0
    """

    result = pd.read_sql_query(query, conn, params=artist_ids)

    if result.empty:
        return pd.DataFrame(columns=["id", "gid", "name", "genre", "url"])

    def agg_genres(series):
        return [g.capitalize() for g in series.dropna().unique().tolist()]

    def agg_urls(group):
        pairs = (
            group[["url", "link_type_id"]]
            .dropna(subset=["url", "link_type_id"])
            .drop_duplicates()
            .to_dict("records")
        )
        return [
            {"url": item["url"], "type": int(item["link_type_id"])}
            for item in pairs
        ]

    grouped = (
        result.groupby(["id", "gid", "name"], as_index=False)
              .apply(lambda g: pd.Series({
                  "genre": agg_genres(g["genre"]),
                  "url": agg_urls(g)
              }))
              .reset_index(drop=True)
    )

    return grouped