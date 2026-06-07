"""PostgreSQL enrichment for artist recommendation responses."""
import os
import pandas as pd
import requests


def enrich_artists_from_db(artist_ids: list[int], conn) -> pd.DataFrame:
    """
    Return one row per artist with genres aggregated as a list.

    Parameters
    ----------
    artist_ids : list[int]
        MusicBrainz artist.id values.
    conn :
        DB connection compatible with pandas.read_sql_query.
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
            url.url AS urls,
            link_type.id AS link_type_id
        FROM artist
        LEFT JOIN artist_tag
            ON artist_tag.artist = artist.id
           AND artist_tag.count >= 1
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
    """

    result = pd.read_sql_query(query, conn, params=artist_ids)

    if result.empty:
        return pd.DataFrame(columns=["id", "gid", "name", "genre", "urls"])

    def agg_genres(series):
        return [g.capitalize() for g in series.dropna().unique().tolist()]

    def agg_urls(group):
        pairs = (
            group[["urls", "link_type_id"]]
            .dropna(subset=["urls", "link_type_id"])
            .drop_duplicates()
            .to_dict("records")
        )
        return [
            {"url": item["urls"], "type": int(item["link_type_id"])}
            for item in pairs
        ]

    grouped = (
        result.groupby(["id", "gid", "name"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "genre": agg_genres(g["genre"]),
                "urls": agg_urls(g),
            })
        )
        .reset_index(drop=True)
    )

    return grouped

def get_top_artists(username, range, min_listen):
    url = f"{os.getenv("LISTENBRAINZ_URL")}/1/stats/user/{username}/artists"
    params = {
        'range': range,
        # 'count': count
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    artists = data['payload']['artists']

    return [artist for artist in artists if artist['listen_count'] > min_listen]