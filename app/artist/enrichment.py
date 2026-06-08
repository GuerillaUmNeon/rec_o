"""PostgreSQL enrichment for artist recommendation responses."""
import os
import pandas as pd
import requests
from fastapi import HTTPException

from app.schemas import PlaylistOutput

LISTENBRAINZ=f"{os.getenv("LISTENBRAINZ_URL")}/1/stats/user"

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

def get_top_lb(username, range, min_listen, type, token):
    url = f"{LISTENBRAINZ}/{username}/{type}"
    params = {
        "range": range,
    }
    auth_header = {
        "Authorization": f"Token {token}"
    }

    try:
        response = requests.get(url, params=params, headers=auth_header, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"ListenBrainz API error: {response.text}"
        ) from e
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"ListenBrainz request failed: {str(e)}"
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=502,
            detail="ListenBrainz returned invalid JSON"
        ) from e

    results = data.get("payload", {}).get(type)
    if results is None:
        raise HTTPException(
            status_code=502,
            detail=f"ListenBrainz response missing payload.{type}"
        )

    filtered = [result for result in results if result["listen_count"] > min_listen]

    if not filtered:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for user '{username}' with listen_count > {min_listen}"
        )

    return filtered

def send_ntfy_artist_notification(input, artist_output: PlaylistOutput):
    if not input.ntfy_url or not input.ntfy_topic:
        return

    lines = ["# Recommended artists", ""]

    for artist in artist_output.artists:
        genre_text = ", ".join(artist.genre) if artist.genre else ""

        line = f"- **{artist.name}**"
        if genre_text:
            line += f" ({genre_text})"

        links = [f"[ListenBrainz](https://listenbrainz.org/artist/{artist.gid})"]

        official_site = next(
            (str(url.url) for url in artist.urls if getattr(url, "type", None) == 183),
            None
        )
        if official_site:
            links.append(f"[Official site]({official_site})")

        line += " — " + " | ".join(links)
        lines.append(line)

    message = "\n".join(lines)

    publish_url = f"{input.ntfy_url.rstrip('/')}/{input.ntfy_topic}"
    response = requests.post(
        publish_url,
        data=message.encode("utf-8"),
        headers={
            "Title": "rec_o recommendation",
            "Markdown": "yes",
            "Tags": "musical_note"
        },
        timeout=10,
    )
    response.raise_for_status()