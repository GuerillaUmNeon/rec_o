"""PostgreSQL enrichment for release group recommendation responses."""

import pandas as pd
import requests
from app.schemas import AlbumPredictOutput
import random

RELEASE_GROUP_RESPONSE_INFOS = """
            WITH release_groups_filter AS (SELECT unnest(%s::int[]) AS release_group_id),
                 ranked_releases AS (SELECT r.release_group, \
                                            r.id  AS release_id, \
                                            row_number() OVER ( \
                                                PARTITION BY r.release_group \
                                                ORDER BY \
                                                    CASE \
                                                        WHEN lower(r.name) LIKE '%%super deluxe%%' THEN 1 \
                                                        WHEN lower(r.name) LIKE '%%deluxe%%' THEN 1 \
                                                        WHEN lower(r.name) LIKE '%%expanded%%' THEN 1 \
                                                        WHEN lower(r.name) LIKE '%%anniversary%%' THEN 1 \
                                                        WHEN lower(r.name) LIKE '%%bonus%%' THEN 1 \
                                                        WHEN lower(r.name) LIKE '%%collector%%' THEN 1 \
                                                        ELSE 0 \
                                                        END, \
                                                    COALESCE(rc.date_year, ruc.date_year) NULLS LAST, \
                                                    COALESCE(rc.date_month, ruc.date_month) NULLS LAST, \
                                                    COALESCE(rc.date_day, ruc.date_day) NULLS LAST, \
                                                    r.id \
                                                ) AS rn \
                                     FROM release_groups_filter rgf \
                                              JOIN release r \
                                                   ON r.release_group = rgf.release_group_id \
                                              LEFT JOIN release_country rc \
                                                        ON rc.release = r.id \
                                              LEFT JOIN release_unknown_country ruc \
                                                        ON ruc.release = r.id \
                                                            AND rc.release IS NULL),
                 first_release AS (SELECT release_group, \
                                          release_id \
                                   FROM ranked_releases \
                                   WHERE rn = 1),
                 release_stats AS (SELECT r.release_group, \
                                          r.id          AS release_id, \
                                          COUNT(t.id)   AS tracks, \
                                          SUM(t.length) AS length \
                                   FROM release r \
                                            JOIN medium m \
                                                 ON m.release = r.id \
                                            JOIN track t \
                                                 ON t.medium = m.id \
                                            JOIN release_groups_filter rgf \
                                                 ON rgf.release_group_id = r.release_group \
                                   GROUP BY r.release_group, r.id),
                 original_release_stats AS (SELECT fr.release_group, \
                                                   rs.tracks AS original_tracks \
                                            FROM first_release fr \
                                                     JOIN release_stats rs \
                                                          ON rs.release_id = fr.release_id),
                 release_track_stats AS (SELECT ors.release_group, \
                                                ors.original_tracks AS tracks, \
                                                ( \
                                                            ARRAY_AGG(rs.length ORDER BY rs.release_id) \
                                                            FILTER (WHERE rs.length IS NOT NULL) \
                                                    )[1]            AS length \
                                         FROM original_release_stats ors \
                                                  JOIN release_stats rs \
                                                       ON rs.release_group = ors.release_group \
                                                           AND rs.tracks = ors.original_tracks \
                                         GROUP BY ors.release_group, ors.original_tracks),
                 artist_links AS (SELECT rgf.release_group_id                              AS release_group, \
                                         string_agg(DISTINCT a.name, ', ' ORDER BY a.name) AS artist, \
                                         array_agg(DISTINCT a.id)                          AS artist_ids \
                                  FROM release_groups_filter rgf \
                                           JOIN release r \
                                                ON r.release_group = rgf.release_group_id \
                                           JOIN artist_credit ac \
                                                ON ac.id = r.artist_credit \
                                           JOIN artist_credit_name acn \
                                                ON acn.artist_credit = ac.id \
                                           JOIN artist a \
                                                ON a.id = acn.artist \
                                  GROUP BY rgf.release_group_id),
                 artist_genre_links AS (SELECT DISTINCT al.release_group, \
                                                        g.id   AS genre_id, \
                                                        g.name AS genre_name \
                                        FROM artist_links al \
                                                 JOIN artist_tag at \
                                                      ON at.artist = ANY (al.artist_ids) \
                                                 JOIN tag t \
                                                      ON t.id = at.tag \
                                                 JOIN genre g \
                                                      ON g.name = t.name),
                 genre_counts AS (SELECT release_group, \
                                         genre_id, \
                                         genre_name, \
                                         COUNT(*) AS genre_count \
                                  FROM artist_genre_links \
                                  GROUP BY release_group, genre_id, genre_name),
                 genre_buckets AS (SELECT release_group, \
                                          array_agg(DISTINCT genre_name ORDER BY genre_name) AS genres \
                                   FROM genre_counts \
                                   WHERE genre_count > 0 \
                                   GROUP BY release_group)
            SELECT rg.id                                 AS id, \
                   rg.gid                                AS gid, \
                   rg.name                               AS title, \
                   al.artist, \
                   COALESCE(gb.genres, ARRAY []::text[]) AS genres, \
                   rts.tracks, \
                   rts.length
            FROM first_release fr
                     JOIN release_group rg
                          ON rg.id = fr.release_group
                     LEFT JOIN release_track_stats rts
                               ON rts.release_group = fr.release_group
                     LEFT JOIN genre_buckets gb
                               ON gb.release_group = fr.release_group
                     JOIN artist_links al
                          ON al.release_group = fr.release_group; \
            """

def enrich_release_groups_from_db(
    release_group_ids: list[int],
    conn,
) -> pd.DataFrame:
    """
    Return one row per release group with genres, URLs, and track names.

    Parameters
    ----------
    release_group_ids : list[int]
        MusicBrainz release_group.id values (order preserved in output).
    conn :
        DB connection compatible with pandas.read_sql_query.
    """
    if not release_group_ids:
        return pd.DataFrame(
            columns=["id", "gid", "title", "artist", "genres", "length", "tracks"]
        )

    # params = release_group_ids + release_group_ids
    result = pd.read_sql_query(RELEASE_GROUP_RESPONSE_INFOS, conn, params=(release_group_ids,))

    if result.empty:
        return pd.DataFrame(
            columns=["id", "gid", "title", "artist", "genres", "length", "tracks"]
        )

    order = {rg_id: idx for idx, rg_id in enumerate(release_group_ids)}
    result["_order"] = result["id"].map(order)
    result = result.sort_values("_order").drop(columns="_order")

    return result[["id", "gid", "title", "artist", "genres", "length", "tracks"]]

def format_tracks(tracks: int | None) -> str:
    if tracks is None:
        return ""
    return f"{tracks} track" if tracks == 1 else f"{tracks} tracks"

def format_duration_ms(length_ms: int | None) -> str:
    if length_ms is None:
        return ""

    total_minutes = length_ms // 1000 // 60
    hours, minutes = divmod(total_minutes, 60)

    if hours >= 1:
        if minutes > 0:
            return f"{hours}h {minutes} min"
        return f"{hours}h"

    return f"{total_minutes} min"

def send_ntfy_album_notification(album_output: AlbumPredictOutput):
    import os
    ntfy_url = os.getenv('NTFY_URL')
    ntfy_topic = os.getenv('NTFY_TOPIC')
    
    if not ntfy_url or not ntfy_topic:
        return

    blocks = ["# Recommended release groups"]

    for album in album_output.albums:
        links = [f"[ListenBrainz](https://listenbrainz.org/release-group/{album.gid})"]

        if album.url:
            links.append(f"[Link]({album.url[0]})")

        random_genres = (
            ", ".join(random.sample(album.genres, min(5, len(album.genres))))
            if album.genres else ""
        )

        details = " - ".join(
            part for part in [
                format_tracks(album.tracks),
                format_duration_ms(album.length)
            ] if part
        )

        block = [
            f"**{album.title}** - {album.artist}",
            random_genres,
            details,
            " | ".join(links)
        ]

        blocks.append("\n".join(line for line in block if line))

    message = "\n\n".join(blocks)

    publish_url = f"{ntfy_url.rstrip('/')}/{ntfy_topic}"
    response = requests.post(
        publish_url,
        data=message.encode("utf-8"),
        headers={
            "Title": "rec_o release groups",
            "Markdown": "yes",
            "Tags": "cd"
        },
        timeout=10,
    )
    response.raise_for_status()