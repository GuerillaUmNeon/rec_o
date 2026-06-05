"""PostgreSQL enrichment for release group recommendation responses."""

import pandas as pd


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
            columns=["id", "gid", "title", "genres", "url", "length", "tracks"]
        )

    placeholders = ",".join(["%s"] * len(release_group_ids))

    query = f"""
        WITH canonical_release AS (
            SELECT DISTINCT ON (release_group)
                release_group,
                id AS release_id
            FROM musicbrainz.release
            WHERE release_group IN ({placeholders})
            ORDER BY release_group, id
        ),
        track_rows AS (
            SELECT
                cr.release_group AS id,
                rec.name AS track_name,
                m.position AS medium_position,
                t.position AS track_position
            FROM canonical_release cr
            JOIN musicbrainz.medium m
                ON m.release = cr.release_id
            JOIN musicbrainz.track t
                ON t.medium = m.id
            JOIN musicbrainz.recording rec
                ON rec.id = t.recording
        )
        SELECT
            rg.id AS id,
            rg.gid AS gid,
            rg.name AS title,
            genre.name AS genre,
            url.url AS url,
            tr.track_name,
            tr.medium_position,
            tr.track_position
        FROM musicbrainz.release_group rg
        LEFT JOIN musicbrainz.release_group_tag rgt
            ON rgt.release_group = rg.id
           AND rgt.count > 0
        LEFT JOIN musicbrainz.tag tag
            ON tag.id = rgt.tag
        LEFT JOIN musicbrainz.genre genre
            ON tag.name = genre.name
        LEFT JOIN musicbrainz.l_release_group_url lrgu
            ON lrgu.entity0 = rg.id
        LEFT JOIN musicbrainz.url url
            ON url.id = lrgu.entity1
        LEFT JOIN track_rows tr
            ON tr.id = rg.id
        WHERE rg.id IN ({placeholders})
    """

    params = release_group_ids + release_group_ids
    result = pd.read_sql_query(query, conn, params=params)

    if result.empty:
        return pd.DataFrame(
            columns=["id", "gid", "title", "genres", "url", "length", "tracks"]
        )

    def agg_genres(series):
        return [g.capitalize() for g in series.dropna().unique().tolist()]

    def agg_urls(series):
        return series.dropna().unique().tolist()

    def agg_tracks(group):
        tracks = (
            group[["track_name", "medium_position", "track_position"]]
            .dropna(subset=["track_name"])
            .drop_duplicates()
            .sort_values(["medium_position", "track_position"])
        )
        return tracks["track_name"].tolist()

    grouped = (
        result.groupby(["id", "gid", "title"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "genres": agg_genres(g["genre"]),
                "url": agg_urls(g["url"]),
                "tracks": agg_tracks(g),
            })
        )
        .reset_index(drop=True)
    )
    grouped["length"] = grouped["tracks"].apply(len)

    order = {rg_id: idx for idx, rg_id in enumerate(release_group_ids)}
    grouped["_order"] = grouped["id"].map(order)
    grouped = grouped.sort_values("_order").drop(columns="_order")

    return grouped[["id", "gid", "title", "genres", "url", "length", "tracks"]]
