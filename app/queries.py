"""
SQL queries for the API.

Use %s placeholders for parameters passed to fetch_all() (psycopg).
"""

ALBUM_SEARCH_QUERY = """
SELECT
    rg.id AS release_group_id,
    rg.name AS title,
    a.name AS artist
FROM musicbrainz.release_group rg
JOIN musicbrainz.artist_credit ac
    ON rg.artist_credit = ac.id
JOIN musicbrainz.artist_credit_name acn
    ON ac.id = acn.artist_credit
JOIN musicbrainz.artist a
    ON acn.artist = a.id
WHERE rg.name ILIKE %s
LIMIT 20;
"""

ARTIST_SEARCH_QUERY = """
SELECT
    id,
    name,
    comment AS disambiguation
FROM musicbrainz.artist
WHERE name ILIKE %s
LIMIT 20;
"""

GENRE_SEARCH_QUERY = """
SELECT
    id,
    name
FROM musicbrainz.genre
WHERE name ILIKE %s
LIMIT 20;
"""
