"""
SQL queries for the API.

Use %s placeholders for parameters passed to fetch_all() (psycopg).
"""

ALBUM_SEARCH_QUERY = """
SELECT id AS release_group_id, name AS title
FROM musicbrainz.release_group
WHERE name ILIKE %s
LIMIT 20;
"""

ARTIST_SEARCH_QUERY = """
SELECT id, name, comment AS disambiguation
FROM musicbrainz.artist
WHERE name ILIKE %s
LIMIT 20;
"""
