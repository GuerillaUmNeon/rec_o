# rec_o

A small recommendation web app for exploring artist and album recommendations with two input modes:
- **Manual search** for artists or albums.
- **ListenBrainz profile** lookup for profile-based recommendations.

The interface supports both artist and album flows, light/dark mode, blacklist controls, optional parameters, and recommendation cards with MusicBrainz links when available.

## Backend requirement

This frontend requires the backend API project to run correctly. Start the backend separately before launching the frontend, and make sure the frontend `API_URL` points to that backend service.

Backend project: [rec_o](https://github.com/GuerillaUmNeon/rec_o)

## Features

- Artist and album recommendation tabs.
- Manual search with autocomplete dropdowns.
- ListenBrainz profile mode with collapsed optional parameters.
- Blacklist support for manual search.
- Adjustable recommendation count.
- Loading state on recommendation requests.
- MusicBrainz and official website links in results.

## Environment

The frontend requires `API_URL` and `TOKEN_API_KEY` environment variables to communicate with the backend API.

All environment variables are defined in the **backend `.env` file** at the project root. To make them available to the frontend, export them before running the development server:

```bash
# From project root
export $(grep -v '^#' .env | xargs)
cd front && npm run dev
```

For production builds, set the environment variables directly on the system or through your deployment platform.

### Variables

| Variable | Description |
|---|---|
| `API_URL` | Base URL for the backend API server (e.g., `http://127.0.0.1:8000`). |
| `TOKEN_API_KEY` | API key used for authenticated backend requests. Must match the backend's `TOKEN_API_KEY`. |

## Getting started

Clone the repository and move into the project folder:

```bash
git clone https://github.com/GuerillaUmNeon/rec_o-next.git
cd rec_o-next
```

Install dependencies:

```bash
npm install
```

Create `.env.local` from `.env.local.sample`, then start the development server:

```bash
npm run dev
```

Start the backend server separately so it is available at the configured `API_URL`.

## Input modes

### Manual search

Use the search box to find artists or albums, select one or more entries, optionally add blacklist items, then click **Get recommendations**.

### ListenBrainz profile

Enter a ListenBrainz username, expand **Optional parameters** if needed, then click **Get recommendations**.

Supported optional ListenBrainz parameters:
- `range`
- `min_listen`
- `blacklist`
- `blacklist_min`
- `max_results`
- `ntfy_url`
- `ntfy_topic`

Supported range values:
- `this_week`
- `this_month`
- `this_year`
- `week`
- `month`
- `year`
- `all_time`

## API routes used by the frontend

### Manual search
- `/api/search/artist`
- `/api/search/album`
- `/api/predict/artist`
- `/api/predict/album`

### ListenBrainz profile
- `/api/listenbrainz/artist`
- `/api/listenbrainz/album`

## Notes

- The ListenBrainz response format is expected to match the recommendation result structure used by manual search.
- If a ListenBrainz request returns 404, verify that the backend route exists and that the frontend path matches the mounted API prefix.
- If recommendations do not appear, check browser network requests and backend logs.