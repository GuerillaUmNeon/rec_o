"use client";
import { Analytics } from "@vercel/analytics/next"
import { Oswald, Inter } from "next/font/google";
import { useEffect, useMemo, useState } from "react";

const MAX_REQUESTS = 5;
const WINDOW_SECONDS = 60;
const SEARCH_DEBOUNCE_MS = 350;
const NUMBER_INPUT_MAX = 50;
const NUMBER_INPUT_MIN_COUNT = 1;

function clampNumberInput(raw: string, min: number): number {
    const value = Number(raw);
    if (!Number.isFinite(value)) return min;
    return Math.min(NUMBER_INPUT_MAX, Math.max(min, Math.trunc(value)));
}

function clampRecommendationCount(raw: string): number {
    return clampNumberInput(raw, NUMBER_INPUT_MIN_COUNT);
}

type ListenBrainzParams = {
    range: string;
    min_listen: number;
    blacklist: string;
    blacklist_min: number;
    max_results: number;
};

function updateLbMinListen(params: ListenBrainzParams, raw: string): ListenBrainzParams {
    const max_results = clampRecommendationCount(String(params.max_results));
    const min_listen = Math.min(clampNumberInput(raw, 0), max_results);
    return { ...params, max_results, min_listen };
}

function updateLbBlacklistMin(params: ListenBrainzParams, raw: string): ListenBrainzParams {
    const max_results = clampRecommendationCount(String(params.max_results));
    const blacklist_min = Math.min(clampNumberInput(raw, 0), max_results);
    return { ...params, max_results, blacklist_min };
}

function updateLbMaxResults(params: ListenBrainzParams, raw: string): ListenBrainzParams {
    const max_results = clampRecommendationCount(raw);
    return {
        ...params,
        max_results,
        min_listen: Math.min(params.min_listen, max_results),
        blacklist_min: Math.min(params.blacklist_min, max_results),
    };
}


const LISTENBRAINZ_RANGE_OPTIONS = [
    /*{ label: "This week", value: "this_week" },*/
    { label: "This month", value: "this_month" },
    { label: "This year", value: "this_year" },
    { label: "Last week", value: "week" },
    { label: "Last month", value: "month" },
    { label: "Last year", value: "year" },
    { label: "All time", value: "all_time" },
] as const;

function BrainLoader({ className = "h-4 w-4" }: { className?: string }) {
    return (
        <span className={`inline-flex shrink-0 items-center justify-center ${className}`} aria-hidden="true">
            <svg
                viewBox="0 0 512 512"
                xmlns="http://www.w3.org/2000/svg"
                className="brain-loader h-full w-full"
                aria-hidden="true"
            >
                <g fill="currentColor" fillRule="evenodd">
                    <path
                        className="st0"
                        d="M502.472,256.833c-6.491-61.075-40.69-110.46-86.082-144.101c-45.887-34.04-103.296-52.724-157.675-52.76
                    c-56.443,0.009-91.262,7.173-114.312,17.082c-22.776,9.644-33.774,22.98-39.813,30.843c-24.68,4.029-49.262,18.348-68.77,38.697
                    C15.107,168.343,0.054,197.423,0,229.381c0,34.97,8.112,64.52,24.299,86.498c14.354,19.596,35.288,32.472,60.207,37.148
                    c1.638,9.456,5.56,20.003,13.672,29.647c8.412,10.06,19.888,17.383,33.454,22.032c13.584,4.675,29.329,6.836,47.234,6.853h75.084
                    c1.85,4.729,4.108,9.236,7.217,13.213c7.642,9.785,18.649,16.656,31.834,20.96c13.248,4.33,28.859,6.288,46.995,6.296
                    c8.909,0,17.348-0.407,24.512-0.752h0.026c5.136-0.274,9.555-0.469,12.698-0.469c9.466,0,18.526-2.302,26.318-6.819
                    c7.793-4.498,14.257-11.166,18.676-19.357c2.232-4.154,3.702-8.51,4.8-12.902c16.727-3.126,30.604-9.236,41.407-17.028
                    c12.663-9.121,21.367-20.11,27.283-30.09c11.556-19.552,16.267-41.247,16.285-61.384
                    C511.982,286.064,508.511,270.08,502.472,256.833z M475.862,352.849c-4.649,7.837-11.352,16.241-20.916,23.121
                    c-9.581,6.872-22.041,12.38-39.06,14.319l-9.519,1.072l-0.7,9.555c-0.292,4.127-1.576,8.767-3.737,12.76
                    c-2.506,4.578-5.835,7.962-9.918,10.335c-4.1,2.356-9.006,3.71-14.78,3.718c-4.073,0-8.714,0.24-13.858,0.496l1.922-0.088
                    l-1.914,0.088c-7.145,0.355-15.178,0.736-23.386,0.736c-21.943,0.035-38.299-3.356-48.747-8.864
                    c-5.251-2.736-9.06-5.906-11.884-9.511c-2.807-3.622-4.711-7.74-5.782-12.884l-1.904-9.218h-92.812
                    c-16.01,0-29.302-1.992-39.725-5.578c-10.44-3.622-17.94-8.678-23.28-15.054c-6.96-8.306-9.024-17.32-9.289-25.237l-0.31-10.077
                    l-10.024-1.044C72.72,328.914,55.354,318.97,42.86,302.18c-12.424-16.815-19.791-41.3-19.791-72.798
                    c-0.054-24.422,11.874-48.474,29.443-66.875c17.463-18.454,40.46-30.674,59.419-32.463l4.348-0.452l2.966-3.206
                    c1.328-1.452,2.382-2.851,3.294-4.002c5.986-7.474,12.114-15.806,31.002-24.139c18.845-8.156,50.652-15.222,105.174-15.213
                    c49.076-0.036,102.278,17.232,143.932,48.217c41.726,31.046,71.78,75.153,77.094,129.578l0.203,2.098l0.922,1.887
                    c4.844,9.776,8.094,23.608,8.066,38.414C488.932,319.776,484.992,337.451,475.862,352.849z"
                    />
                    <path
                        className="st0"
                        d="M357.042,146.417h24.059c5.172,0,9.378-4.242,9.378-9.573c0-5.215-4.206-9.43-9.378-9.43h-24.059
                    c-5.331,0-9.555,4.216-9.555,9.43C347.488,142.175,351.711,146.417,357.042,146.417z"
                    />
                    <path
                        className="st0"
                        d="M244.21,237.307c0,5.287,4.25,9.564,9.501,9.564c5.162,0,9.475-4.276,9.475-9.564v-51.82
                    c0-2.399,0.709-2.958,0.886-3.179c3.02-2.966,14.274-2.966,22.164-2.966l0.301,0.106h62.226c1.204,0,2.48-0.106,3.906-0.106
                    c5.012-0.221,11.202-0.434,13.796,2.072c1.647,1.611,2.604,5.19,2.604,9.988v31.809v1.416c-0.204,6.544-0.24,17.56,7.128,25.042
                    c2.869,2.958,8.2,6.464,16.896,6.464h48.89c5.136,0,9.352-4.233,9.352-9.555c0-5.198-4.216-9.519-9.352-9.519h-48.89l-3.418-0.806
                    c-1.736-1.797-1.621-8.332-1.621-11.467v-33.385c0-10.307-2.886-18.277-8.394-23.599c-8.484-8.138-20.022-7.801-27.629-7.483
                    c-1.258,0-2.302,0.045-3.268,0.045h-31.364c0.372-2.622,0.372-5.26,0.274-7.633v-27.602c0-5.189-4.268-9.476-9.448-9.476
                    c-5.286,0-9.43,4.286-9.43,9.476v27.752c0,2.222,0,5.738-0.47,6.65c0,0-1.301,0.832-6.314,0.832c-1.442,0-2.992,0-4.684,0
                    c-12.92-0.16-27.778-0.204-36.615,8.474c-2.887,2.922-6.5,8.208-6.5,16.648V237.307z"
                    />
                    <path
                        className="st0"
                        d="M213.677,159.709c5.304,0,9.555-4.348,9.555-9.528v-13.594h15.93c5.154,0,9.413-4.268,9.413-9.554
                    c0-5.162-4.259-9.493-9.413-9.493h-15.93v-10.467c0-5.233-4.251-9.528-9.555-9.528c-5.154,0-9.413,4.294-9.413,9.528v43.108
                    C204.264,155.361,208.523,159.709,213.677,159.709z"
                    />
                    <path
                        className="st0"
                        d="M110.841,173.682h39.468c6.438-0.229,12.565-0.229,15.452,2.807c2.559,2.498,3.967,8.111,3.967,16.17v37.051
                    c0,5.242,4.233,9.546,9.581,9.546c5.154,0,9.458-4.303,9.458-9.546v-7.882h14.886c5.251,0,9.44-4.277,9.44-9.51
                    c0-5.251-4.188-9.599-9.44-9.599h-14.886v-10.06c0-13.672-3.135-23.351-9.626-29.736c-8.421-8.448-19.8-8.368-28.877-8.288h-39.423
                    c-5.384,0-9.511,4.312-9.511,9.475C101.33,169.387,105.457,173.682,110.841,173.682z"
                    />
                    <path
                        className="st0"
                        d="M135.892,229.099c0-5.251-4.365-9.555-9.483-9.555H59.791c-5.26,0-9.555,4.304-9.555,9.555
                    c0,5.233,4.295,9.528,9.555,9.528h24.148v17.339c0,5.286,4.188,9.519,9.386,9.519c5.402,0,9.59-4.233,9.59-9.519v-17.339h23.494
                    C131.527,238.627,135.892,234.331,135.892,229.099z"
                    />
                    <path
                        className="st0"
                        d="M194.576,291.412c1.665,0,3.242,0,4.649,0h76.704c17.498,0,30.772-4.64,39.6-13.884
                    c13.566-14.363,12.619-35.634,11.919-49.687c-0.124-2.683-0.248-5.206-0.248-7.323c0-5.296-4.25-9.51-9.608-9.51
                    c-5.18,0-9.368,4.215-9.368,9.51c0,2.408,0.124,5.171,0.248,8.111c0.584,12.256,1.24,27.337-6.854,35.873
                    c-4.941,5.26-13.682,7.89-25.689,7.89h-76.704c-1.337,0-2.7,0-4.348,0c-15.133-0.23-40.584-0.638-56.753,15.319
                    c-9.068,8.944-13.681,21.545-13.681,37.396c0,5.153,4.17,9.52,9.484,9.52c5.18,0,9.51-4.366,9.51-9.52
                    c0-10.768,2.594-18.579,8.049-23.918C161.935,290.934,181.612,291.235,194.576,291.412z"
                    />
                    <path
                        className="st0"
                        d="M323.96,332.616c0-5.162-4.171-9.502-9.475-9.502H194.107c-5.19,0-9.538,4.34-9.538,9.502
                    c0,5.268,4.348,9.519,9.538,9.519h36.81v18.985c0,5.323,4.225,9.502,9.458,9.502c5.251,0,9.493-4.179,9.493-9.502v-18.985h64.617
                    C319.788,342.135,323.96,337.884,323.96,332.616z"
                    />
                    <path
                        className="st0"
                        d="M377.887,370.065h-4.471v-17.693c0-5.384-4.18-9.528-9.475-9.528c-5.26,0-9.502,4.145-9.502,9.528v17.693
                    h-32.941c-5.242,0-9.502,4.241-9.502,9.528c0,5.224,4.26,9.448,9.502,9.448h56.39c5.208,0,9.484-4.224,9.484-9.448
                    C387.371,374.305,383.095,370.065,377.887,370.065z"
                    />
                    <path
                        className="st0"
                        d="M421.579,323.114v-15.523h3.419c5.357,0,9.599-4.17,9.599-9.43c0-5.251-4.242-9.555-9.599-9.555h-66.459
                    c-5.225,0-9.511,4.304-9.511,9.555c0,5.26,4.286,9.43,9.511,9.43h43.983v15.523c0,5.358,4.313,9.502,9.556,9.502
                    C417.311,332.616,421.579,328.472,421.579,323.114z"
                    />
                    <path
                        className="st0"
                        d="M451.333,347.909h-24.042c-5.304,0-9.546,4.18-9.546,9.467c0,5.286,4.241,9.43,9.546,9.43h24.042
                    c5.33,0,9.616-4.144,9.616-9.43C460.95,352.089,456.663,347.909,451.333,347.909z"
                    />
                </g>
            </svg>
        </span>
    );
}

type ThemeMode = "light" | "dark";
type MainTab = "artists" | "albums";
type InputMode = "manual" | "listenbrainz";

type UrlRow = {
    type?: number;
    url?: string;
    urls?: string;
};

type ArtistSearchRow = {
    artist_id: string | number;
    gid?: string;
    name?: string;
    disambiguation?: string;
};

type AlbumSearchRow = {
    release_group_id: string | number;
    gid?: string;
    title?: string;
    artist?: string;
    disambiguation?: string;
};

type ArtistPredictRow = {
    gid?: string;
    name?: string;
    genre?: string[];
    genres?: string[];
    urls?: UrlRow[];
};

type AlbumPredictRow = {
    gid?: string;
    title?: string;
    artist?: string;
    genres?: string[];
    genre?: string[];
    length?: number | string | null;
    tracks?: number | string | null;
    urls?: UrlRow[];
};

function formatArtist(item: ArtistSearchRow) {
    if (item.disambiguation) {
        return `${item.name ?? "Unknown artist"} (${item.disambiguation})`;
    }
    return item.name ?? "Unknown artist";
}

function formatAlbum(item: AlbumSearchRow) {
    const title = item.title ?? "Unknown album";
    return item.artist ? `${title} - ${item.artist}` : title;
}

function normalizeDurationToSeconds(length?: number | string | null) {
    if (length == null) return null;
    const parsed = Number(length);
    if (!Number.isFinite(parsed) || parsed < 0) return null;
    if (parsed >= 10000) return Math.round(parsed / 1000);
    return Math.round(parsed);
}

function formatDuration(length?: number | string | null) {
    const totalSeconds = normalizeDurationToSeconds(length);
    if (totalSeconds == null) return null;

    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) return `${hours}h ${minutes} min`;
    if (minutes > 0) return `${minutes} min`;
    return `${seconds} sec`;
}

function getOfficialUrl(urls?: UrlRow[]) {
    return urls?.find((item) => item.type === 183)?.url || urls?.find((item) => item.type === 183)?.urls || null;
}

function canMakeRequest(timestamps: number[]) {
    const now = Date.now();
    const fresh = timestamps.filter((t) => now - t < WINDOW_SECONDS * 1000);

    return {
        allowed: fresh.length < MAX_REQUESTS,
        fresh,
    };
}

const inter = Inter({
    subsets: ["latin"],
});

const oswald = Oswald({
    subsets: ["latin"],
});

export default function RecoApp() {
    const [theme, setTheme] = useState<ThemeMode>("light");
    const [mounted, setMounted] = useState(false);

    const [activeTab, setActiveTab] = useState<MainTab>("artists");
    const [inputMode, setInputMode] = useState<InputMode>("manual");


    const [artistQuery, setArtistQuery] = useState("");
    const [artistSearchResults, setArtistSearchResults] = useState<ArtistSearchRow[]>([]);
    const [artistSearchError, setArtistSearchError] = useState<string | null>(null);
    const [artistSelected, setArtistSelected] = useState<Record<string, { label: string; name?: string }>>({});
    const [artistBlacklisted, setArtistBlacklisted] = useState<Record<string, { label: string; name?: string }>>({});
    const [artistTopN, setArtistTopN] = useState(5);
    const [artistRecommendations, setArtistRecommendations] = useState<ArtistPredictRow[]>([]);
    const [artistRequestTimestamps, setArtistRequestTimestamps] = useState<number[]>([]);
    const [artistBlacklistQuery, setArtistBlacklistQuery] = useState("");
    const [artistBlacklistSearchResults, setArtistBlacklistSearchResults] = useState<ArtistSearchRow[]>([]);
    const [artistDropdownOpen, setArtistDropdownOpen] = useState(false);
    const [artistBlacklistDropdownOpen, setArtistBlacklistDropdownOpen] = useState(false);
    const [artistHasSearched, setArtistHasSearched] = useState(false);
    const [artistBlacklistHasSearched, setArtistBlacklistHasSearched] = useState(false);
    const [artistOptionalOpen, setArtistOptionalOpen] = useState(false);
    const [artistLoading, setArtistLoading] = useState(false);

    const [albumQuery, setAlbumQuery] = useState("");
    const [albumSearchResults, setAlbumSearchResults] = useState<AlbumSearchRow[]>([]);
    const [albumSearchError, setAlbumSearchError] = useState<string | null>(null);
    const [albumSelected, setAlbumSelected] = useState<
        Record<string, { label: string; title?: string; artist?: string }>
    >({});
    const [albumBlacklisted, setAlbumBlacklisted] = useState<
        Record<string, { label: string; title?: string; artist?: string }>
    >({});
    const [albumTopN, setAlbumTopN] = useState(5);
    const [albumRecommendations, setAlbumRecommendations] = useState<AlbumPredictRow[]>([]);
    const [albumRequestTimestamps, setAlbumRequestTimestamps] = useState<number[]>([]);
    const [albumBlacklistQuery, setAlbumBlacklistQuery] = useState("");
    const [albumBlacklistSearchResults, setAlbumBlacklistSearchResults] = useState<AlbumSearchRow[]>([]);
    const [albumDropdownOpen, setAlbumDropdownOpen] = useState(false);
    const [albumBlacklistDropdownOpen, setAlbumBlacklistDropdownOpen] = useState(false);
    const [albumHasSearched, setAlbumHasSearched] = useState(false);
    const [albumBlacklistHasSearched, setAlbumBlacklistHasSearched] = useState(false);
    const [albumOptionalOpen, setAlbumOptionalOpen] = useState(false);
    const [albumLoading, setAlbumLoading] = useState(false);

    const [artistLbOptionalOpen, setArtistLbOptionalOpen] = useState(false);
    const [albumLbOptionalOpen, setAlbumLbOptionalOpen] = useState(false);

    const [artistLbParams, setArtistLbParams] = useState<ListenBrainzParams>({
        range: "week",
        min_listen: 5,
        blacklist: "",
        blacklist_min: 5,
        max_results: 10,
    });

    const [albumLbParams, setAlbumLbParams] = useState<ListenBrainzParams>({
        range: "week",
        min_listen: 5,
        blacklist: "",
        blacklist_min: 5,
        max_results: 10,
    });

    const artistSelectedEntries = useMemo(() => Object.entries(artistSelected), [artistSelected]);
    const artistBlacklistedEntries = useMemo(() => Object.entries(artistBlacklisted), [artistBlacklisted]);
    const albumSelectedEntries = useMemo(() => Object.entries(albumSelected), [albumSelected]);
    const albumBlacklistedEntries = useMemo(() => Object.entries(albumBlacklisted), [albumBlacklisted]);

    const activeLbParams = activeTab === "artists" ? artistLbParams : albumLbParams;
    const setActiveLbParams =
        activeTab === "artists" ? setArtistLbParams : setAlbumLbParams;
    const activeLbOptionalOpen = activeTab === "artists" ? artistLbOptionalOpen : albumLbOptionalOpen;
    const setActiveLbOptionalOpen =
        activeTab === "artists" ? setArtistLbOptionalOpen : setAlbumLbOptionalOpen;
    const lbEntityLabel = activeTab === "artists" ? "artists" : "albums";
    useEffect(() => {
        setMounted(true);

        const savedTheme = typeof window !== "undefined" ? window.localStorage.getItem("reco-theme") : null;

        if (savedTheme === "light" || savedTheme === "dark") {
            setTheme(savedTheme);
            return;
        }

        if (typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches) {
            setTheme("dark");
        }
    }, []);

    useEffect(() => {
        if (!mounted) return;
        document.documentElement.classList.toggle("dark", theme === "dark");
        window.localStorage.setItem("reco-theme", theme);
    }, [theme, mounted]);

    async function runArtistSearch(query: string, mode: "main" | "blacklist" = "main") {
        const trimmed = query.trim();

        if (mode === "main") {
            setArtistSearchError(null);
            setArtistSearchResults([]);
            setArtistHasSearched(false);
        } else {
            setArtistBlacklistSearchResults([]);
            setArtistBlacklistHasSearched(false);
        }

        if (trimmed.length < 1) {
            if (mode === "main") {
                setArtistDropdownOpen(false);
                setArtistSearchResults([]);
                setArtistSearchError(null);
                setArtistHasSearched(false);
            } else {
                setArtistBlacklistDropdownOpen(false);
                setArtistBlacklistSearchResults([]);
                setArtistBlacklistHasSearched(false);
            }
            return;
        }

        const res = await fetch("/api/search/artist", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: trimmed }),
        });

        if (!res.ok) {
            if (mode === "main") {
                setArtistSearchError(await res.text());
                setArtistHasSearched(true);
                setArtistDropdownOpen(false);
            } else {
                setArtistBlacklistHasSearched(true);
                setArtistBlacklistDropdownOpen(false);
            }
            return;
        }

        const data = await res.json();

        if (!Array.isArray(data)) {
            if (mode === "main") {
                setArtistSearchError("Invalid response format.");
                setArtistHasSearched(true);
                setArtistDropdownOpen(false);
            } else {
                setArtistBlacklistHasSearched(true);
                setArtistBlacklistDropdownOpen(false);
            }
            return;
        }

        if (mode === "main") {
            setArtistSearchResults(data);
            setArtistDropdownOpen(data.length > 0);
            setArtistHasSearched(true);
        } else {
            setArtistBlacklistSearchResults(data);
            setArtistBlacklistDropdownOpen(data.length > 0);
            setArtistBlacklistHasSearched(true);
        }
    }

    async function runAlbumSearch(query: string, mode: "main" | "blacklist" = "main") {
        const trimmed = query.trim();

        if (mode === "main") {
            setAlbumSearchError(null);
            setAlbumSearchResults([]);
            setAlbumHasSearched(false);
        } else {
            setAlbumBlacklistSearchResults([]);
            setAlbumBlacklistHasSearched(false);
        }

        if (trimmed.length < 1) {
            if (mode === "main") {
                setAlbumDropdownOpen(false);
                setAlbumSearchResults([]);
                setAlbumSearchError(null);
                setAlbumHasSearched(false);
            } else {
                setAlbumBlacklistDropdownOpen(false);
                setAlbumBlacklistSearchResults([]);
                setAlbumBlacklistHasSearched(false);
            }
            return;
        }

        const res = await fetch("/api/search/album", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: trimmed }),
        });

        if (!res.ok) {
            if (mode === "main") {
                setAlbumSearchError(await res.text());
                setAlbumHasSearched(true);
                setAlbumDropdownOpen(false);
            } else {
                setAlbumBlacklistHasSearched(true);
                setAlbumBlacklistDropdownOpen(false);
            }
            return;
        }

        const data = await res.json();

        if (!Array.isArray(data)) {
            if (mode === "main") {
                setAlbumSearchError("Invalid response format.");
                setAlbumHasSearched(true);
                setAlbumDropdownOpen(false);
            } else {
                setAlbumBlacklistHasSearched(true);
                setAlbumBlacklistDropdownOpen(false);
            }
            return;
        }

        if (mode === "main") {
            setAlbumSearchResults(data);
            setAlbumDropdownOpen(data.length > 0);
            setAlbumHasSearched(true);
        } else {
            setAlbumBlacklistSearchResults(data);
            setAlbumBlacklistDropdownOpen(data.length > 0);
            setAlbumBlacklistHasSearched(true);
        }
    }

    function handleArtistEnterSearch(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key !== "Enter") return;
        const trimmed = artistQuery.trim();
        if (trimmed.length < 1) return;
        e.preventDefault();
        runArtistSearch(trimmed, "main");
    }

    function handleArtistBlacklistEnterSearch(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key !== "Enter") return;
        const trimmed = artistBlacklistQuery.trim();
        if (trimmed.length < 1) return;
        e.preventDefault();
        runArtistSearch(trimmed, "blacklist");
    }

    function handleAlbumEnterSearch(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key !== "Enter") return;
        const trimmed = albumQuery.trim();
        if (trimmed.length < 1) return;
        e.preventDefault();
        runAlbumSearch(trimmed, "main");
    }

    function handleAlbumBlacklistEnterSearch(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key !== "Enter") return;
        const trimmed = albumBlacklistQuery.trim();
        if (trimmed.length < 1) return;
        e.preventDefault();
        runAlbumSearch(trimmed, "blacklist");
    }

    useEffect(() => {
        if (activeTab !== "artists" || inputMode !== "manual") return;

        const trimmed = artistQuery.trim();

        if (trimmed.length < 1) {
            setArtistSearchResults([]);
            setArtistDropdownOpen(false);
            setArtistSearchError(null);
            setArtistHasSearched(false);
            return;
        }

        const timer = setTimeout(() => {
            runArtistSearch(artistQuery, "main");
        }, SEARCH_DEBOUNCE_MS);

        return () => clearTimeout(timer);
    }, [artistQuery, activeTab, inputMode]);

    useEffect(() => {
        if (activeTab !== "artists" || inputMode !== "manual") return;

        const trimmed = artistBlacklistQuery.trim();

        if (trimmed.length < 1) {
            setArtistBlacklistSearchResults([]);
            setArtistBlacklistDropdownOpen(false);
            setArtistBlacklistHasSearched(false);
            return;
        }

        const timer = setTimeout(() => {
            runArtistSearch(artistBlacklistQuery, "blacklist");
        }, SEARCH_DEBOUNCE_MS);

        return () => clearTimeout(timer);
    }, [artistBlacklistQuery, activeTab, inputMode]);

    useEffect(() => {
        if (activeTab !== "albums" || inputMode !== "manual") return;

        const trimmed = albumQuery.trim();

        if (trimmed.length < 1) {
            setAlbumSearchResults([]);
            setAlbumDropdownOpen(false);
            setAlbumSearchError(null);
            setAlbumHasSearched(false);
            return;
        }

        const timer = setTimeout(() => {
            runAlbumSearch(albumQuery, "main");
        }, SEARCH_DEBOUNCE_MS);

        return () => clearTimeout(timer);
    }, [albumQuery, activeTab, inputMode]);

    useEffect(() => {
        if (activeTab !== "albums" || inputMode !== "manual") return;

        const trimmed = albumBlacklistQuery.trim();

        if (trimmed.length < 1) {
            setAlbumBlacklistSearchResults([]);
            setAlbumBlacklistDropdownOpen(false);
            setAlbumBlacklistHasSearched(false);
            return;
        }

        const timer = setTimeout(() => {
            runAlbumSearch(albumBlacklistQuery, "blacklist");
        }, SEARCH_DEBOUNCE_MS);

        return () => clearTimeout(timer);
    }, [albumBlacklistQuery, activeTab, inputMode]);

    async function fetchArtistRecommendations() {
        const ids = Object.keys(artistSelected);

        if (!ids.length) {
            alert("Select at least one artist.");
            return;
        }

        const checked = canMakeRequest(artistRequestTimestamps);
        setArtistRequestTimestamps(checked.fresh);

        if (!checked.allowed) {
            alert("Too many requests. Try again in a minute.");
            return;
        }

        setArtistRequestTimestamps([...checked.fresh, Date.now()]);
        setArtistLoading(true);

        try {
            const res = await fetch("/api/predict/artist", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ArtistIds: ids,
                    TopN: artistTopN,
                    BlacklistArtistIds: Object.keys(artistBlacklisted),
                }),
            });

            if (!res.ok) {
                alert(await res.text());
                return;
            }

            const data = await res.json();
            setArtistRecommendations(data.artists ?? []);
        } finally {
            setArtistLoading(false);
        }
    }

    async function fetchAlbumRecommendations() {
        const ids = Object.keys(albumSelected);

        if (!ids.length) {
            alert("Select at least one album.");
            return;
        }

        const checked = canMakeRequest(albumRequestTimestamps);
        setAlbumRequestTimestamps(checked.fresh);

        if (!checked.allowed) {
            alert("Too many requests. Try again in a minute.");
            return;
        }

        setAlbumRequestTimestamps([...checked.fresh, Date.now()]);
        setAlbumLoading(true);

        try {
            const res = await fetch("/api/predict/album", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    release_group_id: ids,
                    genre_id: [],
                    response_length: albumTopN,
                    blacklist_release_group_id: Object.keys(albumBlacklisted),
                }),
            });

            if (!res.ok) {
                alert(await res.text());
                return;
            }

            const data = await res.json();
            setAlbumRecommendations(Array.isArray(data) ? data : data.albums ?? []);
        } finally {
            setAlbumLoading(false);
        }
    }

    async function fetchListenBrainzArtistRecommendations() {
        const checked = canMakeRequest(artistRequestTimestamps);
        setArtistRequestTimestamps(checked.fresh);

        if (!checked.allowed) {
            alert("Too many requests. Try again in a minute.");
            return;
        }

        setArtistRequestTimestamps([...checked.fresh, Date.now()]);
        setArtistLoading(true);

        try {
            const payload = {
                range: artistLbParams.range,
                min_listen: artistLbParams.min_listen,
                blacklist: artistLbParams.blacklist.trim() || null,
                blacklist_min: artistLbParams.blacklist_min,
                max_results: artistLbParams.max_results,
            };

            const res = await fetch("/api/listenbrainz/artist", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                alert(await res.text());
                return;
            }

            const data = await res.json();
            setArtistRecommendations(data.artists ?? (Array.isArray(data) ? data : []));
        } finally {
            setArtistLoading(false);
        }
    }

    async function fetchListenBrainzAlbumRecommendations() {
        const checked = canMakeRequest(albumRequestTimestamps);
        setAlbumRequestTimestamps(checked.fresh);

        if (!checked.allowed) {
            alert("Too many requests. Try again in a minute.");
            return;
        }

        setAlbumRequestTimestamps([...checked.fresh, Date.now()]);
        setAlbumLoading(true);

        try {
            const payload = {
                range: albumLbParams.range,
                min_listen: albumLbParams.min_listen,
                blacklist: albumLbParams.blacklist.trim() || null,
                blacklist_min: albumLbParams.blacklist_min,
                max_results: albumLbParams.max_results,
            };

            const res = await fetch("/api/listenbrainz/album", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                alert(await res.text());
                return;
            }

            const data = await res.json();
            setAlbumRecommendations(Array.isArray(data) ? data : data.albums ?? []);
        } finally {
            setAlbumLoading(false);
        }
    }

    const isDark = theme === "dark";

    const mutedTextClass = isDark ? "text-[#cdbfb1]" : "text-[#74685d]";
    const subtleTextClass = isDark ? "text-[#aa998a]" : "text-[#908379]";
    const dropdownMetaClass = isDark ? "text-[#aa998a]" : "text-[#908379]";

    const shellClass = isDark
        ? "min-h-screen bg-[#181513] text-[#f6ede3] selection:bg-[#c7744d] selection:text-white"
        : "min-h-screen bg-[#f6f0e8] text-[#2f2925] selection:bg-[#b8653c] selection:text-white";

    const heroClass = isDark
        ? "mb-4 rounded-[28px] border border-[#3d342e] bg-[linear-gradient(135deg,#221d19_0%,#2b231e_55%,#1c1815_100%)] p-6 shadow-[0_18px_44px_rgba(0,0,0,0.28)]"
        : "mb-4 rounded-[28px] border border-[#e6dacb] bg-[linear-gradient(135deg,#fdf9f4_0%,#fbf5ee_55%,#fffbf7_100%)] p-6 shadow-[0_14px_32px_rgba(91,63,35,0.06)]";

    const panelClass = isDark
        ? "rounded-[26px] border border-[#3b332d] bg-[#221d19] p-5 shadow-[0_12px_30px_rgba(0,0,0,0.22)]"
        : "rounded-[26px] border border-[#e8ddcf] bg-[#fffdf9] p-5 shadow-[0_10px_26px_rgba(91,63,35,0.05)]";

    const recommendationsPanelClass = panelClass;

    const resultCardClass = isDark
        ? "rounded-[24px] border border-[#4b4038] bg-[#2a2420] p-5 shadow-[0_10px_22px_rgba(0,0,0,0.18)]"
        : "rounded-[24px] border border-[#e9dfd3] bg-[#fffbf7] p-5 shadow-[0_8px_18px_rgba(91,63,35,0.03)]";

    const nestedPanelClass = isDark
        ? "mt-5 rounded-[22px] border border-[#342c27] bg-[#1a1714]"
        : "mt-5 rounded-[22px] border border-[#e7dccf] bg-[#fcf7f0]";

    const inputClass = isDark
        ? "w-full rounded-2xl border border-[#4b4038] bg-[#171310] px-4 py-3 text-[#f6ede3] outline-none placeholder:text-[#8d7d70] focus:border-[#c58a63] focus:ring-4 focus:ring-[#c58a63]/10"
        : "w-full rounded-2xl border border-[#e8c9be] bg-[#fcf4f1] px-4 py-3 text-[#2f2925] outline-none placeholder:text-[#9a8d82] focus:border-[#b8653c] focus:ring-4 focus:ring-[#b8653c]/10";

    const selectClass = isDark
        ? "w-full appearance-none rounded-2xl border border-[#4b4038] bg-[#171310] px-4 py-3 text-[#f6ede3] outline-none focus:border-[#c58a63] focus:ring-4 focus:ring-[#c58a63]/10"
        : "w-full appearance-none rounded-2xl border border-[#e8c9be] bg-[#fcf4f1] px-4 py-3 text-[#2f2925] outline-none focus:border-[#b8653c] focus:ring-4 focus:ring-[#b8653c]/10";

    const blacklistClass = isDark
        ? "flex items-center justify-between gap-3 rounded-2xl border border-[#73453b] bg-[#39211c] px-4 py-3"
        : "flex items-center justify-between gap-3 rounded-2xl border border-[#e8c9be] bg-[#fcf4f1] px-4 py-3";

    const dropdownClass = isDark
        ? "absolute z-20 mt-2 max-h-72 w-full overflow-auto rounded-2xl border border-[#40362f] bg-[#221d19] shadow-[0_20px_40px_rgba(0,0,0,0.34)]"
        : "absolute z-20 mt-2 max-h-72 w-full overflow-auto rounded-2xl border border-[#e6dacb] bg-[#fffdf9] shadow-[0_18px_40px_rgba(91,63,35,0.10)]";

    const dropdownItemClass = isDark
        ? "block w-full border-b border-[#312924] px-4 py-3 text-left last:border-b-0 hover:bg-[#2c2521]"
        : "block w-full border-b border-[#f0e7db] px-4 py-3 text-left last:border-b-0 hover:bg-[#fcf7f0]";

    const emptyClass = isDark
        ? "rounded-2xl border border-dashed border-[#4a4038] bg-[#1a1714] px-4 py-3 text-sm text-[#ab998b]"
        : "rounded-2xl border border-dashed border-[#dfd3c5] bg-[#fdf8f2] px-4 py-3 text-sm text-[#82766b]";

    const selectedClass = isDark
        ? "flex items-center justify-between gap-3 rounded-2xl border border-[#6b5d46] bg-[#332d25] px-4 py-3"
        : "flex items-center justify-between gap-3 rounded-2xl border border-[#cfd1aa] bg-[#eef0d8] px-4 py-3";

    const removeBtnClass = isDark
        ? "rounded-xl border border-[#4b4038] bg-[#171310] px-3 py-1 text-sm text-[#f6ede3] hover:bg-[#241f1b]"
        : "rounded-xl border border-[#e0d5c7] bg-white px-3 py-1 text-sm text-[#3a332d] hover:bg-[#fdf8f2]";

    const redRemoveBtnClass = isDark
        ? "rounded-xl border border-[#73453b] bg-[#171310] px-3 py-1 text-sm text-[#f2c1b4] hover:bg-[#241613]"
        : "rounded-xl border border-[#e1b0a2] bg-[#fff8f5] px-3 py-1 text-sm text-[#924d3d] hover:bg-[#fdf0eb]";

    const chipClass = isDark
        ? "rounded-full border border-[#6b614c] bg-[#342f27] px-2.5 py-1 text-xs text-[#e5d9bf]"
        : "rounded-full border border-[#e4d9bd] bg-[#f6efdc] px-2.5 py-1 text-xs text-[#5f5449]";

    const primaryButtonClass = isDark
        ? "mt-4 flex w-full items-center justify-center gap-2 rounded-2xl bg-[#c7744d] px-4 py-3 text-sm font-semibold text-[#fff7f1] transition hover:bg-[#b66642] shadow-[0_12px_26px_rgba(199,116,77,0.24)]"
        : "mt-4 flex w-full items-center justify-center gap-2 rounded-2xl bg-[#b8653c] px-4 py-3 text-sm font-semibold text-[#fff7f0] shadow-[0_10px_20px_rgba(184,101,60,0.18)] transition hover:bg-[#a55834]";

    const tabActiveClass = isDark
        ? "border border-[#f2dfd0] bg-[#f2dfd0] text-[#241c17]"
        : "bg-[#2f2925] text-[#fffaf4]";

    const tabInactiveClass = isDark
        ? "border border-[#403731] bg-[#221d19] text-[#f6ede3] hover:bg-[#2a2420]"
        : "border border-[#e4d8ca] bg-[#fffdf9] text-[#5f5449] hover:bg-[#fcf7f0]";

    const subTabActiveClass = isDark
        ? "border border-[#4b4038] bg-[#2b2521] text-[#f6ede3]"
        : "border border-[#e8c9be] bg-[#fcf4f1] text-[#2f2925]";

    const subTabInactiveClass = isDark
        ? "border border-[#342c27] bg-transparent text-[#aa998a] hover:bg-[#221d19]"
        : "border border-[#e7dccf] bg-transparent text-[#908379] hover:bg-[#fcf7f0]";

    const themeButtonClass = isDark
        ? "rounded-2xl border border-[#433931] bg-[#221d19] px-3 py-2 text-lg leading-none text-[#f6ede3] hover:bg-[#2b2521]"
        : "rounded-2xl border border-[#e4d8ca] bg-[#fffdf9] px-3 py-2 text-lg leading-none text-[#2f2925] hover:bg-[#fcf7f0]";

    const optionalHeaderClass =
        "flex w-full items-center justify-between px-4 py-4 text-left text-base font-bold tracking-[0.01em]";

    function renderThinking(isLoading: boolean) {
        return isLoading ? (
            <span className="loading-blink inline-flex items-center gap-2">
                <BrainLoader className="h-6 w-6" />
                <span>rec_o is thinking</span>
                <BrainLoader className="h-6 w-6" />
            </span>
        ) : (
            "Get recommendations"
        );
    }

    function renderArtistResults() {
        return (
            <div className={recommendationsPanelClass} aria-live="polite" aria-busy={artistLoading}>
                <h2 className="mb-3 text-base font-bold">Recommendations</h2>

                {artistRecommendations.length === 0 ? (
                    <div className={emptyClass}>
                        {artistLoading ? "Loading artist recommendations..." : "Recommendations will appear here after you run the search."}
                    </div>
                ) : (
                    <div className="space-y-4">
                        {artistRecommendations.map((artist, idx) => {
                            const genres = artist.genre || artist.genres || [];
                            const musicbrainzUrl = artist.gid ? `https://musicbrainz.org/artist/${artist.gid}` : null;
                            const website = getOfficialUrl(artist.urls);

                            return (
                                <div key={`${artist.gid ?? artist.name ?? "artist"}-${idx}`} className={resultCardClass}>
                                    <div className="mb-2 text-lg font-bold">
                                        {musicbrainzUrl ? (
                                            <a
                                                href={musicbrainzUrl}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="hover:underline"
                                            >
                                                {artist.name ?? "Unknown artist"}
                                            </a>
                                        ) : (
                                            artist.name ?? "Unknown artist"
                                        )}
                                    </div>

                                    {genres.length > 0 && (
                                        <div className="mb-3 flex flex-wrap gap-2">
                                            {genres.slice(0, 8).map((genre) => (
                                                <span key={genre} className={chipClass}>
                                                    {genre}
                                                </span>
                                            ))}
                                        </div>
                                    )}

                                    {website && (
                                        <a
                                            href={website}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-sm hover:underline"
                                        >
                                            Official website
                                        </a>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        );
    }

    function renderAlbumResults() {
        return (
            <div className={recommendationsPanelClass} aria-live="polite" aria-busy={albumLoading}>
                <h2 className="mb-3 text-base font-bold">Recommendations</h2>

                {albumRecommendations.length === 0 ? (
                    <div className={emptyClass}>
                        {albumLoading ? "Loading album recommendations..." : "Recommendations will appear here after you run the search."}
                    </div>
                ) : (
                    <div className="space-y-4">
                        {albumRecommendations.map((album, idx) => {
                            const genres = album.genres || album.genre || [];
                            const musicbrainzUrl = album.gid ? `https://musicbrainz.org/release-group/${album.gid}` : null;
                            const website = getOfficialUrl(album.urls);
                            const duration = formatDuration(album.length);

                            return (
                                <div key={`${album.gid ?? album.title ?? "album"}-${idx}`} className={resultCardClass}>
                                    <div className="mb-2 text-lg font-bold">
                                        {musicbrainzUrl ? (
                                            <a
                                                href={musicbrainzUrl}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="hover:underline"
                                            >
                                                {album.title ?? "Unknown album"}
                                            </a>
                                        ) : (
                                            album.title ?? "Unknown album"
                                        )}

                                        {album.artist && (
                                            <span className={`ml-2 text-[0.9em] font-normal ${mutedTextClass}`}>
                                                {album.artist}
                                            </span>
                                        )}
                                    </div>

                                    {(duration || album.tracks != null) && (
                                        <div className={`mb-3 text-sm ${mutedTextClass}`}>
                                            {[
                                                duration,
                                                album.tracks != null
                                                    ? `${album.tracks} ${Number(album.tracks) === 1 ? "track" : "tracks"}`
                                                    : null,
                                            ]
                                                .filter(Boolean)
                                                .join(" • ")}
                                        </div>
                                    )}

                                    {genres.length > 0 && (
                                        <div className="mb-3 flex flex-wrap gap-2">
                                            {genres.slice(0, 8).map((genre) => (
                                                <span key={genre} className={chipClass}>
                                                    {genre}
                                                </span>
                                            ))}
                                        </div>
                                    )}

                                    {website && (
                                        <a
                                            href={website}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-sm hover:underline"
                                        >
                                            Official website
                                        </a>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        );
    }

    return (
        <main className={`${shellClass} ${inter.className}`}>
            <div className="mx-auto max-w-6xl px-6 py-8">
                <div className={heroClass}>
                    <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
                        <div className="min-w-0">
                            <div className="mb-3 inline-flex rounded-full border border-current/10 px-3 py-1 text-[11px] font-semibold lowercase tracking-[0.22em] opacity-80">
                                music discovery engine
                            </div>
                            <h1
                                className={`${oswald.className} lowercase text-[3.2rem] leading-[0.9] tracking-[0.03em] sm:text-[4.4rem]`}
                                style={{ textTransform: "lowercase" }}
                            >
                                rec_o
                            </h1>
                            <p
                                className={`mt-3 max-w-none text-sm sm:text-[15px] sm:whitespace-nowrap ${mutedTextClass}`}
                            >
                                Search manually or use a ListenBrainz profile to generate artist and album recommendations.
                            </p>
                        </div>

                        <button
                            type="button"
                            onClick={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
                            className={`shrink-0 self-start ${themeButtonClass}`}
                            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
                        >
                            {theme === "light" ? "🌙" : "☀️"}
                        </button>
                    </div>
                </div>

                <div className="mb-6 flex flex-wrap gap-2">
                    <button
                        type="button"
                        onClick={() => setInputMode("manual")}
                        className={`rounded-[16px] px-4 py-2 text-sm font-semibold transition ${
                            inputMode === "manual" ? subTabActiveClass : subTabInactiveClass
                        }`}
                    >
                        🔎 Manual search
                    </button>
                    <button
                        type="button"
                        onClick={() => setInputMode("listenbrainz")}
                        className={`rounded-[16px] px-4 py-2 text-sm font-semibold transition ${
                            inputMode === "listenbrainz" ? subTabActiveClass : subTabInactiveClass
                        }`}
                    >
                        🧠 ListenBrainz profile
                    </button>
                </div>

                <div className="grid gap-6 lg:grid-cols-2">
                    <div>
                        <div className={`mb-4 ${panelClass}`}>
                            <div className="mb-4 flex flex-wrap gap-2">
                                <button
                                    type="button"
                                    onClick={() => setActiveTab("artists")}
                                    className={`rounded-[16px] px-4 py-2 text-sm font-semibold transition ${
                                        activeTab === "artists" ? tabActiveClass : tabInactiveClass
                                    }`}
                                >
                                    🎤 Artists
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setActiveTab("albums")}
                                    className={`rounded-[16px] px-4 py-2 text-sm font-semibold transition ${
                                        activeTab === "albums" ? tabActiveClass : tabInactiveClass
                                    }`}
                                >
                                    💿 Albums
                                </button>
                            </div>

                            {activeTab === "artists" && inputMode === "manual" && (
                                <>
                                    <h2 className="mb-2 text-base font-bold">Search artist</h2>

                                    <div className="relative">
                                        <input
                                            className={inputClass}
                                            placeholder="Type an artist name..."
                                            value={artistQuery}
                                            onChange={(e) => setArtistQuery(e.target.value)}
                                            onKeyDown={handleArtistEnterSearch}
                                            onFocus={() => artistSearchResults.length > 0 && setArtistDropdownOpen(true)}
                                            onBlur={() => setTimeout(() => setArtistDropdownOpen(false), 150)}
                                        />

                                        {artistDropdownOpen && artistSearchResults.length > 0 && (
                                            <div className={dropdownClass}>
                                                {artistSearchResults.map((artist, idx) => {
                                                    const id = String(artist.artist_id);

                                                    return (
                                                        <button
                                                            key={`${id}-${idx}`}
                                                            type="button"
                                                            className={dropdownItemClass}
                                                            onMouseDown={(e) => e.preventDefault()}
                                                            onClick={() => {
                                                                setArtistSelected((prev) => ({
                                                                    ...prev,
                                                                    [id]: {
                                                                        label: formatArtist(artist),
                                                                        name: artist.name,
                                                                    },
                                                                }));

                                                                setArtistBlacklisted((prev) => {
                                                                    const copy = { ...prev };
                                                                    delete copy[id];
                                                                    return copy;
                                                                });

                                                                setArtistQuery("");
                                                                setArtistSearchResults([]);
                                                                setArtistDropdownOpen(false);
                                                                setArtistSearchError(null);
                                                                setArtistHasSearched(false);
                                                            }}
                                                        >
                                                            <div className="text-sm font-medium">{artist.name ?? "Unknown artist"}</div>
                                                            {artist.disambiguation && (
                                                                <div className={`mt-0.5 text-xs ${dropdownMetaClass}`}>
                                                                    {artist.disambiguation}
                                                                </div>
                                                            )}
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </div>

                                    {artistSearchError && <p className="mt-3 text-sm text-amber-700">{artistSearchError}</p>}

                                    {artistHasSearched && !artistSearchResults.length && !artistSearchError && (
                                        <div className={`mt-3 ${emptyClass}`}>No artist found for this search.</div>
                                    )}
                                </>
                            )}

                            {activeTab === "albums" && inputMode === "manual" && (
                                <>
                                    <h2 className="mb-2 text-base font-bold">Search album</h2>

                                    <div className="relative">
                                        <input
                                            className={inputClass}
                                            placeholder="Type an album, EP or single title..."
                                            value={albumQuery}
                                            onChange={(e) => setAlbumQuery(e.target.value)}
                                            onKeyDown={handleAlbumEnterSearch}
                                            onFocus={() => albumSearchResults.length > 0 && setAlbumDropdownOpen(true)}
                                            onBlur={() => setTimeout(() => setAlbumDropdownOpen(false), 150)}
                                        />

                                        {albumDropdownOpen && albumSearchResults.length > 0 && (
                                            <div className={dropdownClass}>
                                                {albumSearchResults.map((album, idx) => {
                                                    const id = String(album.release_group_id);

                                                    return (
                                                        <button
                                                            key={`${id}-${idx}`}
                                                            type="button"
                                                            className={dropdownItemClass}
                                                            onMouseDown={(e) => e.preventDefault()}
                                                            onClick={() => {
                                                                setAlbumSelected((prev) => ({
                                                                    ...prev,
                                                                    [id]: {
                                                                        label: formatAlbum(album),
                                                                        title: album.title,
                                                                        artist: album.artist,
                                                                    },
                                                                }));

                                                                setAlbumBlacklisted((prev) => {
                                                                    const copy = { ...prev };
                                                                    delete copy[id];
                                                                    return copy;
                                                                });

                                                                setAlbumQuery("");
                                                                setAlbumSearchResults([]);
                                                                setAlbumDropdownOpen(false);
                                                                setAlbumSearchError(null);
                                                                setAlbumHasSearched(false);
                                                            }}
                                                        >
                                                            <div className="text-sm font-medium">{formatAlbum(album)}</div>
                                                            {album.disambiguation && (
                                                                <div className={`mt-0.5 text-xs ${dropdownMetaClass}`}>
                                                                    {album.disambiguation}
                                                                </div>
                                                            )}
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </div>

                                    {albumSearchError && <p className="mt-3 text-sm text-amber-700">{albumSearchError}</p>}

                                    {albumHasSearched && !albumSearchResults.length && !albumSearchError && (
                                        <div className={`mt-3 ${emptyClass}`}>No album found for this search.</div>
                                    )}
                                </>
                            )}

                            {inputMode === "listenbrainz" && (
                                <>
                                    <div className={nestedPanelClass}>
                                        <button
                                            type="button"
                                            onClick={() => setActiveLbOptionalOpen((v) => !v)}
                                            className={optionalHeaderClass}
                                        >
                                            <span>Optional parameters</span>
                                            <span className={`text-sm ${subtleTextClass}`}>
                                                {activeLbOptionalOpen ? "Hide" : "Show"}
                                            </span>
                                        </button>

                                        {activeLbOptionalOpen && (
                                            <div className={`grid gap-4 border-t px-4 py-4 ${isDark ? "border-[#342d28]" : "border-[#e4d9cb]"}`}>
                                                <div>
                                                    <label className="mb-2 block text-sm font-semibold">Range</label>
                                                    <select
                                                        className={selectClass}
                                                        value={activeLbParams.range}
                                                        onChange={(e) =>
                                                            setActiveLbParams((prev) => ({ ...prev, range: e.target.value }))
                                                        }
                                                    >
                                                        {LISTENBRAINZ_RANGE_OPTIONS.map((option) => (
                                                            <option key={option.value} value={option.value}>
                                                                {option.label}
                                                            </option>
                                                        ))}
                                                    </select>
                                                </div>

                                                <div>
                                                    <label className="mb-2 block text-sm font-semibold">Minimum listen count</label>
                                                    <input
                                                        type="number"
                                                        min={0}
                                                        max={activeLbParams.max_results}
                                                        value={activeLbParams.min_listen}
                                                        onChange={(e) =>
                                                            setActiveLbParams((prev) => updateLbMinListen(prev, e.target.value))
                                                        }
                                                        onBlur={(e) =>
                                                            setActiveLbParams((prev) => updateLbMinListen(prev, e.target.value))
                                                        }
                                                        className={inputClass}
                                                    />
                                                </div>

                                                <div>
                                                    <label className="mb-2 block text-sm font-semibold">
                                                        Blacklist {lbEntityLabel} listened
                                                    </label>
                                                    <select
                                                        className={selectClass}
                                                        value={activeLbParams.blacklist}
                                                        onChange={(e) =>
                                                            setActiveLbParams((prev) => ({ ...prev, blacklist: e.target.value }))
                                                        }
                                                    >
                                                        <option value="">No blacklist range</option>
                                                        {LISTENBRAINZ_RANGE_OPTIONS.map((option) => (
                                                            <option key={option.value} value={option.value}>
                                                                {option.label}
                                                            </option>
                                                        ))}
                                                    </select>
                                                </div>

                                                <div>
                                                    <label className="mb-2 block text-sm font-semibold">Blacklist minimum listen count</label>
                                                    <input
                                                        type="number"
                                                        min={0}
                                                        max={activeLbParams.max_results}
                                                        value={activeLbParams.blacklist_min}
                                                        onChange={(e) =>
                                                            setActiveLbParams((prev) => updateLbBlacklistMin(prev, e.target.value))
                                                        }
                                                        onBlur={(e) =>
                                                            setActiveLbParams((prev) => updateLbBlacklistMin(prev, e.target.value))
                                                        }
                                                        className={inputClass}
                                                    />
                                                </div>

                                                <div>
                                                    <label className="mb-2 block text-sm font-semibold">Number of recommendations</label>
                                                    <input
                                                        type="number"
                                                        min={NUMBER_INPUT_MIN_COUNT}
                                                        max={NUMBER_INPUT_MAX}
                                                        value={activeLbParams.max_results}
                                                        onChange={(e) =>
                                                            setActiveLbParams((prev) => updateLbMaxResults(prev, e.target.value))
                                                        }
                                                        onBlur={(e) =>
                                                            setActiveLbParams((prev) => updateLbMaxResults(prev, e.target.value))
                                                        }
                                                        className={inputClass}
                                                    />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </>
                            )}
                        </div>

                        {activeTab === "artists" && inputMode === "manual" && (
                            <div className={panelClass}>
                                <h3 className="mb-3 text-base font-bold">Selected artists</h3>

                                {artistSelectedEntries.length === 0 ? (
                                    <div className={emptyClass}>No selected artists.</div>
                                ) : (
                                    <div className="space-y-2">
                                        {artistSelectedEntries.map(([id, item]) => (
                                            <div key={id} className={selectedClass}>
                                                <span className="text-sm font-medium">{item.label}</span>
                                                <button
                                                    className={removeBtnClass}
                                                    onClick={() =>
                                                        setArtistSelected((prev) => {
                                                            const copy = { ...prev };
                                                            delete copy[id];
                                                            return copy;
                                                        })
                                                    }
                                                >
                                                    ✕
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                <div className={nestedPanelClass}>
                                    <button
                                        type="button"
                                        onClick={() => setArtistOptionalOpen((v) => !v)}
                                        className={optionalHeaderClass}
                                    >
                                        <span>Optional parameters</span>
                                        <span className={`text-sm ${subtleTextClass}`}>{artistOptionalOpen ? "Hide" : "Show"}</span>
                                    </button>

                                    {artistOptionalOpen && (
                                        <div className={`border-t px-4 py-4 ${isDark ? "border-[#342d28]" : "border-[#e4d9cb]"}`}>
                                            <label className="block text-sm font-semibold">Blacklist artists</label>

                                            <div className="relative mt-2">
                                                <input
                                                    className={inputClass}
                                                    placeholder="Search artist to blacklist..."
                                                    value={artistBlacklistQuery}
                                                    onChange={(e) => setArtistBlacklistQuery(e.target.value)}
                                                    onKeyDown={handleArtistBlacklistEnterSearch}
                                                    onFocus={() =>
                                                        artistBlacklistSearchResults.length > 0 && setArtistBlacklistDropdownOpen(true)
                                                    }
                                                    onBlur={() => setTimeout(() => setArtistBlacklistDropdownOpen(false), 150)}
                                                />

                                                {artistBlacklistDropdownOpen && artistBlacklistSearchResults.length > 0 && (
                                                    <div className={dropdownClass}>
                                                        {artistBlacklistSearchResults.map((artist, idx) => {
                                                            const id = String(artist.artist_id);

                                                            return (
                                                                <button
                                                                    key={`${id}-${idx}`}
                                                                    type="button"
                                                                    className={dropdownItemClass}
                                                                    onMouseDown={(e) => e.preventDefault()}
                                                                    onClick={() => {
                                                                        setArtistBlacklisted((prev) => ({
                                                                            ...prev,
                                                                            [id]: {
                                                                                label: formatArtist(artist),
                                                                                name: artist.name,
                                                                            },
                                                                        }));

                                                                        setArtistSelected((prev) => {
                                                                            const copy = { ...prev };
                                                                            delete copy[id];
                                                                            return copy;
                                                                        });

                                                                        setArtistBlacklistQuery("");
                                                                        setArtistBlacklistSearchResults([]);
                                                                        setArtistBlacklistDropdownOpen(false);
                                                                        setArtistBlacklistHasSearched(false);
                                                                    }}
                                                                >
                                                                    <div className="text-sm font-medium">{artist.name ?? "Unknown artist"}</div>
                                                                    {artist.disambiguation && (
                                                                        <div className={`mt-0.5 text-xs ${dropdownMetaClass}`}>
                                                                            {artist.disambiguation}
                                                                        </div>
                                                                    )}
                                                                </button>
                                                            );
                                                        })}
                                                    </div>
                                                )}
                                            </div>

                                            {artistBlacklistHasSearched && !artistBlacklistSearchResults.length && (
                                                <div className={`mt-3 ${emptyClass}`}>No artist found for blacklist search.</div>
                                            )}

                                            <div className="mt-4">
                                                {artistBlacklistedEntries.length === 0 ? (
                                                    <div className={emptyClass}>No blacklisted artists.</div>
                                                ) : (
                                                    <div className="space-y-2">
                                                        {artistBlacklistedEntries.map(([id, item]) => (
                                                            <div key={id} className={blacklistClass}>
                                                                <span className={`text-sm font-medium ${isDark ? "text-[#efc1b5]" : "text-[#924d3d]"}`}>
                                                                    {item.label}
                                                                </span>
                                                                <button
                                                                    className={redRemoveBtnClass}
                                                                    onClick={() =>
                                                                        setArtistBlacklisted((prev) => {
                                                                            const copy = { ...prev };
                                                                            delete copy[id];
                                                                            return copy;
                                                                        })
                                                                    }
                                                                >
                                                                    ✕
                                                                </button>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>

                                            <label className="mt-5 block text-sm font-semibold">Number of recommendations</label>
                                            <input
                                                type="number"
                                                min={NUMBER_INPUT_MIN_COUNT}
                                                max={NUMBER_INPUT_MAX}
                                                value={artistTopN}
                                                onChange={(e) => setArtistTopN(clampRecommendationCount(e.target.value))}
                                                onBlur={(e) => setArtistTopN(clampRecommendationCount(e.target.value))}
                                                className={`${inputClass} mt-2`}
                                            />
                                        </div>
                                    )}
                                </div>

                                <button
                                    className={primaryButtonClass}
                                    onClick={fetchArtistRecommendations}
                                    aria-busy={artistLoading}
                                    aria-disabled={artistLoading}
                                >
                                    {renderThinking(artistLoading)}
                                </button>
                            </div>
                        )}

                        {activeTab === "artists" && inputMode === "listenbrainz" && (
                            <div className={panelClass}>
                                <button
                                    className={primaryButtonClass}
                                    onClick={() => {
                                        if (artistLoading) return;
                                        fetchListenBrainzArtistRecommendations();
                                    }}
                                    aria-busy={artistLoading}
                                    aria-disabled={artistLoading}
                                >
                                    {renderThinking(artistLoading)}
                                </button>
                            </div>
                        )}

                        {activeTab === "albums" && inputMode === "manual" && (
                            <div className={panelClass}>
                                <h3 className="mb-3 text-base font-bold">Selected albums</h3>

                                {albumSelectedEntries.length === 0 ? (
                                    <div className={emptyClass}>No selected albums.</div>
                                ) : (
                                    <div className="space-y-2">
                                        {albumSelectedEntries.map(([id, item]) => (
                                            <div key={id} className={selectedClass}>
                                                <span className="text-sm font-medium">{item.label}</span>
                                                <button
                                                    className={removeBtnClass}
                                                    onClick={() =>
                                                        setAlbumSelected((prev) => {
                                                            const copy = { ...prev };
                                                            delete copy[id];
                                                            return copy;
                                                        })
                                                    }
                                                >
                                                    ✕
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                <div className={nestedPanelClass}>
                                    <button
                                        type="button"
                                        onClick={() => setAlbumOptionalOpen((v) => !v)}
                                        className={optionalHeaderClass}
                                    >
                                        <span>Optional parameters</span>
                                        <span className={`text-sm ${subtleTextClass}`}>{albumOptionalOpen ? "Hide" : "Show"}</span>
                                    </button>

                                    {albumOptionalOpen && (
                                        <div className={`border-t px-4 py-4 ${isDark ? "border-[#342d28]" : "border-[#e4d9cb]"}`}>
                                            <label className="block text-sm font-semibold">Blacklist albums</label>

                                            <div className="relative mt-2">
                                                <input
                                                    className={inputClass}
                                                    placeholder="Search album to blacklist..."
                                                    value={albumBlacklistQuery}
                                                    onChange={(e) => setAlbumBlacklistQuery(e.target.value)}
                                                    onKeyDown={handleAlbumBlacklistEnterSearch}
                                                    onFocus={() =>
                                                        albumBlacklistSearchResults.length > 0 && setAlbumBlacklistDropdownOpen(true)
                                                    }
                                                    onBlur={() => setTimeout(() => setAlbumBlacklistDropdownOpen(false), 150)}
                                                />

                                                {albumBlacklistDropdownOpen && albumBlacklistSearchResults.length > 0 && (
                                                    <div className={dropdownClass}>
                                                        {albumBlacklistSearchResults.map((album, idx) => {
                                                            const id = String(album.release_group_id);

                                                            return (
                                                                <button
                                                                    key={`${id}-${idx}`}
                                                                    type="button"
                                                                    className={dropdownItemClass}
                                                                    onMouseDown={(e) => e.preventDefault()}
                                                                    onClick={() => {
                                                                        setAlbumBlacklisted((prev) => ({
                                                                            ...prev,
                                                                            [id]: {
                                                                                label: formatAlbum(album),
                                                                                title: album.title,
                                                                                artist: album.artist,
                                                                            },
                                                                        }));

                                                                        setAlbumSelected((prev) => {
                                                                            const copy = { ...prev };
                                                                            delete copy[id];
                                                                            return copy;
                                                                        });

                                                                        setAlbumBlacklistQuery("");
                                                                        setAlbumBlacklistSearchResults([]);
                                                                        setAlbumBlacklistDropdownOpen(false);
                                                                        setAlbumBlacklistHasSearched(false);
                                                                    }}
                                                                >
                                                                    <div className="text-sm font-medium">{formatAlbum(album)}</div>
                                                                    {album.disambiguation && (
                                                                        <div className={`mt-0.5 text-xs ${dropdownMetaClass}`}>
                                                                            {album.disambiguation}
                                                                        </div>
                                                                    )}
                                                                </button>
                                                            );
                                                        })}
                                                    </div>
                                                )}
                                            </div>

                                            {albumBlacklistHasSearched && !albumBlacklistSearchResults.length && (
                                                <div className={`mt-3 ${emptyClass}`}>No album found for blacklist search.</div>
                                            )}

                                            <div className="mt-4">
                                                {albumBlacklistedEntries.length === 0 ? (
                                                    <div className={emptyClass}>No blacklisted albums.</div>
                                                ) : (
                                                    <div className="space-y-2">
                                                        {albumBlacklistedEntries.map(([id, item]) => (
                                                            <div key={id} className={blacklistClass}>
                                                                <span className={`text-sm font-medium ${isDark ? "text-[#efc1b5]" : "text-[#924d3d]"}`}>
                                                                    {item.label}
                                                                </span>
                                                                <button
                                                                    className={redRemoveBtnClass}
                                                                    onClick={() =>
                                                                        setAlbumBlacklisted((prev) => {
                                                                            const copy = { ...prev };
                                                                            delete copy[id];
                                                                            return copy;
                                                                        })
                                                                    }
                                                                >
                                                                    ✕
                                                                </button>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>

                                            <label className="mt-5 block text-sm font-semibold">Number of recommendations</label>
                                            <input
                                                type="number"
                                                min={NUMBER_INPUT_MIN_COUNT}
                                                max={NUMBER_INPUT_MAX}
                                                value={albumTopN}
                                                onChange={(e) => setAlbumTopN(clampRecommendationCount(e.target.value))}
                                                onBlur={(e) => setAlbumTopN(clampRecommendationCount(e.target.value))}
                                                className={`${inputClass} mt-2`}
                                            />
                                        </div>
                                    )}
                                </div>

                                <button
                                    className={primaryButtonClass}
                                    onClick={fetchAlbumRecommendations}
                                    aria-busy={albumLoading}
                                    aria-disabled={albumLoading}
                                >
                                    {renderThinking(albumLoading)}
                                </button>
                            </div>
                        )}

                        {activeTab === "albums" && inputMode === "listenbrainz" && (
                            <div className={panelClass}>
                                <button
                                    className={primaryButtonClass}
                                    onClick={() => {
                                        if (albumLoading) return;
                                        fetchListenBrainzAlbumRecommendations();
                                    }}
                                    aria-busy={albumLoading}
                                    aria-disabled={albumLoading}
                                >
                                    {renderThinking(albumLoading)}
                                </button>
                            </div>
                        )}
                    </div>

                    <div>{activeTab === "artists" ? renderArtistResults() : renderAlbumResults()}</div>
                </div>
            </div>
        </main>
    );
}
