import os
import time
import random
import re
import urllib.parse
import requests
import pandas as pd

session = requests.Session()

def clean_text(x):
    if not isinstance(x, str):
        return ""
    x = x.lower().strip()
    x = re.sub(r"\([^)]*\)", "", x)   # remove anything in ()
    x = re.sub(r"\[[^]]*\]", "", x)   # removes anything in []
    x = re.sub(r"feat\..*", "", x)    # remove feat. and everything after
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def apple_music_search_safe(title, artist, country="US", max_attempts=8):
    """
    Grabs download url from Apple iTunes Search API for a track.
    Returns dict with preview_url and other metadata.
    """
    query = f"{title} {artist}".strip()
    if not query:
        return None

    params = {
        "term": query,
        "entity": "song",
        "limit": 5,
        "country": "US"
    }
    url = "https://itunes.apple.com/search"

    for attempt in range(max_attempts):
        try:
            resp = session.get(url, params=params, timeout=10)
        except requests.RequestException as e:
            sleep_t = min(300, (2 ** attempt) + random.uniform(0, 1)) # exponential backoff (max 300 seconds)
            print(f"[NETWORK ERROR] {e}. Sleeping {sleep_t:.2f}s then retrying...")
            time.sleep(sleep_t)
            continue

        if resp.status_code == 200:
            data = resp.json()
            if data.get("resultCount", 0) == 0:
                return None
            best = data["results"][0]
            return {
                "preview_url": best.get("previewUrl"),
                "track_name": best.get("trackName"),
                "artist_name": best.get("artistName"),
                "collection_name": best.get("collectionName"),
                "track_id": best.get("trackId"),
                "track_view_url": best.get("trackViewUrl")
            }

        if resp.status_code in (403, 429, 503):
            sleep_t = min(300, (2 ** attempt) + random.uniform(0, 1))  # exponential backoff (max 300 seconds)
            print(f"[RATE LIMIT {resp.status_code}] Sleeping {sleep_t:.2f}s then retrying...")
            time.sleep(sleep_t)
            continue

        # give up for this song if different error
        print(f"[HTTP {resp.status_code}] {resp.text[:200]}")
        return None

    print("Gave up on ", query)
    return None


def save_progress(matches_df):
    """Save current matches df + preview text file."""
    print(f"[SAVE] Saving {len(matches_df)} rows to {"apple_music_matches.csv"} and {"apple_previews.txt"}")
    matches_df.to_csv("apple_music_matches.csv", index=False)

    valid = matches_df.dropna(subset=["preview_url"])
    with open("apple_previews.txt", "w", encoding="utf-8") as f:
        for _, r in valid.iterrows():
            sid = int(r["song_id"])
            url = r["preview_url"]
            f.write(f"{sid}|{url}\n")



def main():
    if not os.path.exists("merged_df_with_ids.csv"):
        raise FileNotFoundError(f"Could not find {"merged_df_with_ids.csv"} in current directory.")

    merged_df = pd.read_csv("merged_df_with_ids.csv")

    if "song_id" not in merged_df.columns:
        merged_df = merged_df.reset_index(drop=True)
        merged_df["song_id"] = merged_df.index

    if "title" in merged_df.columns:
        merged_df["clean_title"] = merged_df["title"].apply(clean_text)
    elif "title_norm" in merged_df.columns:
        merged_df["clean_title"] = merged_df["title_norm"].apply(clean_text)
    else:
        raise ValueError("Need a 'title' or 'title_norm' column in merged_df.")

    if "artist_list" in merged_df.columns:
        def get_primary_artist(row):
            al = row["artist_list"]
            if isinstance(al, str):
                return clean_text(al.split(";")[0])
            if isinstance(al, list) and al:
                return clean_text(al[0])
            if "artist" in row and isinstance(row["artist"], str):
                return clean_text(row["artist"])
            return ""
        merged_df["clean_artist"] = merged_df.apply(get_primary_artist, axis=1)
    elif "artist" in merged_df.columns:
        merged_df["clean_artist"] = merged_df["artist"].apply(clean_text)
    elif "artists" in merged_df.columns:
        merged_df["clean_artist"] = merged_df["artists"].apply(clean_text)
    else:
        raise ValueError("Need an 'artist', 'artists', or 'artist_list' column.")

    if os.path.exists("apple_music_matches.csv"):
        matches_df = pd.read_csv("apple_music_matches.csv")
        done_ids = set(matches_df["song_id"].tolist())
        print(f"[RESUME] Found existing {len(done_ids)} matched songs. Will skip those.")
    else:
        matches_df = pd.DataFrame(columns=[
            "song_id",
            "query_title",
            "query_artist",
            "preview_url",
            "track_name",
            "artist_name",
            "collection_name",
            "track_id",
            "track_view_url"
        ])
        done_ids = set()

    rows = []
    if not matches_df.empty:
        rows = matches_df.to_dict("records")

    total = len(merged_df)
    print(f"[INFO] Total songs in merged_df: {total}")

    for idx, row in merged_df.iterrows():
        song_id = int(row["song_id"])
        if song_id in done_ids:
            if idx % 500 == 0:
                print(f"[SKIP] Already have song_id={song_id}, idx={idx}/{total}")
            continue

        title = row["clean_title"]
        artist = row["clean_artist"]

        if not title or not artist:
            info = {
                "preview_url": None,
                "track_name": None,
                "artist_name": None,
                "collection_name": None,
                "track_id": None,
                "track_view_url": None
            }
        else:
            info = apple_music_search_safe(title, artist)

        if info is None:
            info = {
                "preview_url": None,
                "track_name": None,
                "artist_name": None,
                "collection_name": None,
                "track_id": None,
                "track_view_url": None
            }

        record = {
            "song_id": song_id,
            "query_title": title,
            "query_artist": artist,
            **info
        }
        rows.append(record)
        done_ids.add(song_id)

        if len(rows) % 100 == 0: # save every 100 songs
            matches_df = pd.DataFrame(rows)
            save_progress(matches_df)

        if idx % 50 == 0:
            print(f"[PROGRESS] idx={idx}/{total}, collected={len(rows)}")

        sleep_t = 1.3 + random.uniform(0, 0.4) # 1.3 seconds between requests (about 50 req per min)
        time.sleep(sleep_t)

    matches_df = pd.DataFrame(rows)
    save_progress(matches_df)
    print("[DONE] All songs processed.")


if __name__ == "__main__":
    main()
