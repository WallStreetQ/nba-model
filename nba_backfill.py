"""
nba_backfill.py
----------------
Pulls every NBA game (regular season, play-in, playoffs) for the last N seasons from ESPN's
daily scoreboards: date, teams, final score, quarter scores, OT points, posted line.

Resume-friendly: dates whose games are all final in data/games.parquet are skipped.

Output: data/games.parquet / .csv

Usage:
  python nba_backfill.py              # last 4 seasons incl. the current one (3 complete + current)
  python nba_backfill.py --seasons 5
"""

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from nba_common import GAMES_PATH, fetch_scoreboard, read_table, save_both, season_ids, today_et, upsert_games


def season_days(season: int, until: dt.date) -> list[dt.date]:
    start, end = dt.date(season - 1, 10, 1), min(dt.date(season, 6, 30), until)
    return [start + dt.timedelta(d) for d in range((end - start).days + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, default=4, help="how many seasons back, including the current one")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (be polite: 4 is plenty)")
    args = ap.parse_args()

    until = today_et() + dt.timedelta(days=1)
    existing = read_table(GAMES_PATH, required=False)
    done_dates = set()
    if existing is not None and not existing.empty:
        by_day = existing.groupby("game_date")["completed"].apply(lambda s: s.astype(str).str.lower().eq("true").all())
        done_dates = set(by_day[by_day].index)

    for season in season_ids(args.seasons):
        days = [d for d in season_days(season, until) if d.isoformat() not in done_dates]
        if not days:
            print(f"Season {season - 1}-{str(season)[-2:]}: already complete")
            continue
        print(f"Season {season - 1}-{str(season)[-2:]}: fetching {len(days)} dates ...")
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for day_rows in pool.map(fetch_scoreboard, days):
                rows.extend(day_rows)
        new = pd.DataFrame(rows)
        print(f"  {len(new):,} games")
        existing = upsert_games(existing, new)
        save_both(existing, GAMES_PATH)  # checkpoint after every season
    print("Done.")


if __name__ == "__main__":
    main()
