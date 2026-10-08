"""
nba_update.py
--------------
Refreshes games.parquet with anything that finished since the last run (plus a few days back
for late or postponed games). Pre-game lines already captured are kept.

Usage:
  python nba_update.py               # sweep 2 days back through today
  python nba_update.py --days-back 5
"""

import argparse
import datetime as dt

import pandas as pd

from nba_common import GAMES_PATH, fetch_scoreboard, read_table, save_both, today_et, upsert_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-back", type=int, default=2)
    args = ap.parse_args()

    games = read_table(GAMES_PATH)
    done = games[games["completed"].astype(str).str.lower() == "true"]
    last = pd.to_datetime(done["game_date"]).max().date() if not done.empty else today_et()
    start = min(last, today_et()) - dt.timedelta(days=args.days_back)
    days = [start + dt.timedelta(d) for d in range((today_et() - start).days + 1)]
    rows = [r for d in days for r in fetch_scoreboard(d)]
    games = upsert_games(games, pd.DataFrame(rows))
    newly = games[games["completed"].astype(str).str.lower() == "true"]
    print(f"  {start} -> {today_et()}: {len(rows)} games seen | {len(newly) - len(done)} newly final")
    save_both(games, GAMES_PATH)


if __name__ == "__main__":
    main()
