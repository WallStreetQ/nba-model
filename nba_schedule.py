"""
nba_schedule.py
----------------
Pulls not-yet-started games for today and the next few days, with ESPN's posted line
(spread, total, moneyline) when available, into data/upcoming.json. refresh.py feeds this into
the model artifact so the app shows the slate, flags back-to-backs and logs pre-game predictions.

Usage:
  python nba_schedule.py            # today + next 3 days
  python nba_schedule.py --days 7
"""

import argparse
import datetime as dt
import json

from nba_common import UPCOMING_PATH, fetch_scoreboard, today_et

KEEP = ["game_id", "game_date", "start_utc", "game_type", "neutral", "home_team", "away_team",
        "total_line", "spread_home", "ml_home", "ml_away", "odds_provider"]


def pull_upcoming(days: int = 4, start: dt.date | None = None) -> list[dict]:
    start = start or today_et()
    out = []
    for i in range(days):
        for g in fetch_scoreboard(start + dt.timedelta(days=i)):
            if g["game_state"] == "pre":
                out.append({k: g.get(k) for k in KEEP})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=4, help="how many days ahead, including today")
    args = ap.parse_args()
    games = pull_upcoming(args.days)
    UPCOMING_PATH.parent.mkdir(exist_ok=True)
    UPCOMING_PATH.write_text(json.dumps(games, indent=1))
    print(f"  saved {len(games)} upcoming games -> {UPCOMING_PATH.name}")


if __name__ == "__main__":
    main()
