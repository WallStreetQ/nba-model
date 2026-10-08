"""
nba_boxscores.py
-----------------
Backfills team and player box scores for every finished game in data/games.parquet.

Data source: ESPN `summary?event={game_id}`
  boxscore.teams[]    team totals: FG, 3PT, FT (made-attempted), OREB, DREB, AST, STL, BLK, TO, ...
  boxscore.players[]  every player: starter flag, did-not-play + reason, MIN, PTS, ... , +/-

Team rows give TRUE POSSESSIONS:  FGA - OREB + TOV + 0.44 * FTA   (averaged over both teams)
which the model uses for pace and points per 100 possessions. Player rows (minutes, DNP reasons)
are stored for the availability / injury adjustment (the NBA version of the NHL goalie module).

Run AFTER nba_backfill.py. Resume-friendly; checkpoints every 250 games.

Output: data/game_team_box.parquet, data/game_players.parquet (+ .csv twins)

Usage:
  python nba_boxscores.py              # every finished game missing a box score
  python nba_boxscores.py --limit 500  # cap this run
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from nba_common import (BASE, GAMES_PATH, PLAYERS_PATH, TEAM_BOX_PATH, competitive_only, get, num, read_table,
                        save_both)

TEAM_STATS = {  # ESPN stat name -> our column(s)
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
    "offensiveRebounds": "oreb", "defensiveRebounds": "dreb", "totalRebounds": "reb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "fouls": "pf",
    "turnovers": "tov_player", "totalTurnovers": "tov",
    "pointsInPaint": "pts_paint", "fastBreakPoints": "pts_fastbreak", "turnoverPoints": "pts_off_tov",
}
TEAM_COLS = ["game_id", "team", "opponent", "is_home", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb",
             "dreb", "reb", "ast", "stl", "blk", "pf", "tov", "pts_paint", "pts_fastbreak", "pts_off_tov"]
PLAYER_COLS = ["game_id", "team", "player_id", "name", "position", "starter", "did_not_play", "reason",
               "minutes", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast",
               "stl", "blk", "tov", "pf", "plus_minus"]
PLAYER_KEYS = {
    "minutes": "minutes", "points": "pts", "rebounds": "reb", "assists": "ast", "turnovers": "tov",
    "steals": "stl", "blocks": "blk", "offensiveRebounds": "oreb", "defensiveRebounds": "dreb",
    "fouls": "pf", "plusMinus": "plus_minus",
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
}


def _made_att(v):
    try:
        a, b = str(v).split("-")
        return float(a), float(b)
    except ValueError:
        return None, None


def parse_summary(game_id: str, js: dict) -> tuple[list[dict], list[dict]]:
    box = (js or {}).get("boxscore") or {}
    teams = box.get("teams") or []
    if len(teams) != 2:
        return [], []
    abbr = [t["team"].get("abbreviation") for t in teams]
    team_rows = []
    for i, t in enumerate(teams):
        row = {"game_id": game_id, "team": abbr[i], "opponent": abbr[1 - i], "is_home": t.get("homeAway") == "home"}
        for s in t.get("statistics", []):
            col = TEAM_STATS.get(s.get("name"))
            if isinstance(col, tuple):
                row[col[0]], row[col[1]] = _made_att(s.get("displayValue"))
            elif col:
                row[col] = num(s.get("displayValue"))
        if row.get("tov") is None:
            row["tov"] = row.get("tov_player")
        team_rows.append({c: row.get(c) for c in TEAM_COLS})

    player_rows = []
    for p in box.get("players") or []:
        team = (p.get("team") or {}).get("abbreviation")
        for block in p.get("statistics") or []:
            keys = block.get("keys") or block.get("names") or []
            for ath in block.get("athletes") or []:
                a = ath.get("athlete") or {}
                row = {"game_id": game_id, "team": team, "player_id": str(a.get("id")), "name": a.get("displayName"),
                       "position": (a.get("position") or {}).get("abbreviation"), "starter": bool(ath.get("starter")),
                       "did_not_play": bool(ath.get("didNotPlay")), "reason": ath.get("reason")}
                for k, v in zip(keys, ath.get("stats") or []):
                    col = PLAYER_KEYS.get(k)
                    if isinstance(col, tuple):
                        row[col[0]], row[col[1]] = _made_att(v)
                    elif col:
                        row[col] = num(v)
                player_rows.append({c: row.get(c) for c in PLAYER_COLS})
    return team_rows, player_rows


def fetch(game_id: str):
    return parse_summary(game_id, get(f"{BASE}/summary", {"event": game_id}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max games to pull this run")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    games = competitive_only(read_table(GAMES_PATH))
    fin = games[games["completed"].astype(str).str.lower() == "true"]
    teams_old = read_table(TEAM_BOX_PATH, required=False)
    players_old = read_table(PLAYERS_PATH, required=False)
    have = set(teams_old["game_id"]) if teams_old is not None else set()
    todo = [g for g in fin.sort_values("game_date")["game_id"] if g not in have]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Pulling box scores for {len(todo)} games ({len(have)} already done)...")

    t_frames = [teams_old] if teams_old is not None else []
    p_frames = [players_old] if players_old is not None else []
    for start in range(0, len(todo), 250):
        chunk = todo[start:start + 250]
        t_rows, p_rows = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for tr, pr in pool.map(fetch, chunk):
                t_rows.extend(tr)
                p_rows.extend(pr)
        t_frames.append(pd.DataFrame(t_rows, columns=TEAM_COLS))
        p_frames.append(pd.DataFrame(p_rows, columns=PLAYER_COLS))
        tb = pd.concat(t_frames, ignore_index=True).drop_duplicates(["game_id", "team"], keep="last")
        pb = pd.concat(p_frames, ignore_index=True).drop_duplicates(["game_id", "player_id"], keep="last")
        save_both(tb, TEAM_BOX_PATH)
        save_both(pb, PLAYERS_PATH)
        t_frames, p_frames = [tb], [pb]
        print(f"  {min(start + 250, len(todo))}/{len(todo)} games processed")
    print("Done.")


if __name__ == "__main__":
    main()
