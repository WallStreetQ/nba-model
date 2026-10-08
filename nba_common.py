"""
nba_common.py
--------------
Shared paths, HTTP helper and ESPN scoreboard parsing used by every data script.

Data source: ESPN's public NBA endpoints (no key needed). They work from GitHub Actions and
Streamlit Cloud, unlike stats.nba.com which blocks most cloud IPs.
  scoreboard  /scoreboard?dates=YYYYMMDD   schedule, final + quarter scores, posted line
  summary     /summary?event={game_id}      full box score (team + player), injuries
"""

import datetime as dt
import re
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
GAMES_PATH = DATA_DIR / "games.parquet"
TEAM_BOX_PATH = DATA_DIR / "game_team_box.parquet"
PLAYERS_PATH = DATA_DIR / "game_players.parquet"
UPCOMING_PATH = DATA_DIR / "upcoming.json"
ODDS_PATH = DATA_DIR / "odds.json"
INJURIES_PATH = DATA_DIR / "injuries.json"
SNAPSHOTS_PATH = DATA_DIR / "injury_snapshots.parquet"
TZ = ZoneInfo("America/New_York")

HEADERS = {"User-Agent": "Mozilla/5.0 (nba-model)"}
VALID_SEASON_TYPES = {2, 3, 5}  # regular season, playoffs, play-in.  1 = preseason (never used)
OFFSEASON_MONTHS = {7, 8, 9}     # summer league / exhibitions only - no competitive NBA games

# ESPN abbreviations of the 30 franchises (drops All-Star and exhibition games)
NBA_TEAMS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SA", "TOR", "UTAH", "WSH",
}
TEAM_NAME_TO_ABBR = {  # full team names (Odds API, NBA injury report) -> ESPN abbreviations
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO",
    "New York Knicks": "NY", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SA",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTAH", "Washington Wizards": "WSH",
}
# Line columns kept from ESPN, never overwritten by a later (post-game) fetch that lacks them
LINE_COLS = ["total_line", "spread_home", "ml_home", "ml_away", "odds_provider"]


def today_et() -> dt.date:
    return dt.datetime.now(TZ).date()


def season_of(d) -> int:
    """ESPN labels a season by the year it ends: 2026-27 -> 2027."""
    d = pd.Timestamp(d)
    return d.year + 1 if d.month >= 7 else d.year


def season_ids(n: int, today: dt.date | None = None) -> list[int]:
    """The last n seasons up to and including the current one."""
    cur = season_of(today or today_et())
    return list(range(cur - n + 1, cur + 1))


def get(url, params=None, retries=3, pause=1.5):
    """GET JSON with retries. Returns None on 404 or repeated failure."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                print(f"  [warn] {url} {params or ''}: {e}")
                return None
            time.sleep(pause * (attempt + 1))
    return None


def save_both(df: pd.DataFrame, path: Path):
    """Parquet (committed, read by the pipeline) + CSV twin (for eyeballing in VS Code; git-ignored)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)
    print(f"  saved {len(df):,} rows -> {path.relative_to(ROOT)}")


def read_table(path: Path, required=True):
    for p, reader in ((path, pd.read_parquet), (path.with_suffix(".csv"), pd.read_csv)):
        if p.exists():
            try:
                df = reader(p)
            except ImportError:
                continue
            if "game_id" in df:
                df["game_id"] = df["game_id"].astype(str)
            return df
    if required:
        raise FileNotFoundError(f"{path.name} not found - run nba_backfill.py first")
    return None


# ------------------------------------------------------------------ scoreboard parsing
def num(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.upper() in {"EVEN", "EV"}:
        return 100.0
    s = re.sub(r"[^0-9.+\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _deep(d, path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _parse_odds(comp, home, away):
    out = {c: None for c in LINE_COLS}
    odds = comp.get("odds") or []
    if not odds:
        return out
    o = odds[0]
    out["odds_provider"] = (o.get("provider") or {}).get("name")
    out["total_line"] = num(o.get("overUnder")) or num(_deep(o, ["total", "over", "close", "line"]))
    details = (o.get("details") or "").strip()
    m = re.match(r"^([A-Za-z]+)\s+([+-]?\d+(?:\.\d+)?)$", details)
    if m:
        abbr, val = m.group(1).upper(), float(m.group(2))
        out["spread_home"] = val if abbr == home else (-val if abbr == away else None)
    elif details.upper() in {"EVEN", "PK", "PICK"}:
        out["spread_home"] = 0.0
    if out["spread_home"] is None:
        out["spread_home"] = num(_deep(o, ["pointSpread", "home", "close", "line"]))
    for side, key in (("home", "homeTeamOdds"), ("away", "awayTeamOdds")):
        ml = (o.get(key) or {}).get("moneyLine")
        if ml is None:
            ml = _deep(o, ["moneyline", side, "close", "odds"]) or _deep(o, ["moneyline", side, "open", "odds"])
        out[f"ml_{side}"] = num(ml)
    return out


def is_competitive(season_type, slug, day) -> bool:
    """Regular season, play-in or playoffs only. Preseason, summer league, All-Star and exhibitions
    never enter the data, so they can't touch ratings, player values or the track record."""
    slug = (slug or "").lower()
    return (season_type in VALID_SEASON_TYPES and "preseason" not in slug and "summer" not in slug
            and pd.Timestamp(day).month not in OFFSEASON_MONTHS)


def competitive_only(games: pd.DataFrame) -> pd.DataFrame:
    """Same rule applied to a stored games table (cleans anything saved by older code)."""
    if games is None or games.empty:
        return games
    gt = pd.to_numeric(games["game_type"], errors="coerce")
    month = pd.to_datetime(games["game_date"]).dt.month
    keep = gt.isin(VALID_SEASON_TYPES) & ~month.isin(OFFSEASON_MONTHS) & games["home_team"].isin(NBA_TEAMS) \
        & games["away_team"].isin(NBA_TEAMS)
    return games[keep]


def parse_scoreboard(js: dict, day: dt.date) -> list[dict]:
    rows = []
    for ev in (js or {}).get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in sides or "away" not in sides:
            continue
        h, a = sides["home"], sides["away"]
        ha, aa = h["team"].get("abbreviation"), a["team"].get("abbreviation")
        season = ev.get("season") or {}
        if ha not in NBA_TEAMS or aa not in NBA_TEAMS or not is_competitive(season.get("type"), season.get("slug"), day):
            continue
        status = (comp.get("status") or ev.get("status") or {}).get("type", {})
        row = {
            "game_id": str(ev["id"]), "game_date": day.isoformat(), "start_utc": ev.get("date"),
            "season": season.get("year"), "game_type": season.get("type"),
            "game_state": status.get("state"), "completed": bool(status.get("completed")),
            "status_detail": status.get("shortDetail"), "neutral": bool(comp.get("neutralSite")),
            "home_team": ha, "away_team": aa,
        }
        n_periods = 0
        for side, c in (("home", h), ("away", a)):
            ls = [(num(x.get("value")) or 0.0) for x in (c.get("linescores") or [])]
            n_periods = max(n_periods, len(ls))
            row[f"{side}_score"] = num(c.get("score"))
            for i in range(4):
                row[f"{side}_q{i + 1}"] = ls[i] if len(ls) > i else None
            row[f"{side}_ot"] = float(sum(ls[4:])) if len(ls) > 4 else 0.0
        row["n_periods"] = n_periods
        row.update(_parse_odds(comp, ha, aa))
        rows.append(row)
    return rows


def fetch_scoreboard(day: dt.date) -> list[dict]:
    return parse_scoreboard(get(f"{BASE}/scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": 100}), day)


def upsert_games(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Newer rows win, except pre-game lines: ESPN drops them after tip-off, so keep what we saw."""
    if old is None or old.empty:
        return new.reset_index(drop=True)
    if new.empty:
        return old
    new = new.set_index("game_id")
    prev = old.set_index("game_id").reindex(new.index)
    for c in LINE_COLS:
        if c in prev:
            new[c] = new[c].where(new[c].notna(), prev[c])
    out = pd.concat([old[~old["game_id"].isin(new.index)], new.reset_index()], ignore_index=True)
    return out.sort_values(["game_date", "game_id"]).reset_index(drop=True)
