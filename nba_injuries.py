"""
nba_injuries.py
----------------
Who is available tonight. Writes data/injuries.json and appends to data/injury_snapshots.parquet.

Sources (all free):
  1) ESPN injuries   /injuries                  status (Out / Day-To-Day), injury, expected return
  2) ESPN rosters    /teams/{abbr}/roster       current rosters (catches trades and signings,
                                                incl. the off-season, before a team has played)
  3) Official NBA injury report (optional)      PDF published through the day; adds the finer
     https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH_MMAM.pdf
                                                statuses Out / Doubtful / Questionable / Probable.
     Needs `pdfplumber`; skipped quietly if it's missing or the PDF can't be parsed.

ESPN athlete ids are the same ids used in the box scores, so players join cleanly.

Snapshots: every run stores each listed player's status for today's date. Once games finish, the
model learns how often players with each status actually played (data beats guessed rates).

Usage:
  python nba_injuries.py
  python nba_injuries.py --no-official     # ESPN only
"""

import argparse
import datetime as dt
import io
import json
import re
import unicodedata

import pandas as pd
import requests

from nba_common import (BASE, HEADERS, INJURIES_PATH, NBA_TEAMS, SNAPSHOTS_PATH, TEAM_NAME_TO_ABBR, TZ, get,
                        read_table, save_both, today_et)

REPORT_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{time}.pdf"
OFFICIAL_STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}


def norm_name(name: str) -> str:
    """'Antetokounmpo, Giannis' / 'Giannis Antetokounmpo' / 'Nikola Jokić' -> comparable key."""
    if not name:
        return ""
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first} {last}"
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z ]", "", name.lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ------------------------------------------------------------------ ESPN
def _athlete_id(a: dict):
    if a.get("id"):
        return str(a["id"])
    for link in a.get("links") or []:
        m = re.search(r"/id/(\d+)", link.get("href", ""))
        if m:
            return m.group(1)
    return None


def fetch_rosters() -> dict:
    rosters = {}
    for team in sorted(NBA_TEAMS):
        js = get(f"{BASE}/teams/{team.lower()}/roster")
        athletes = (js or {}).get("athletes") or []
        if athletes and isinstance(athletes[0], dict) and "items" in athletes[0]:   # grouped by position
            athletes = [a for grp in athletes for a in grp.get("items", [])]
        rosters[team] = [{"player_id": str(a.get("id")), "name": a.get("displayName") or a.get("fullName"),
                          "position": (a.get("position") or {}).get("abbreviation")} for a in athletes if a.get("id")]
    return rosters


def fetch_espn_injuries(rosters: dict) -> list[dict]:
    js = get(f"{BASE}/injuries") or {}
    by_name = {(t, norm_name(p["name"])): p["player_id"] for t, ps in rosters.items() for p in ps}
    rows = []
    for team in js.get("injuries", []):
        abbr = TEAM_NAME_TO_ABBR.get(team.get("displayName"))
        for inj in team.get("injuries", []):
            a = inj.get("athlete") or {}
            abbr_ = abbr or TEAM_NAME_TO_ABBR.get(((a.get("team") or {}).get("displayName")))
            name = a.get("displayName") or " ".join(x for x in (a.get("firstName"), a.get("lastName")) if x)
            pid = _athlete_id(a) or by_name.get((abbr_, norm_name(name)))
            d = inj.get("details") or {}
            rows.append({"team": abbr_, "player_id": pid, "name": name, "espn_status": inj.get("status"),
                         "detail": " ".join(x for x in (d.get("side"), d.get("location"), d.get("type")) if x) or None,
                         "return_date": d.get("returnDate"), "comment": inj.get("shortComment")})
    return [r for r in rows if r["team"]]


# ------------------------------------------------------------------ official NBA report (optional)
def _report_candidates(now: dt.datetime, hours_back=10):
    """Newest first. Current filenames use 15-minute stamps (10_15AM); older ones are hourly (10AM)."""
    t = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    for i in range(hours_back * 4):
        x = t - dt.timedelta(minutes=15 * i)
        yield x, x.strftime("%I_%M%p")
        if x.minute == 30:            # the old scheme labelled the half-past report by its hour
            yield x, x.strftime("%I%p")


def parse_report_lines(lines: list[str]) -> list[dict]:
    """Rows of (team, player, status, reason) from the report's text lines.
    Team and matchup cells only appear on the first row of a group, so they're carried forward."""
    team_names = sorted(TEAM_NAME_TO_ABBR, key=len, reverse=True)
    rows, team = [], None
    status_re = re.compile(r"\b(Out|Doubtful|Questionable|Probable|Available)\b")
    player_re = re.compile(r"([A-Z][A-Za-z'.\-]+(?: [A-Z][A-Za-z'.\-]+)*(?: (?:Jr\.|Sr\.|II|III|IV))?, "
                           r"[A-Z][A-Za-z'.\-]+(?: [A-Z][A-Za-z'.\-]+)*)")
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("Injury Report", "Page ", "Game Date")):
            continue
        for tn in team_names:
            if tn in line:
                team = TEAM_NAME_TO_ABBR[tn]
                line = line.split(tn, 1)[1].strip()
                break
        if team is None or "NOT YET SUBMITTED" in line:
            continue
        sm = status_re.search(line)
        pm = player_re.search(line[: sm.start()] if sm else line)
        if sm and pm:
            rows.append({"team": team, "name": pm.group(1), "official_status": sm.group(1),
                         "reason": line[sm.end():].strip() or None})
    return rows


def fetch_official(now: dt.datetime | None = None):
    try:
        import pdfplumber
    except ImportError:
        print("  official report: skipped (pip install pdfplumber to enable)")
        return None, []
    now = now or dt.datetime.now(TZ)
    for stamp, label in _report_candidates(now):
        url = REPORT_URL.format(date=stamp.strftime("%Y-%m-%d"), time=label)
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
        except requests.RequestException:
            continue
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            continue
        try:
            with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                lines = [ln for page in pdf.pages for ln in (page.extract_text() or "").splitlines()]
            rows = parse_report_lines(lines)
            print(f"  official report {stamp:%Y-%m-%d %I:%M %p}: {len(rows)} players listed")
            return stamp.isoformat(), rows
        except Exception as e:  # noqa: BLE001 - a parsing problem must never stop the refresh
            print(f"  [warn] official report {label} could not be parsed ({e}) - using ESPN only")
            return None, []
    print("  official report: none found in the last 10 hours")
    return None, []


# ------------------------------------------------------------------ merge + save
def merge(espn_rows, official_rows, rosters):
    """One row per listed player. Official status wins when the player is matched by team + name."""
    off = {(r["team"], norm_name(r["name"])): r for r in official_rows}
    by_name = {(t, norm_name(p["name"])): p for t, ps in rosters.items() for p in ps}
    out, seen = [], set()
    for r in espn_rows:
        key = (r["team"], norm_name(r["name"]))
        o = off.get(key)
        status = o["official_status"] if o else r["espn_status"]
        out.append({**r, "status": status, "official_status": o["official_status"] if o else None,
                    "reason": o["reason"] if o else r.get("detail")})
        seen.add(key)
    for key, o in off.items():          # on the official report but not on ESPN's list
        if key in seen or o["official_status"] == "Available":
            continue
        p = by_name.get(key)
        out.append({"team": o["team"], "player_id": p["player_id"] if p else None,
                    "name": p["name"] if p else norm_name(o["name"]).title(), "espn_status": None,
                    "detail": None, "return_date": None, "comment": None, "status": o["official_status"],
                    "official_status": o["official_status"], "reason": o["reason"]})
    return out


def save_snapshot(players, fetched_at):
    snap = pd.DataFrame([{"date": today_et().isoformat(), "team": p["team"], "player_id": p["player_id"],
                          "name": p["name"], "status": p["status"], "fetched_at": fetched_at}
                         for p in players if p.get("player_id")])
    if snap.empty:
        return
    old = read_table(SNAPSHOTS_PATH, required=False)
    allp = pd.concat([old, snap], ignore_index=True) if old is not None else snap
    allp = allp.sort_values("fetched_at").drop_duplicates(["date", "player_id"], keep="last")
    save_both(allp.reset_index(drop=True), SNAPSHOTS_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-official", action="store_true", help="skip the official NBA PDF report")
    args = ap.parse_args()
    fetched_at = dt.datetime.now(TZ).isoformat(timespec="seconds")
    rosters = fetch_rosters()
    n_roster = sum(len(v) for v in rosters.values())
    espn = fetch_espn_injuries(rosters)
    stamp, official = (None, []) if args.no_official else fetch_official()
    players = merge(espn, official, rosters)
    INJURIES_PATH.write_text(json.dumps({"fetched_at": fetched_at, "official_report": stamp,
                                         "players": players, "rosters": rosters}, indent=1))
    print(f"  {n_roster} roster players | {len(players)} listed ({len(espn)} ESPN, {len(official)} official) "
          f"-> {INJURIES_PATH.name}")
    save_snapshot(players, fetched_at)


if __name__ == "__main__":
    main()
