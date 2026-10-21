"""
nba_odds.py
------------
OPTIONAL sportsbook lines from The Odds API (https://the-odds-api.com), saved to data/odds.json.
Needs ODDS_API_KEY in the environment; without it this script does nothing and the model uses
ESPN's posted line (spread, total, moneyline only).

  featured markets (moneyline, spread, total) for every upcoming game: 3 credits per run
  --period: team totals + 1H/2H totals for TODAY's games: ~3 credits per game

Usage:
  python nba_odds.py            # featured markets
  python nba_odds.py --period   # + team totals and 1H/2H totals
"""

import argparse
import datetime as dt
import json
import os

import requests

from nba_common import ODDS_PATH, TEAM_NAME_TO_ABBR as NAME_TO_ABBR, TZ, today_et

API = "https://api.the-odds-api.com/v4/sports/basketball_nba"
BOOK_PRIORITY = ["hardrock", "fanduel", "betmgm", "caesars", "espnbet", "betrivers", "bovada"]


def _get(url, key, **params):
    r = requests.get(url, params={"apiKey": key, "regions": "us", "oddsFormat": "american", **params}, timeout=20)
    r.raise_for_status()
    print(f"  odds api: {r.headers.get('x-requests-remaining', '?')} credits left")
    return r.json()


def _market(books, key):
    by = {b["key"]: b for b in books}
    for k in [k for k in BOOK_PRIORITY if k in by] + [k for k in by if k not in BOOK_PRIORITY]:
        for m in by[k].get("markets", []):
            if m["key"] == key and m.get("outcomes"):
                return k, m["outcomes"]
    return None, None


def _ou(outcomes, desc=None):
    line = over = under = None
    for o in outcomes or []:
        if desc and o.get("description") != desc:
            continue
        if o.get("name") == "Over":
            line, over = o.get("point"), o.get("price")
        elif o.get("name") == "Under":
            under, line = o.get("price"), line if line is not None else o.get("point")
    return line, over, under


def parse_event(ev):
    hf, af = ev["home_team"], ev["away_team"]
    start = dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).astimezone(TZ)
    row = {"home_team": NAME_TO_ABBR.get(hf), "away_team": NAME_TO_ABBR.get(af), "game_date": start.date().isoformat()}
    books = ev.get("bookmakers", [])
    book, outs = _market(books, "h2h")
    if outs:
        p = {o["name"]: o["price"] for o in outs}
        row.update(book=book, ml_home=p.get(hf), ml_away=p.get(af))
    _, outs = _market(books, "spreads")
    if outs:
        d = {o["name"]: o for o in outs}
        if hf in d:
            row.update(spread_home=d[hf].get("point"), spread_home_price=d[hf].get("price"))
        if af in d:
            row["spread_away_price"] = d[af].get("price")
    for key, pre in (("totals", "total"), ("totals_h1", "h1_total"), ("totals_h2", "h2_total")):
        _, outs = _market(books, key)
        if outs:
            line, o, u = _ou(outs)
            row.update({f"{pre}_line": line, f"{pre}_over_price": o, f"{pre}_under_price": u})
    _, outs = _market(books, "team_totals")
    if outs:
        for side, full in (("home", hf), ("away", af)):
            line, o, u = _ou(outs, full)
            row.update({f"{side}_tt_line": line, f"{side}_tt_over_price": o, f"{side}_tt_under_price": u})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", action="store_true", help="also team totals + 1H/2H totals for today's games")
    args = ap.parse_args()
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("  ODDS_API_KEY not set - skipping (ESPN lines will be used)")
        return
    try:
        events = _get(f"{API}/odds", key, markets="h2h,spreads,totals")
    except requests.RequestException as e:
        print(f"  [warn] Odds API unavailable ({e}) - keeping ESPN lines")
        return
    rows = []
    for ev in events:
        if args.period:
            start = dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).astimezone(TZ).date()
            if start == today_et():
                try:
                    extra = _get(f"{API}/events/{ev['id']}/odds", key, markets="team_totals,totals_h1,totals_h2")
                    have = {b["key"]: b for b in ev.get("bookmakers", [])}
                    for b in extra.get("bookmakers", []):
                        if b["key"] in have:
                            have[b["key"]]["markets"] += b.get("markets", [])
                        else:
                            ev.setdefault("bookmakers", []).append(b)
                except requests.HTTPError:
                    pass
        rows.append(parse_event(ev))
    rows = [r for r in rows if r["home_team"] and r["away_team"]]
    ODDS_PATH.write_text(json.dumps({"fetched_at": dt.datetime.now(TZ).isoformat(timespec="seconds"), "games": rows}, indent=1))
    print(f"  saved lines for {len(rows)} games -> {ODDS_PATH.name}")


if __name__ == "__main__":
    main()
