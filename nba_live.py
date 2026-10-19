"""
nba_live.py
------------
Halftime second-half pricing. Watches ESPN's live scoreboard; when a game reaches halftime it prices
the second half from the first-half score (Predictor.halftime) and writes:

  model/live.json          tonight's priced games: 1H score, 2H projection and fair lines, live win
                           chance, and the book's 2H line + edge when The Odds API has one
  model/halftime_log.csv   one row per game priced, graded when the game ends (track record)
  model/live_picks.csv     2H selections when a book line clears the edge threshold (needs --odds)

Only this script writes those three files and refresh.py never touches them, so the evening
GitHub Action and refresh.py can both push without stepping on each other.

A game first seen after the 3rd quarter has started is still logged (its projection only uses the
1H score, so it counts for accuracy) but marked `late`: no book line, no selection.

Usage:
  python nba_live.py                  # one pass: price games at halftime, grade finished ones
  python nba_live.py --watch 345      # poll every 90 s for up to 345 minutes (what the Action runs)
  python nba_live.py --odds           # also buy live 2H lines (The Odds API, ~2 credits per game at half)
  python nba_live.py --push           # commit + push whenever something new is priced or graded
"""

import argparse
import datetime as dt
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nba_common import BASE, TEAM_NAME_TO_ABBR, TZ, get, parse_scoreboard, today_et
from nba_model import Predictor, load_artifact
from nba_picks import MAX_EDGE, MAX_GAME_EXPOSURE, MIN_EDGE, grade_one, halftime_rows, price, stars

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "model" / "model_artifact.json"
PRED_LOG = ROOT / "model" / "prediction_log.csv"
LIVE_PATH = ROOT / "model" / "live.json"
HT_LOG = ROOT / "model" / "halftime_log.csv"
LIVE_PICKS = ROOT / "model" / "live_picks.csv"
ODDS_API = "https://api.the-odds-api.com/v4/sports/basketball_nba"
POLL_SECONDS = 90
KEEP_DAYS = 2            # live.json keeps games from today and yesterday


# ------------------------------------------------------------------ scoreboard
def live_games(day: dt.date) -> list[dict]:
    """parse_scoreboard rows plus live status (period, clock, ESPN status name)."""
    js = get(f"{BASE}/scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": 100})
    rows = parse_scoreboard(js, day)
    status = {}
    for ev in (js or {}).get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        s = comp.get("status") or ev.get("status") or {}
        status[str(ev["id"])] = {"status_name": (s.get("type") or {}).get("name"), "period": s.get("period") or 0,
                                 "clock": s.get("displayClock")}
    return [{**r, **status.get(r["game_id"], {})} for r in rows]


def at_half(g) -> bool:
    if g.get("status_name") == "STATUS_HALFTIME" or "half" in str(g.get("status_detail") or "").lower():
        return True
    return g.get("game_state") == "in" and g.get("period") == 2 and str(g.get("clock")) in ("0.0", "0:00")


def past_half(g) -> bool:
    return (g.get("game_state") == "in" and (g.get("period") or 0) >= 3) or bool(g.get("completed"))


def first_half(g):
    try:
        return float(g["home_q1"] + g["home_q2"]), float(g["away_q1"] + g["away_q2"])
    except (TypeError, KeyError):
        return None


# ------------------------------------------------------------------ pre-game numbers
def pregame_for(pred, g, plog):
    """Pre-tip means for this game: saved in the prediction log by the last refresh before tip-off
    (injuries as of then), else recomputed from the model with the slate's rest/travel flags."""
    if not plog.empty and "reg_home" in plog:
        row = plog[plog["game_id"] == g["game_id"]]
        if not row.empty and row[["reg_home", "reg_away", "exp_h1_home", "exp_h1_away"]].notna().all(axis=1).iloc[0]:
            r = row.iloc[-1]
            return {"mu_h": float(r["reg_home"]), "mu_a": float(r["reg_away"]), "h1_h": float(r["exp_h1_home"]),
                    "h1_a": float(r["exp_h1_away"])}, "prediction log"
    sg = next((s for s in pred.slate if s["game_id"] == g["game_id"]), None)
    if sg:
        return pred.pregame(sg["home"], sg["away"], sg["home_b2b"], sg["away_b2b"], sg.get("neutral", False),
                            travel=sg.get("travel")), "model (slate)"
    return pred.pregame(g["home_team"], g["away_team"], neutral=bool(g.get("neutral"))), "model"


# ------------------------------------------------------------------ live 2H lines (optional)
def live_lines(games, key):
    """{game_id: lines} with h2_total_line/_prices and h2_spread_home/_prices from The Odds API."""
    from nba_odds import BOOK_PRIORITY, _ou
    out = {}
    try:
        r = requests.get(f"{ODDS_API}/events", params={"apiKey": key}, timeout=20)   # free: no credits
        r.raise_for_status()
        events = r.json()
    except requests.RequestException as e:
        print(f"  [warn] Odds API events unavailable ({e})")
        return out
    by_teams = {(TEAM_NAME_TO_ABBR.get(e["home_team"]), TEAM_NAME_TO_ABBR.get(e["away_team"])): e for e in events}
    for g in games:
        ev = by_teams.get((g["home_team"], g["away_team"]))
        if not ev:
            continue
        try:
            r = requests.get(f"{ODDS_API}/events/{ev['id']}/odds", timeout=20, params={
                "apiKey": key, "regions": "us", "oddsFormat": "american", "markets": "totals_h2,spreads_h2"})
            r.raise_for_status()
            js = r.json()
            print(f"  odds api: {r.headers.get('x-requests-remaining', '?')} credits left")
        except requests.RequestException as e:
            print(f"  [warn] live 2H lines for {g['away_team']} @ {g['home_team']}: {e}")
            continue
        books = {b["key"]: b for b in js.get("bookmakers", [])}
        lines = {}
        for k in [k for k in BOOK_PRIORITY if k in books] + [k for k in books if k not in BOOK_PRIORITY]:
            mk = {m["key"]: m.get("outcomes") for m in books[k].get("markets", [])}
            if "h2_total_line" not in lines and mk.get("totals_h2"):
                ln, o, u = _ou(mk["totals_h2"])
                lines.update(h2_total_line=ln, h2_total_over_price=o, h2_total_under_price=u, book=k)
            if "h2_spread_home" not in lines and mk.get("spreads_h2"):
                d = {o["name"]: o for o in mk["spreads_h2"]}
                hf, af = ev["home_team"], ev["away_team"]
                if hf in d:
                    lines.update(h2_spread_home=d[hf].get("point"), h2_spread_home_price=d[hf].get("price"),
                                 h2_spread_away_price=(d.get(af) or {}).get("price"), book=lines.get("book") or k)
        if lines:
            out[g["game_id"]] = lines
    return out


# ------------------------------------------------------------------ pricing
def price_game(pred, g, pre, src, lines, late, now):
    h1h, h1a = first_half(g)
    L = lines or {}
    ht = pred.halftime(h1h, h1a, pre, L.get("h2_total_line"), L.get("h2_spread_home"))
    rec = {"game_id": g["game_id"], "date": g["game_date"], "start_utc": g.get("start_utc"),
           "home": g["home_team"], "away": g["away_team"], "priced_at": now, "late": bool(late), "pre_source": src,
           "h1_home": h1h, "h1_away": h1a, "pre_2h": ht["pre_2h"], "pre_2hm": ht["pre_2hm"],
           "mu_2h": ht["mu_2h"], "sd_2h": ht["sd_2h"], "mu_2hm": ht["mu_2hm"], "sd_2hm": ht["sd_2hm"],
           "p_home_win": ht["p_home_win"], "h2_total_line": L.get("h2_total_line"),
           "h2_spread_home": L.get("h2_spread_home"), "book": L.get("book")}
    markets, picks = [], []
    mk = halftime_rows(ht, g["home_team"], g["away_team"], L)
    if not mk.empty:
        mk = price(mk).dropna(subset=["dec"])
        markets = [{"label": m["label"], "type": m["type"], "price": float(m["price"]), "model": float(m["p_cond"]),
                    "book": float(m["book"]), "edge": float(m["edge"]), "ev": float(m["ev"]),
                    "kelly": float(m["kelly"])} for _, m in mk.iterrows()]
        best = mk.sort_values("edge", ascending=False).groupby("group").head(1)
        best = best[(best["edge"] >= best["type"].map(MIN_EDGE)) & (best["edge"] <= MAX_EDGE) & (best["ev"] > 0)]
        scale = min(1.0, MAX_GAME_EXPOSURE / best["kelly"].sum()) if best["kelly"].sum() > 0 else 1.0
        for _, m in best.iterrows():
            proj = ht["mu_2h"] if m["type"] == "Live 2H Total" else (ht["mu_2hm"] if m["side"] == "home" else -ht["mu_2hm"])
            picks.append({"pick_id": f"{g['game_id']}|{m['group']}", "game_id": g["game_id"], "date": g["game_date"],
                          "start_utc": g.get("start_utc"), "home": g["home_team"], "away": g["away_team"],
                          "matchup": f"{g['away_team']} @ {g['home_team']}", "market": m["type"],
                          "selection": m["label"], "side": m["side"], "team": m["team"], "line": m["line"],
                          "price": float(m["price"]), "model_proj": float(proj), "model_prob": float(m["p_cond"]),
                          "push_prob": float(m["push"]), "book_prob": float(m["book"]), "edge": float(m["edge"]),
                          "ev": float(m["ev"]), "kelly": float(m["kelly"]) * scale, "kelly_solo": float(m["kelly"]),
                          "n_same_game": int(len(best)), "corr_with": None, "stars": stars(m["edge"]),
                          "book": L.get("book"), "depends_on": None, "status": "pending", "actual": np.nan,
                          "profit": np.nan, "logged_at": now})
    view = {**rec, "h1_line": f"{g['away_team']} {h1a:.0f} – {h1h:.0f} {g['home_team']}",
            "markets": markets, "picks": [p["selection"] for p in picks]}
    return rec, view, picks


# ------------------------------------------------------------------ files
def _read(path):
    return pd.read_csv(path, dtype={"game_id": str, "pick_id": str}) if path.exists() else pd.DataFrame()


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_if_changed(path, text):
    old = path.read_text() if path.exists() else None
    if old == text:
        return False
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    return True


def csv_text(df):
    """CSV text normalised through one read/write round-trip, so an unchanged table compares equal to
    the file it was read from (otherwise every poll would look like a change and commit)."""
    first = df.to_csv(index=False)
    return pd.read_csv(io.StringIO(first), dtype={"game_id": str, "pick_id": str}).to_csv(index=False)


def settle(htlog, picks, finals):
    """Fill final scores into the halftime log and grade live picks. finals: {game_id: game row}."""
    if not htlog.empty:
        for c in ("home_score", "away_score", "act_2h", "act_2hm"):
            if c not in htlog:
                htlog[c] = np.nan
        for i, r in htlog[htlog["home_score"].isna()].iterrows():
            g = finals.get(r["game_id"])
            if g is None:
                continue
            hs, as_ = float(g["home_score"]), float(g["away_score"])
            htlog.loc[i, ["home_score", "away_score", "act_2h", "act_2hm"]] = [
                hs, as_, hs + as_ - r["h1_home"] - r["h1_away"], (hs - as_) - (r["h1_home"] - r["h1_away"])]
    if not picks.empty:
        for i, p in picks[picks["status"] == "pending"].iterrows():
            g = finals.get(p["game_id"])
            if g is not None:
                picks.loc[i, ["status", "actual", "profit"]] = grade_one(p.to_dict(), g)
    return htlog, picks


def one_pass(pred, args, now_et):
    tonight_days = [now_et.date()] + ([now_et.date() - dt.timedelta(days=1)] if now_et.hour < 6 else [])
    htlog, picks, plog = _read(HT_LOG), _read(LIVE_PICKS), _read(PRED_LOG)
    unsettled = set(htlog.loc[htlog["home_score"].isna(), "date"]) if "home_score" in htlog else set(htlog.get("date", []))
    week_ago = now_et.date() - dt.timedelta(days=7)
    days = tonight_days + [d for d in sorted(dt.date.fromisoformat(x) for x in unsettled)
                           if d not in tonight_days and d >= week_ago]           # games that ended after the last run
    games = {d: live_games(d) for d in days}
    tonight = [g for d in tonight_days for g in games[d]]
    done_ids = set(htlog["game_id"]) if not htlog.empty else set()
    stamp = now_et.isoformat(timespec="seconds")

    todo = [g for g in tonight if g["game_id"] not in done_ids and first_half(g) and (at_half(g) or past_half(g))]
    lines = {}
    key = os.environ.get("ODDS_API_KEY")
    if args.odds and key and any(at_half(g) for g in todo):
        lines = live_lines([g for g in todo if at_half(g)], key)
    live = json.loads(LIVE_PATH.read_text()) if LIVE_PATH.exists() else {"games": {}}
    cutoff = (now_et.date() - dt.timedelta(days=KEEP_DAYS - 1)).isoformat()
    views = {k: v for k, v in (live.get("games") or {}).items() if v.get("date", "") >= cutoff}
    new_log, new_picks = [], []
    for g in todo:
        try:
            pre, src = pregame_for(pred, g, plog)
        except ValueError as e:
            print(f"  [skip] {g['away_team']} @ {g['home_team']}: {e}")
            continue
        late = not at_half(g)
        rec, view, pk = price_game(pred, g, pre, src, None if late else lines.get(g["game_id"]), late, stamp)
        new_log.append(rec)
        new_picks += pk
        views[g["game_id"]] = view
        ln = rec["h2_total_line"]
        print(f"  {'LATE ' if late else ''}{view['h1_line']} at half | 2H total {rec['mu_2h']:.1f} (pre-game "
              f"{rec['pre_2h']:.1f}){f' vs book {ln:g}' if ln is not None else ''} | {g['home_team']} wins "
              f"{rec['p_home_win']:.0%}" + (f" | picks: {', '.join(view['picks'])}" if view["picks"] else ""))
    if new_log:
        htlog = pd.concat([htlog, pd.DataFrame(new_log)], ignore_index=True)
    if new_picks:
        picks = pd.concat([picks, pd.DataFrame(new_picks)], ignore_index=True)

    finals = {}
    for d in days:
        for g in games[d]:
            if g.get("completed") and first_half(g):
                finals[g["game_id"]] = g
    htlog, picks = settle(htlog, picks, finals)
    for gid, v in views.items():
        if gid in finals:
            v["final"] = {"home": finals[gid]["home_score"], "away": finals[gid]["away_score"]}
            if gid in set(htlog.get("game_id", [])):
                r = htlog[htlog["game_id"] == gid].iloc[-1]
                v["act_2h"] = r.get("act_2h")

    changed = False
    if not htlog.empty:
        changed |= write_if_changed(HT_LOG, csv_text(htlog.sort_values(["date", "priced_at"])))
    if not picks.empty:
        changed |= write_if_changed(LIVE_PICKS, csv_text(picks.sort_values(["date", "game_id"])))
    body = {"games": _jsonable(views), "model_through": pred.meta["trained_through"],
            "halftime_fitted": bool((pred.art.get("halftime") or {}).get("fitted"))}
    if (live.get("games"), live.get("model_through")) != (body["games"], body["model_through"]):
        changed |= write_if_changed(LIVE_PATH, json.dumps({**body, "updated_at": stamp}, indent=1))

    remaining = [g for g in tonight if not g.get("completed")]
    state = "none" if not tonight else ("done" if not remaining else "live")
    return changed, state, len(new_log)


# ------------------------------------------------------------------ git
def git(*a, check=False):
    return subprocess.run(["git", *a], cwd=ROOT, check=check, text=True, capture_output=True)


def publish(msg):
    for p in (LIVE_PATH, HT_LOG, LIVE_PICKS):
        if p.exists():
            git("add", str(p.relative_to(ROOT)))
    if git("diff", "--cached", "--quiet").returncode == 0:
        return
    git("commit", "-m", msg)
    for attempt in range(3):
        if git("push").returncode == 0:
            print("  pushed")
            return
        git("pull", "--rebase", "--autostash")
        time.sleep(3 * (attempt + 1))
    print("  [warn] push failed three times; will retry on the next change")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=float, default=0, help="keep polling for this many minutes")
    ap.add_argument("--poll", type=int, default=POLL_SECONDS, help="seconds between polls in --watch mode")
    ap.add_argument("--odds", action="store_true", help="buy live 2H lines from The Odds API (needs ODDS_API_KEY)")
    ap.add_argument("--push", action="store_true", help="git commit + push when something changes")
    args = ap.parse_args()
    if not ARTIFACT.exists():
        sys.exit("model/model_artifact.json not found - run refresh.py first")
    pred = Predictor(load_artifact(ARTIFACT))
    if args.odds and not os.environ.get("ODDS_API_KEY"):
        print("  ODDS_API_KEY not set - pricing without book lines (fair lines only)")
    deadline = time.time() + args.watch * 60
    while True:
        now = dt.datetime.now(TZ)
        changed, state, n_new = one_pass(pred, args, now)
        if changed and args.push:
            publish(f"Halftime pricing {now:%Y-%m-%d %H:%M} ET")
        if state == "none":
            print(f"{now:%H:%M} no games today ({today_et()}) - nothing to watch")
            break
        if state == "done":
            print(f"{now:%H:%M} all of tonight's games are final")
            break
        if not args.watch or time.time() + args.poll > deadline:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
