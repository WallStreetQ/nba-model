"""
refresh.py
-----------
One command to update everything as the season progresses, and (optionally) publish to GitHub so
the Streamlit app redeploys with the new model.

  python refresh.py                  # fetch new games, box scores, schedule, lines -> retrain ->
                                     # log pre-game predictions + selections, grade finished ones
  python refresh.py --push           # ...then git commit + push  (Streamlit Cloud redeploys automatically)
  python refresh.py --skip-fetch     # just retrain from the data you already have
  python refresh.py --backtest       # also run the walk-forward backtest (~1 min) and store its scores
  python refresh.py --period-odds    # also buy team totals + 1H/2H lines from The Odds API (~3 credits/game)
  python refresh.py --days-back 3    # sweep more days for late/postponed games (default 2)

What the app reads: model/model_artifact.json, model/prediction_log.csv, model/picks.csv.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "model" / "model_artifact.json"
PRED_LOG = ROOT / "model" / "prediction_log.csv"
PICKS_LOG = ROOT / "model" / "picks.csv"
VOLATILE_META = {"fitted_at", "as_of", "odds_fetched_at"}
PUBLISH = ["model", "data/games.parquet", "data/game_team_box.parquet", "data/game_players.parquet",
           "data/upcoming.json", "data/odds.json", "data/injuries.json", "data/injury_snapshots.parquet"]


def _same_model(old, new):
    """True if two artifacts are identical apart from timestamps."""
    strip = lambda a: {**a, "meta": {k: v for k, v in a["meta"].items() if k not in VOLATILE_META}}  # noqa: E731
    return strip(old) == strip(new)


def run(cmd, label):
    print(f"\n=== {label} ===")
    print("$", " ".join(cmd))
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit(f"\n[stopped] '{label}' failed. Nothing was published. Fix the error above "
                 f"(or re-run with --skip-fetch to retrain on existing data).")


def git(*args, check=True, capture=False):
    return subprocess.run(["git", *args], cwd=ROOT, check=check, text=True, capture_output=capture)


def publish(message):
    print("\n=== Publish to GitHub ===")
    if git("rev-parse", "--is-inside-work-tree", check=False, capture=True).returncode != 0:
        sys.exit("[stopped] This folder isn't a git repository yet. See README.md, 'Deploy'.")
    for p in PUBLISH:
        if (ROOT / p).exists():
            git("add", p)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("Nothing changed - nothing to commit.")
        return
    git("commit", "-m", message)
    if git("push", check=False).returncode != 0:
        print("\n[push failed] The commit is saved locally. If the remote has newer commits (e.g. from the\n"
              "GitHub Action), run\n  git pull --rebase\n  git push\nthen Streamlit will pick it up.")
        sys.exit(1)
    print("Pushed. Streamlit Cloud will redeploy in about a minute.")


def _read(path):
    return pd.read_csv(path, dtype={"game_id": str, "pick_id": str}) if path.exists() else pd.DataFrame()


# ------------------------------------------------------------------------------------------
# Prediction log (every game) and selections (games with an edge)
# ------------------------------------------------------------------------------------------

def finished_games():
    from nba_common import GAMES_PATH, read_table
    g = read_table(GAMES_PATH)
    g = g[g["completed"].astype(str).str.lower() == "true"].dropna(subset=["home_q2", "away_q2"])
    return g.set_index("game_id")


def update_logs(art):
    from nba_model import Predictor
    from nba_picks import grade_one, select

    pred = Predictor(art)
    fin = finished_games()
    now = datetime.now().isoformat(timespec="seconds")

    # ---- prediction log: settle, then (re)log games that haven't started ----
    log = _read(PRED_LOG)
    if not log.empty:
        for col, src in (("home_score", "home_score"), ("away_score", "away_score"), ("n_periods", "n_periods")):
            if col not in log:
                log[col] = np.nan
            log[col] = log[col].fillna(log["game_id"].map(fin[src]))
        if "h1_total_act" not in log:
            log["h1_total_act"] = np.nan
        h1 = fin["home_q1"] + fin["home_q2"] + fin["away_q1"] + fin["away_q2"]
        log["h1_total_act"] = log["h1_total_act"].fillna(log["game_id"].map(h1))
    new = []
    for g in pred.slate:
        L = g.get("lines") or {}
        from nba_picks import game_lines_args
        try:
            r = pred.predict_game(g["home"], g["away"], g["home_b2b"], g["away_b2b"], g.get("neutral", False),
                                  **game_lines_args(L))
        except ValueError as e:
            print(f"  [skip] {g['away']} @ {g['home']}: {e}")
            continue
        new.append({"game_id": g["game_id"], "date": g["date"], "start_utc": g.get("start_utc"),
                    "home": g["home"], "away": g["away"], "home_b2b": g["home_b2b"], "away_b2b": g["away_b2b"],
                    "exp_home": r["exp_home"], "exp_away": r["exp_away"], "exp_total": r["exp_total"],
                    "exp_margin": r["exp_margin"], "p_home_win": r["p_home_win"], "p_ot": r["p_ot"],
                    "exp_h1": r["exp_h1"], "exp_h2": r["exp_h2"], "pace": r["pace"],
                    "total_line": L.get("total_line"), "spread_home": L.get("spread_home"),
                    "ml_home": L.get("ml_home"), "ml_away": L.get("ml_away"),
                    "home_tt_line": L.get("home_tt_line"), "away_tt_line": L.get("away_tt_line"),
                    "h1_total_line": L.get("h1_total_line"), "h2_total_line": L.get("h2_total_line"),
                    "p_over": r["total"]["over"] if L.get("total_line") is not None else np.nan,
                    "p_home_cover": r["spread"]["cover"] if L.get("spread_home") is not None else np.nan,
                    "home_av_adj": (r["avail"]["home"] or {}).get("adj"), "away_av_adj": (r["avail"]["away"] or {}).get("adj"),
                    "absences": "; ".join(f"{p['name']} ({p['status']})" for side in ("away", "home")
                                          for p in ((r["avail"][side] or {}).get("players") or [])) or None,
                    "book": L.get("book"), "logged_at": now, "model_through": art["meta"]["trained_through"]})
    if new:
        new = pd.DataFrame(new)
        if not log.empty:
            settled = log["home_score"].notna()
            keep = log[settled | ~log["game_id"].isin(new["game_id"])]
            new = new[~new["game_id"].isin(log.loc[settled, "game_id"])]
            log = pd.concat([keep, new], ignore_index=True)
        else:
            log = new
    if not log.empty:
        log = log.sort_values(["date", "game_id"]).reset_index(drop=True)
        PRED_LOG.parent.mkdir(exist_ok=True)
        log.to_csv(PRED_LOG, index=False)
        print(f"  prediction log: {len(log)} games ({int(log['home_score'].notna().sum()) if 'home_score' in log else 0} settled)")

    # ---- selections: replace pending picks for unstarted games, grade finished ones ----
    picks = _read(PICKS_LOG)
    slate_ids = {g["game_id"] for g in pred.slate}
    if not picks.empty:
        picks = picks[~(picks["game_id"].isin(slate_ids) & (picks["status"] == "pending"))]
    fresh = [p for g in pred.slate for p in select(pred, g)]
    picks = pd.concat([picks, pd.DataFrame(fresh)], ignore_index=True) if fresh else picks
    if not picks.empty:
        for i, p in picks[picks["status"] == "pending"].iterrows():
            if p["game_id"] in fin.index:
                gr = fin.loc[p["game_id"]].to_dict()
                gr["home_team"], gr["away_team"] = fin.loc[p["game_id"], "home_team"], fin.loc[p["game_id"], "away_team"]
                picks.loc[i, ["status", "actual", "profit"]] = grade_one(p.to_dict(), gr)
        picks = picks.sort_values(["date", "game_id", "market"]).reset_index(drop=True)
        picks.to_csv(PICKS_LOG, index=False)
        done = picks[picks["status"].isin(["W", "L", "P"])]
        print(f"  selections: {len(fresh)} open today | record {(done['status'] == 'W').sum()}-"
              f"{(done['status'] == 'L').sum()}-{(done['status'] == 'P').sum()}, {done['profit'].sum():+.2f} units")


def backtest_summary(**kw):
    from nba_model import walk_forward
    r = walk_forward(ROOT / "data", log=lambda *a: None, **kw)
    r.pop("raw")
    return r


def main():
    from nba_model import DEFAULT_CARRYOVER, DEFAULT_HALF_LIFE, DEFAULT_PRIOR_GAMES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-fetch", action="store_true", help="don't call any API; retrain from existing data/")
    ap.add_argument("--days-back", type=int, default=2, help="days to sweep for newly finished games")
    ap.add_argument("--push", action="store_true", help="git commit + push model and data when done")
    ap.add_argument("--backtest", action="store_true", help="run the walk-forward backtest and store its scores")
    ap.add_argument("--period-odds", action="store_true", help="team totals + 1H/2H lines from The Odds API")
    ap.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE, help="recency half-life (days)")
    ap.add_argument("--prior-games", type=float, default=DEFAULT_PRIOR_GAMES, help="ridge prior strength (games)")
    ap.add_argument("--carryover", type=float, default=DEFAULT_CARRYOVER, help="weight per past season (1 = off)")
    args = ap.parse_args()

    py = sys.executable
    if not args.skip_fetch:
        run([py, "nba_update.py", "--days-back", str(args.days_back)], "1/5 New games")
        run([py, "nba_boxscores.py"], "2/5 Box scores for new games")
        run([py, "nba_schedule.py"], "3/5 Upcoming schedule + ESPN lines")
        run([py, "nba_injuries.py"], "4/5 Injuries + rosters")
        run([py, "nba_odds.py"] + (["--period"] if args.period_odds else []), "5/5 Sportsbook lines (optional)")

    sys.path.insert(0, str(ROOT))
    from nba_common import INJURIES_PATH, ODDS_PATH, UPCOMING_PATH
    from nba_model import load_artifact, save_artifact, train_artifact
    upcoming = json.loads(UPCOMING_PATH.read_text()) if UPCOMING_PATH.exists() else []
    odds = json.loads(ODDS_PATH.read_text()) if ODDS_PATH.exists() else None
    injuries = json.loads(INJURIES_PATH.read_text()) if INJURIES_PATH.exists() else None

    print("\n=== Retrain model ===")
    settings = dict(half_life_days=args.half_life, prior_games=args.prior_games, carryover=args.carryover)
    art = train_artifact(ROOT / "data", upcoming=upcoming, odds=odds, injuries=injuries, log=lambda *a: print(" ", *a), **settings)
    old = load_artifact(ARTIFACT) if ARTIFACT.exists() else None
    if old and old.get("version", 0) < art["version"]:
        old = None

    if args.backtest:
        print("\n=== Walk-forward backtest ===")
        bt = backtest_summary(**settings)
        print(f"  {bt['n_games']:,} games ({', '.join(bt['seasons'])}) | win Brier {bt['win_brier']:.4f} "
              f"(constant {bt['const_brier']:.4f}) | total MAE {bt['total_mae']:.2f} | margin MAE {bt['margin_mae']:.2f}")
        art["meta"]["backtest"] = json.loads(json.dumps(bt, default=float))
    elif old and old["meta"].get("backtest", {}).get("settings") == settings:
        art["meta"]["backtest"] = old["meta"]["backtest"]

    m = art["meta"]
    if old and _same_model(old, art):
        print(f"\nModel unchanged - keeping existing {ARTIFACT.relative_to(ROOT)}")
    else:
        save_artifact(art, ARTIFACT)
        print(f"\nSaved {ARTIFACT.relative_to(ROOT)} | trained through {m['trained_through']} | "
              f"box scores {'ON' if m['box_on'] else 'OFF'} | {len(art['slate'])} upcoming games")

    print("\n=== Prediction log + selections ===")
    update_logs(art)

    if args.push:
        publish(f"Update model through {m['trained_through']}")
    else:
        print("\nTo publish: python refresh.py --skip-fetch --push   (or git add/commit/push model/ and data/)")


if __name__ == "__main__":
    main()
