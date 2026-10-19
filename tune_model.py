"""
tune_model.py
--------------
Walk-forward tuning of the model settings in nba_model.py:
    DEFAULT_HALF_LIFE, DEFAULT_PRIOR_GAMES, DEFAULT_CARRYOVER, DEFAULT_TRAVEL

How it works (same method as the NHL tuner)
  1. Backtests the CURRENT defaults (the baseline).
  2. Sweeps each setting on its own, holding the others at their current values.
  3. Compares every run to the baseline on the SAME games, with a bootstrap standard error,
     so you can tell a real improvement from noise.
  4. Combines the per-setting winners and checks the combo still beats the baseline.
  5. Prints exactly which DEFAULT_ lines to change, or tells you to keep what you have.

  python tune_model.py                          # judge on moneyline Brier
  python tune_model.py --metric total_abs       # judge on full-game total error
  python tune_model.py --only half_life         # sweep just some settings
  python tune_model.py --only travel --metric margin_abs   # do the travel terms earn their place?
  python tune_model.py --seasons 3              # test on the last 3 complete seasons (default 2)
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

import nba_model as nm
from nba_model import load_data, walk_forward

warnings.filterwarnings("ignore")

# setting name -> (walk_forward keyword, DEFAULT_ constant, refresh.py flag, values to try)
SETTINGS = {
    "half_life": ("half_life_days", "DEFAULT_HALF_LIFE", "--half-life", [45, 60, 90, 120, 180, 270]),
    "prior_games": ("prior_games", "DEFAULT_PRIOR_GAMES", "--prior-games", [2, 4, 8, 12, 20, 30]),
    "carryover": ("carryover", "DEFAULT_CARRYOVER", "--carryover", [0.3, 0.45, 0.6, 0.8, 1.0]),
    # 1 = rest/travel terms on, 0 = back-to-backs only. Keep travel only if it beats "off" by more than noise.
    "travel": ("travel", "DEFAULT_TRAVEL", "--travel", [0, 1]),
}

# metric name -> per-game loss (lower is better) from walk_forward's raw rows
METRICS = {
    "win_brier": lambda r: (r["p_win"] - r["y_win"]) ** 2,
    "margin_abs": lambda r: (r["act_m_reg"] - r["mu_m"]).abs(),
    "total_abs": lambda r: (r["act_tot"] - r["exp_final"]).abs(),
    "h1_abs": lambda r: (r["act_h1"] - r["mu_h1"]).abs(),
}


def current_defaults():
    return {kw: getattr(nm, const) for kw, const, _, _ in SETTINGS.values()}


def run(D, n_seasons, **kw):
    return walk_forward(D=D, n_seasons=n_seasons, log=lambda *a: None, **kw)["raw"].reset_index(drop=True)


def compare(raw, base_raw, metric, rng, reps=1000):
    if len(raw) != len(base_raw):
        raise RuntimeError("runs scored different games - can't compare them pairwise")
    d = (METRICS[metric](raw) - METRICS[metric](base_raw)).values
    idx = rng.integers(0, len(d), (reps, len(d)))
    return float(d.mean()), float(d[idx].mean(axis=1).std())


def scores(raw):
    return {m: float(f(raw).mean()) for m, f in METRICS.items()}


def fmt(x):
    return f"{x:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="win_brier", choices=list(METRICS))
    ap.add_argument("--only", nargs="+", choices=list(SETTINGS))
    ap.add_argument("--seasons", type=int, default=2, help="how many recent complete seasons to test on")
    ap.add_argument("--z", type=float, default=2.0, help="improvement must exceed z x SE to count")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    D = load_data(log=lambda *a: None)
    base = current_defaults()
    names = args.only or list(SETTINGS)

    t0 = time.time()
    print(f"Baseline (current defaults): {', '.join(f'{k}={fmt(v)}' for k, v in base.items())}")
    base_raw = run(D, args.seasons, **base)
    print(f"  {len(base_raw):,} test games in seasons {', '.join(sorted(base_raw['season'].astype(str).unique()))}"
          f"  ({time.time() - t0:.0f} s per run)\n")

    rows = [{"setting": "baseline", "value": "", **scores(base_raw), "diff": 0.0, "se": 0.0}]
    winners = {}
    for name in names:
        kw, const, flag, values = SETTINGS[name]
        best = None
        for v in values:
            if np.isclose(v, base[kw]):
                continue
            raw = run(D, args.seasons, **{**base, kw: v})
            diff, se = compare(raw, base_raw, args.metric, rng)
            rows.append({"setting": name, "value": fmt(v), **scores(raw), "diff": diff, "se": se})
            print(f"  {name} = {fmt(v):<6} {args.metric} diff {diff:+.5f} (SE {se:.5f})", flush=True)
            if diff < -args.z * se and (best is None or diff < best[1]):
                best = (v, diff, se)
        if best:
            winners[name] = best[0]

    pd.set_option("display.width", 220)
    print(f"\nAll runs (lower is better; diff/se are {args.metric} vs baseline on the same games):")
    print(pd.DataFrame(rows).set_index(["setting", "value"]).round(5).to_string())

    print("\n" + "=" * 70)
    if not winners:
        print(f"RECOMMENDATION: keep the current defaults. Nothing beat them on {args.metric} by more "
              f"than {args.z:g} standard errors.")
        return
    combo = {**base, **{SETTINGS[n][0]: v for n, v in winners.items()}}
    if len(winners) > 1:
        print("Checking the per-setting winners together...")
        diff, se = compare(run(D, args.seasons, **combo), base_raw, args.metric, rng)
        print(f"  combined: {args.metric} diff {diff:+.5f} (SE {se:.5f})")
        if diff >= -args.z * se:
            print("\nRECOMMENDATION: keep the current defaults. Individually some settings helped, but together "
                  "they don't clearly beat the baseline. Try --only with a single setting.")
            return
    print("RECOMMENDATION: in nba_model.py change")
    for n, v in winners.items():
        print(f"    {SETTINGS[n][1]} = {fmt(v)}      (was {fmt(base[SETTINGS[n][0]])})")
    print("then run:   python refresh.py --skip-fetch --backtest --push")
    flags = " ".join(f"{SETTINGS[n][2]} {fmt(v)}" for n, v in winners.items())
    print(f"(or test first without editing:   python refresh.py --skip-fetch --backtest {flags})")
    edge = [n for n, v in winners.items() if n != "travel" and v in (SETTINGS[n][3][0], SETTINGS[n][3][-1])]
    if edge:
        print(f"Note: {', '.join(edge)} won at the edge of its range - extend the values in SETTINGS and re-run.")


if __name__ == "__main__":
    main()
