"""
nba_picks.py
-------------
Market table, pricing (no-vig, edge, EV, quarter Kelly), automatic selections and grading.
Shared by refresh.py (selections + prediction log), nba_live.py (halftime) and the app.

Correlated exposure: selections in the same game are sized TOGETHER. Their outcomes are simulated
from the model's joint score distribution and the stakes maximise expected log bankroll as a set
(joint Kelly), then scaled by KELLY_FRACTION and capped at MAX_GAME_EXPOSURE per game. An Over plus
a team-total Over in one game share most of their risk, so together they get roughly one bet's
stake; picks that barely move together keep close to their single-bet size.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DEFAULT_PRICE = -110          # assumed when a spread/total line has no price
# A selection needs model win probability (pushes excluded) to beat the no-vig book probability by this much
MIN_EDGE = {"Total": 0.04, "Spread": 0.04, "Moneyline": 0.04, "Team Total": 0.05, "1H Total": 0.05, "2H Total": 0.05,
            "Live 2H Total": 0.05, "Live 2H Spread": 0.05}
STAR_EDGES = (0.04, 0.07, 0.10)   # edge needed for 1, 2, 3 stars
MAX_EDGE = 0.25                   # bigger "edges" are almost always stale lines or injury news
KELLY_FRACTION = 0.25
MAX_GAME_EXPOSURE = 0.05          # most of the bankroll staked on one game, all its selections combined
CORR_SHOW = 0.30                  # outcome correlation before a card names the pick it moves with
N_SIMS = 20000


def to_decimal(us):
    if us is None or pd.isna(us) or abs(us) < 100:
        return np.nan
    return 1 + (us / 100 if us > 0 else 100 / -us)


def _has(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def market_rows(r, lines):
    """Rows of (group, type, side, team, label, line, p_win, p_push, price) for every market with a line.
    `r` is Predictor.predict_game(...) evaluated AT these lines. Rows in one group are the sides of one market."""
    home, away, L = r["home"], r["away"], lines
    rows = []

    def add(group, mtype, side, team, label, line, p, push, price):
        rows.append({"group": group, "type": mtype, "side": side, "team": team, "label": label, "line": line,
                     "p": float(p), "push": float(push), "price": price})

    if _has(L.get("spread_home")):
        s, sh = r["spread"], float(L["spread_home"])
        add("spread", "Spread", "home", home, f"{home} {sh:+g}", sh, s["cover"], s["push"],
            L.get("spread_home_price") or DEFAULT_PRICE)
        add("spread", "Spread", "away", away, f"{away} {-sh:+g}", -sh, s["fail"], s["push"],
            L.get("spread_away_price") or DEFAULT_PRICE)
    if _has(L.get("ml_home")) and _has(L.get("ml_away")):
        add("ml", "Moneyline", "home", home, f"{home} ML", None, r["p_home_win"], 0.0, L["ml_home"])
        add("ml", "Moneyline", "away", away, f"{away} ML", None, 1 - r["p_home_win"], 0.0, L["ml_away"])
    for group, mtype, key, line_key, price_key, prefix, team in (
            ("tot", "Total", "total", "total_line", "total", "", None),
            ("htt", "Team Total", "home_total", "home_tt_line", "home_tt", f"{home} TT ", home),
            ("att", "Team Total", "away_total", "away_tt_line", "away_tt", f"{away} TT ", away),
            ("h1", "1H Total", "h1_total", "h1_total_line", "h1_total", "1H ", None),
            ("h2", "2H Total", "h2_total", "h2_total_line", "h2_total", "2H ", None)):
        if not _has(L.get(line_key)):
            continue
        t, ln = r[key], float(L[line_key])
        add(group, mtype, "over", team, f"{prefix}Over {ln:g}", ln, t["over"], t["push"],
            L.get(f"{price_key}_over_price") or DEFAULT_PRICE)
        add(group, mtype, "under", team, f"{prefix}Under {ln:g}", ln, t["under"], t["push"],
            L.get(f"{price_key}_under_price") or DEFAULT_PRICE)
    return pd.DataFrame(rows)


def price(mk, odds=None):
    """No-vig book probability, edge, EV and quarter Kelly (same math as the NHL app).
    odds: optional per-row American prices overriding mk['price']."""
    mk = mk.copy()
    if odds is not None:
        mk["price"] = pd.to_numeric(pd.Series(odds, index=mk.index), errors="coerce")
    mk["dec"] = mk["price"].map(to_decimal).astype(float)
    mk["implied"] = 1 / mk["dec"]
    full = mk.groupby("group")["dec"].transform(lambda s: s.notna().all())
    over = mk.groupby("group")["implied"].transform("sum").replace(0, np.nan)
    mk["book"] = np.where(full, mk["implied"] / over, mk["implied"])
    p_res = 1 - mk["push"]
    mk["p_cond"] = mk["p"] / p_res.where(p_res > 0, 1)
    mk["edge"] = mk["p_cond"] - mk["book"]
    lose = (1 - mk["p"] - mk["push"]).clip(lower=0)
    mk["ev"] = mk["p"] * (mk["dec"] - 1) - lose
    b = mk["dec"] - 1
    mk["kelly"] = ((b * mk["p_cond"] - (1 - mk["p_cond"])) / b).clip(lower=0) * KELLY_FRACTION
    return mk


def stars(edge):
    return int(sum(edge >= e for e in STAR_EDGES))


# ------------------------------------------------------------------ correlated exposure
def outcomes(rows, sims, home):
    """+1 win / 0 push / -1 loss for each market row (columns) in each simulated game (rows).
    rows: market_rows-style records (type, side, team, line); sims: Predictor.sample_scores output."""
    h, a, h1 = sims["home"], sims["away"], sims["h1"]
    cols = []
    for _, m in pd.DataFrame(rows).iterrows():
        t = m["type"]
        if t in ("Spread", "Moneyline"):
            val = (h - a) if m["side"] == "home" else (a - h)
            diff = val + (float(m["line"]) if t == "Spread" else 0.0)
        else:
            val = {"Total": h + a, "1H Total": h1, "2H Total": h + a - h1,
                   "Team Total": h if m["team"] == home else a}[t]
            diff = (val - float(m["line"])) * (1 if m["side"] == "over" else -1)
        cols.append(np.sign(diff))
    return np.column_stack(cols) if cols else np.zeros((len(h), 0))


def joint_kelly(R, start=None, upper=None):
    """Full-Kelly stakes for bets settled together. R[s, k] = profit per unit staked on bet k in
    scenario s (dec-1 win, 0 push, -1 loss). Maximises the mean log of 1 + R @ f with 0 <= f <= upper.
    upper (e.g. each bet's single-bet Kelly) stops a bet being sized UP because it hedges another:
    that benefit rests entirely on the model's correlations, so we only ever take risk off."""
    k = R.shape[1]
    if k == 0:
        return np.zeros(0)
    cap = 0.95
    ub = np.full(k, cap) if upper is None else np.clip(np.asarray(upper, float), 0, cap)

    def obj(f):
        w = 1 + R @ f
        return -np.mean(np.log(np.maximum(w, 1e-9)))

    def jac(f):
        w = np.maximum(1 + R @ f, 1e-9)
        return -(R / w[:, None]).mean(axis=0)

    x0 = np.clip(np.asarray(start if start is not None else np.full(k, 0.01), float), 0, ub)
    if x0.sum() > 0.5:
        x0 *= 0.5 / x0.sum()
    res = minimize(obj, x0, jac=jac, method="SLSQP", bounds=list(zip(np.zeros(k), ub)),
                   constraints=[{"type": "ineq", "fun": lambda f: cap - f.sum(), "jac": lambda f: -np.ones(k)}],
                   options={"ftol": 1e-10, "maxiter": 200})
    f = np.clip(res.x, 0, ub)
    f[f < 1e-5] = 0.0
    return f


def size_together(rows, sims, home, fraction=KELLY_FRACTION, cap=MAX_GAME_EXPOSURE):
    """Joint fractional Kelly for a game's selections + who each one moves with.
    rows need type/side/team/line/label/dec/kelly(solo, already fractional). Returns (stakes, notes)."""
    rows = pd.DataFrame(rows).reset_index(drop=True)
    if rows.empty:
        return np.zeros(0), []
    O = outcomes(rows, sims, home)
    R = np.where(O > 0, rows["dec"].values[None, :] - 1, np.where(O < 0, -1.0, 0.0))
    solo_full = rows["kelly"].values.astype(float) / fraction
    if len(rows) == 1:
        stakes = rows["kelly"].values.astype(float)
    else:
        stakes = joint_kelly(R, solo_full, upper=solo_full) * fraction
    if stakes.sum() > cap:
        stakes *= cap / stakes.sum()
    notes = [None] * len(rows)
    if len(rows) > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(O.T)
        for i in range(len(rows)):
            c = np.where(np.arange(len(rows)) == i, np.nan, np.nan_to_num(C[i], nan=0.0))
            j = int(np.nanargmax(np.abs(c)))
            if abs(c[j]) >= CORR_SHOW:
                notes[i] = f"{'moves with' if c[j] > 0 else 'works against'} {rows.loc[j, 'label']} ({c[j]:+.2f})"
    return stakes, notes


def game_lines_args(lines):
    """Predictor.predict_game keyword arguments for the posted lines."""
    f = lambda k: float(lines[k]) if _has(lines.get(k)) else None  # noqa: E731
    kw = {"home_team_line": f("home_tt_line"), "away_team_line": f("away_tt_line"),
          "h1_line": f("h1_total_line"), "h2_line": f("h2_total_line")}
    if f("total_line") is not None:
        kw["total_line"] = f("total_line")
    if f("spread_home") is not None:
        kw["home_spread"] = f("spread_home")
    return kw


def _passes(m):
    return MIN_EDGE[m["type"]] <= m["edge"] <= MAX_EDGE and m["ev"] > 0


def uncertain_players(r, lo=0.05, hi=0.95):
    """Named players (impact worth showing) whose status leaves real doubt."""
    out = []
    for side in ("home", "away"):
        av = (r.get("avail") or {}).get(side) or {}
        out += [p for p in av.get("players", []) if lo < p["p_play"] < hi]
    return out


def select(pred, slate_game):
    """Model selections for one slate game: best side per market that clears the thresholds.
    A pick that only clears them in one outcome of a coin-flip player's status says so ('depends_on')."""
    g, lines = slate_game, slate_game.get("lines") or {}
    args = (g["home"], g["away"], g.get("home_b2b", False), g.get("away_b2b", False), g.get("neutral", False))
    travel = g.get("travel")
    r = pred.predict_game(*args, **game_lines_args(lines), travel=travel)
    mk = market_rows(r, lines)
    if mk.empty:
        return []
    mk = price(mk).dropna(subset=["dec"])
    best = mk.sort_values("edge", ascending=False).groupby("group").head(1)
    best = best[best.apply(_passes, axis=1)] if not best.empty else best
    depends = {i: [] for i in best.index}
    if not best.empty:
        for pl in uncertain_players(r):
            for mode, needs in (("plays", "sitting"), ("sits", "playing")):
                alt = price(market_rows(pred.predict_game(*args, **game_lines_args(lines), travel=travel,
                                                          overrides={pl["player_id"]: mode}), lines))
                alt = alt.set_index(["group", "side"])
                for i, m in best.iterrows():
                    a = alt.loc[(m["group"], m["side"])]
                    if not _passes(a):
                        depends[i].append(f"{pl['name']} {needs}")
    # same-game selections are sized together (see module docstring)
    joint, corr = {i: min(float(m["kelly"]), MAX_GAME_EXPOSURE) for i, m in best.iterrows()}, {}
    if len(best) > 1:
        sims = pred.sample_scores(*args, n=N_SIMS, seed=0, travel=travel)
        stakes, notes = size_together(best, sims, g["home"])
        joint = dict(zip(best.index, stakes))
        corr = dict(zip(best.index, notes))
    now = datetime.now().isoformat(timespec="seconds")
    out = []
    for idx, m in best.iterrows():
        proj = {"Spread": r["exp_margin"] if m["side"] == "home" else -r["exp_margin"],
                "Moneyline": r["p_home_win"] if m["side"] == "home" else 1 - r["p_home_win"],
                "Total": r["exp_total"], "1H Total": r["exp_h1"], "2H Total": r["exp_h2"],
                "Team Total": r["exp_home"] if m["team"] == g["home"] else r["exp_away"]}[m["type"]]
        out.append({"pick_id": f"{g['game_id']}|{m['group']}", "game_id": str(g["game_id"]), "date": g["date"],
                    "start_utc": g.get("start_utc"), "home": g["home"], "away": g["away"],
                    "matchup": f"{g['away']} @ {g['home']}", "market": m["type"], "selection": m["label"],
                    "side": m["side"], "team": m["team"], "line": m["line"], "price": float(m["price"]),
                    "model_proj": float(proj), "model_prob": float(m["p_cond"]), "push_prob": float(m["push"]),
                    "book_prob": float(m["book"]), "edge": float(m["edge"]), "ev": float(m["ev"]),
                    "kelly": float(joint.get(idx, m["kelly"])), "kelly_solo": float(m["kelly"]),
                    "n_same_game": int(len(best)), "corr_with": corr.get(idx),
                    "stars": stars(m["edge"]), "book": lines.get("book"),
                    "depends_on": "; ".join(depends.get(idx, [])) or None,
                    "status": "pending", "actual": np.nan, "profit": np.nan, "logged_at": now})
    return out


def halftime_rows(ht, home, away, lines):
    """Market rows for the second half at halftime. ht: Predictor.halftime(...) evaluated at these lines.
    lines: h2_total_line (+ _over_price/_under_price) and h2_spread_home (+ h2_spread_home_price/_away_price)."""
    rows = []
    L = lines or {}
    if _has(L.get("h2_total_line")):
        ln, t = float(L["h2_total_line"]), ht["total"]
        for side, p in (("over", t["over"]), ("under", t["under"])):
            rows.append({"group": "lh2t", "type": "Live 2H Total", "side": side, "team": None,
                         "label": f"2H {side.title()} {ln:g}", "line": ln, "p": p, "push": t["push"],
                         "price": L.get(f"h2_total_{side}_price") or DEFAULT_PRICE})
    if _has(L.get("h2_spread_home")):
        sh, s = float(L["h2_spread_home"]), ht["spread"]
        rows += [{"group": "lh2s", "type": "Live 2H Spread", "side": "home", "team": home, "label": f"{home} 2H {sh:+g}",
                  "line": sh, "p": s["cover"], "push": s["push"], "price": L.get("h2_spread_home_price") or DEFAULT_PRICE},
                 {"group": "lh2s", "type": "Live 2H Spread", "side": "away", "team": away, "label": f"{away} 2H {-sh:+g}",
                  "line": -sh, "p": s["fail"], "push": s["push"], "price": L.get("h2_spread_away_price") or DEFAULT_PRICE}]
    return pd.DataFrame(rows)


def grade_one(p, g):
    """p: pick row; g: finished game row (scores + quarters). Returns (status, actual, units)."""
    hs, as_ = float(g["home_score"]), float(g["away_score"])
    h1 = float(g["home_q1"] + g["home_q2"] + g["away_q1"] + g["away_q2"])
    h1m = float(g["home_q1"] + g["home_q2"] - g["away_q1"] - g["away_q2"])
    t = p["market"]
    if t == "Total":
        actual = hs + as_
    elif t == "1H Total":
        actual = h1
    elif t in ("2H Total", "Live 2H Total"):
        actual = hs + as_ - h1
    elif t == "Team Total":
        actual = hs if p["team"] == g["home_team"] else as_
    elif t == "Live 2H Spread":
        actual = (hs - as_ - h1m) if p["side"] == "home" else -(hs - as_ - h1m)
    else:
        actual = (hs - as_) if p["side"] == "home" else (as_ - hs)
    if t in ("Spread", "Live 2H Spread"):
        diff = actual + float(p["line"])
    elif t == "Moneyline":
        diff = actual
    else:
        diff = (actual - float(p["line"])) * (1 if p["side"] == "over" else -1)
    status = "W" if diff > 0 else ("L" if diff < 0 else "P")
    units = (to_decimal(p["price"]) - 1) if status == "W" else (-1.0 if status == "L" else 0.0)
    return status, actual, round(units, 3)
