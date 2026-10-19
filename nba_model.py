"""
nba_model.py
-------------
Shared model code for the refresh pipeline AND the Streamlit app.

  train_artifact(data_dir)  -> dict    fits the model and returns a JSON-safe "artifact"
  walk_forward(data_dir)    -> dict    leak-free backtest (refresh.py --backtest, tune_model.py)
  save_artifact / load_artifact        the ONLY model file the app reads
  Predictor(artifact)                  every market for any matchup

Model (v1)
  * Team ratings by weighted ridge regression with team (offense) and opponent (defense) terms:
        efficiency  = points per 100 possessions ~ league + off[team] + def[opp] + home + rest
        pace        = possessions               ~ league + pace[team] + pace[opp] + home + rest
        regulation points = pace x efficiency / 100
    Possessions come from box scores (FGA - OREB + TOV + 0.44 FTA). Without enough box-score
    coverage the model falls back to a direct points rating. A separate rating set models
    first-half points.
  * Target is REGULATION scoring (48 minutes). A tie after regulation is settled with the
    league's real overtime scoring (empirical OT score distribution), so every full-game market
    (total, team totals, spread, moneyline) comes from one joint final-score grid.
  * Margin calibration: plain normal margins under-predict regulation ties (~3% vs ~6%) and
    close-game margins. The grid is re-weighted by |regulation margin| with factors fitted to
    out-of-sample games.
  * Recency half-life + season carryover; the ridge prior is worth `prior_games` games of
    league-average play, so ratings regress toward average when there is little current data.
  * Rest and travel: back-to-back, 3 games in 4 nights, time zones crossed since the last game,
    game number of a road trip and visiting at altitude (DEN, UTAH), each for the team and its
    opponent. Neutral-site games get no home edge. Travel terms can be switched off (DEFAULT_TRAVEL)
    and tune_model.py checks whether they help.
  * Halftime: second-half total and margin given the first-half score, fitted on out-of-sample
    games (Predictor.halftime), used by nba_live.py and the app's Halftime tab.
  * Uncertainty (score spread) is measured on out-of-sample predictions, never in-sample fits.
  * Player availability (nba_availability.py): tonight's injuries move the projection by the value of the
    minutes that change hands; see that module's docstring.
"""

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import nba_availability as nav
from nba_common import DATA_DIR, competitive_only, read_table, season_of

ARTIFACT_VERSION = 2
DEFAULT_HALF_LIFE = 120      # days
DEFAULT_PRIOR_GAMES = 2      # ridge prior strength, in games of league-average play
DEFAULT_CARRYOVER = 0.6      # weight multiplier per season boundary crossed (1 = off)
DEFAULT_TRAVEL = 0           # 1 = use the travel terms below, 0 = back-to-backs only
DEFAULT_PLAYOFF_WEIGHT = 0.7

# Per-team schedule features. Each enters the ratings twice: for the team and for its opponent.
TRAVEL_FEATS = ["three_in_four", "tz", "trip", "alt"]
REST_COLS = ["b2b", "opp_b2b"]
# Altitude has no opponent term: with only two altitude teams the data can't separate "visitors score
# less" from "the home side scores more", so one visitor-side term carries the whole effect.
TRAVEL_COLS = REST_COLS + [c for f in TRAVEL_FEATS for c in ((f,) if f == "alt" else (f, f"opp_{f}"))]
TEAM_TZ = {**{t: 0 for t in ("ATL", "BOS", "BKN", "CHA", "CLE", "DET", "IND", "MIA", "NY", "ORL", "PHI", "TOR", "WSH")},
           **{t: -1 for t in ("CHI", "DAL", "HOU", "MEM", "MIL", "MIN", "NO", "OKC", "SA")},
           **{t: -2 for t in ("DEN", "UTAH", "PHX")},
           **{t: -3 for t in ("GS", "LAC", "LAL", "POR", "SAC")}}   # hours vs Eastern (Arizona ~ Mountain in season)
ALTITUDE_TEAMS = {"DEN", "UTAH"}
TRIP_BREAK_DAYS = 5          # a gap this long ends a road trip (All-Star break, season boundary)
TRIP_CAP = 4                 # road-trip game number counted up to this (0 = first road game)
TRAVEL_SHOW = 0.5            # margin points before the app names a travel effect
TRAVEL_HALF_LIFE = 730       # days: travel effects are league-wide and slow-moving, so they're fitted on
                             # long-horizon weights, not the ratings' recency half-life (too noisy)
REG_MIN, OT_MIN = 48, 5
LO, HI = 40, 220             # regulation points grid per team
MARGIN_BUCKETS = 16          # |regulation margin| 0..14, 15+
CALIBRATION_DAYS = 365       # out-of-sample window used for score spread + margin weights
REFIT_DAYS = 14


# ======================================================================================
# Market math
# ======================================================================================

def _ou(values, probs, line):
    over = float(probs[values > line + 1e-9].sum())
    push = float(probs[np.abs(values - line) < 1e-9].sum())
    return {"line": float(line), "over": over, "under": float(max(0.0, 1 - over - push)), "push": push}


def discrete_normal_ou(mu, sd, line):
    """Over/under on an integer-valued score ~ Normal(mu, sd), with push on whole-number lines."""
    lo = np.floor(mu - 6 * sd)
    v = np.arange(lo, np.ceil(mu + 6 * sd) + 1)
    p = norm.cdf(v + 0.5, mu, sd) - norm.cdf(v - 0.5, mu, sd)
    return _ou(v, p / p.sum(), line)


class GameDist:
    """Joint distribution of one game.
    P[i, j] = regulation score (LO + i, LO + j); F = final score after settling ties with OT."""

    def __init__(self, mu_h, mu_a, sig, margin_w=None, ot_pmf=None):
        k = np.arange(LO, HI + 1)
        H, A = np.meshgrid(k, k, indexing="ij")
        P = norm.pdf(H + A, mu_h + mu_a, sig["total_reg"]) * norm.pdf(H - A, mu_h - mu_a, sig["margin_reg"])
        if margin_w is not None:
            P = P * np.asarray(margin_w)[np.minimum(np.abs(H - A), MARGIN_BUCKETS - 1)]
        P /= P.sum()
        ot = np.asarray(ot_pmf if ot_pmf is not None else default_ot_pmf(), float)
        ot[:, 2] /= ot[:, 2].sum()
        pad = int(ot[:, :2].max()) + 1
        n = len(k)
        F = np.zeros((n + pad, n + pad))
        F[:n, :n] = P
        d = np.diag(P).copy()
        idx = np.arange(n)
        F[idx, idx] = 0.0
        for oh, oa, p in ot:
            F[idx + int(oh), idx + int(oa)] += d * p
        self.P, self.F = P, F
        self.pts = LO + np.arange(n + pad)
        I, J = np.indices(F.shape)
        self._I, self._J = I, J
        self.p_ot = float(d.sum())
        self.p_home_win = float(F[I > J].sum())
        self.home_marg, self.away_marg = F.sum(axis=1), F.sum(axis=0)
        self.exp_home = float(self.home_marg @ self.pts)
        self.exp_away = float(self.away_marg @ self.pts)

    def margin_pmf(self):
        m = (self._I - self._J).ravel()
        off = m.min()
        p = np.bincount(m - off, weights=self.F.ravel())
        return np.arange(off, off + len(p)), p

    def total_pmf(self):
        t = (self._I + self._J).ravel()
        p = np.bincount(t, weights=self.F.ravel())
        return 2 * LO + np.arange(len(p)), p

    def total(self, line):
        return _ou(*self.total_pmf(), line)

    def team_total(self, side, line):
        return _ou(self.pts, self.home_marg if side == "home" else self.away_marg, line)

    def spread(self, home_spread):
        """-5.5 = home lays 5.5. Graded on the final score including OT."""
        d, p = self.margin_pmf()
        adj = d + home_spread
        cover, push = float(p[adj > 1e-9].sum()), float(p[np.abs(adj) < 1e-9].sum())
        return {"line": float(home_spread), "cover": cover, "push": push, "fail": float(max(0.0, 1 - cover - push))}


def default_ot_pmf():
    """Fallback OT scoring: each side ~ N(10.5, 3.5), ties dropped (they'd go to 2OT)."""
    k = np.arange(0, 26)
    p = norm.pdf(k, 10.5, 3.5)
    g = np.outer(p, p)
    np.fill_diagonal(g, 0)
    g /= g.sum()
    return [[i, j, float(g[i, j])] for i in k for j in k if g[i, j] > 1e-5]


def margin_probs(mu_m, sd_m, margin_w):
    """P(regulation margin = m) for m in -70..70, per game (rows), with |margin| weights."""
    m = np.arange(-70, 71)
    p = norm.pdf(m[None, :], np.asarray(mu_m)[:, None], sd_m)
    if margin_w is not None:
        p = p * np.asarray(margin_w)[np.minimum(np.abs(m), MARGIN_BUCKETS - 1)][None, :]
    return m, p / p.sum(axis=1, keepdims=True)


def p_home_win_fast(mu_m, sd_m, margin_w, home_ot_win):
    m, p = margin_probs(mu_m, sd_m, margin_w)
    return p[:, m > 0].sum(axis=1) + p[:, m == 0].sum(axis=1) * home_ot_win


def fit_margin_weights(mu_m, act_m, sd_m, w=None, iters=60):
    """Multiplicative weights on |regulation margin| buckets so the model's average bucket
    frequencies match what actually happened (iterative proportional fitting)."""
    w = np.ones(len(mu_m)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    m, base = margin_probs(mu_m, sd_m, None)
    b = np.minimum(np.abs(m), MARGIN_BUCKETS - 1)
    act = np.bincount(np.minimum(np.abs(np.asarray(act_m).astype(int)), MARGIN_BUCKETS - 1),
                      weights=w, minlength=MARGIN_BUCKETS)
    wt = np.ones(MARGIN_BUCKETS)
    for _ in range(iters):
        G = base * wt[b][None, :]
        G /= G.sum(axis=1, keepdims=True)
        pred = np.array([(G[:, b == j].sum(axis=1)) @ w for j in range(MARGIN_BUCKETS)])
        wt *= np.where((pred > 1e-9) & (act > 0), act / np.maximum(pred, 1e-12), 1.0)
        wt /= wt[-1]
    return wt.tolist()


# ======================================================================================
# Data preparation
# ======================================================================================

def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    return obj


def _truthy(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def schedule_features(games):
    """Rest and travel for every (game, team), using only that team's earlier games.

    games: game_id, game_date, home_team, away_team and optionally neutral, in any order. Upcoming
    games can be appended to finished ones so tonight's slate gets the same features.
    Returns one row per (game_id, team): b2b, three_in_four, tz, trip, alt."""
    neutral = _truthy(games["neutral"]) if "neutral" in games else pd.Series(False, index=games.index)
    g = pd.DataFrame({"game_id": games["game_id"].astype(str).values,
                      "game_date": pd.to_datetime(games["game_date"]).values,
                      "home_team": games["home_team"].values, "neutral": neutral.values})
    rows = pd.concat([g.assign(team=g["home_team"], is_home=~g["neutral"]),
                      g.assign(team=games["away_team"].values, is_home=False)], ignore_index=True)
    rows["venue_tz"] = rows["home_team"].map(TEAM_TZ).where(~rows["neutral"])   # neutral site: location unknown
    rows = rows.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)
    out = {k: np.zeros(len(rows)) for k in ("b2b", "three_in_four", "tz", "trip", "alt")}
    dates = rows["game_date"].values.astype("datetime64[D]").astype(np.int64)
    tz, home, team = rows["venue_tz"].values, rows["is_home"].values, rows["team"].values
    for idx in rows.groupby("team").indices.values():
        trip = -1
        for n, i in enumerate(idx):
            gap = dates[i] - dates[idx[n - 1]] if n else 999
            out["b2b"][i] = float(gap == 1)
            out["three_in_four"][i] = float(n >= 2 and dates[i] - dates[idx[n - 2]] <= 3)
            if n and gap <= 2 and not np.isnan(tz[i]) and not np.isnan(tz[idx[n - 1]]):
                out["tz"][i] = min(abs(tz[i] - tz[idx[n - 1]]), 3)
            trip = -1 if (home[i] or gap > TRIP_BREAK_DAYS) else trip + 1
            out["trip"][i] = 0.0 if home[i] else float(min(max(trip, 0), TRIP_CAP))
    venue = rows["home_team"].values
    out["alt"] = (~rows["is_home"].values & ~rows["neutral"].values & np.isin(venue, list(ALTITUDE_TEAMS))
                  & (team != venue)).astype(float)
    res = rows[["game_id", "team"]].copy()
    for k, v in out.items():
        res[k] = v
    return res


def attach_schedule_features(games, feats):
    """home_<f> / away_<f> columns on a games table from schedule_features() output."""
    games = games.copy()
    f = feats.set_index(["game_id", "team"])
    for side in ("home", "away"):
        key = list(zip(games["game_id"].astype(str), games[f"{side}_team"]))
        for c in ["b2b"] + TRAVEL_FEATS:
            games[f"{side}_{c}"] = f[c].reindex(key).fillna(0).values
    return games


def load_data(data_dir=DATA_DIR, use_box="auto", min_box_coverage=0.95,
              playoff_weight=DEFAULT_PLAYOFF_WEIGHT, log=print):
    """Everything the fit and the backtest need, in one dict."""
    data_dir = Path(data_dir)
    games = read_table(data_dir / "games.parquet")
    n_all = len(games)
    games = competitive_only(games).copy()
    if len(games) < n_all:
        log(f"  ignored {n_all - len(games)} preseason / off-season / exhibition games")
    box = read_table(data_dir / "game_team_box.parquet", required=False)
    players = read_table(data_dir / "game_players.parquet", required=False)
    snapshots = read_table(data_dir / "injury_snapshots.parquet", required=False)
    games["game_date"] = pd.to_datetime(games["game_date"])

    num = ["home_score", "away_score", "n_periods", "home_ot", "away_ot"] + \
          [f"{s}_q{i}" for s in ("home", "away") for i in range(1, 5)]
    for c in num:
        games[c] = pd.to_numeric(games[c], errors="coerce")
    fin = games[_truthy(games["completed"])].dropna(subset=num).copy()
    fin = fin[(fin["home_score"] > 0) & (fin["away_score"] > 0)]
    fin = fin.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    for s in ("home", "away"):
        fin[f"{s}_reg"] = fin[[f"{s}_q{i}" for i in range(1, 5)]].sum(axis=1)
        fin[f"{s}_h1"] = fin[f"{s}_q1"] + fin[f"{s}_q2"]
    fin["n_ot"] = (fin["n_periods"] - 4).clip(lower=0)
    ok = np.where(fin["n_ot"] > 0, fin["home_reg"] == fin["away_reg"],
                  (fin["home_reg"] == fin["home_score"]) & (fin["away_reg"] == fin["away_score"]))
    if (~ok).any():
        log(f"  {int((~ok).sum())} games with inconsistent quarter scores dropped")
    fin = fin[ok].reset_index(drop=True)
    fin["type_weight"] = np.where(fin["game_type"].astype(int) == 3, playoff_weight, 1.0)
    fin["neutral"] = _truthy(fin["neutral"]) if "neutral" in fin else False

    # ---- overtime: empirical (home OT pts, away OT pts) distribution ----
    otg = fin[fin["n_ot"] > 0]
    if len(otg) >= 40:
        vc = otg.groupby(["home_ot", "away_ot"]).size() / len(otg)
        ot_pmf = [[int(h), int(a), float(p)] for (h, a), p in vc.items() if h != a]
        home_ot_win = float((otg["home_ot"] > otg["away_ot"]).mean())
    else:
        ot_pmf, home_ot_win = default_ot_pmf(), 0.52
    ot_rate = float((fin["n_ot"] > 0).mean())
    ot_mean_total = float((otg["home_ot"] + otg["away_ot"]).mean()) if len(otg) else 21.0

    # ---- rest + travel (back-to-backs, 3 in 4, time zones, road trips, altitude) ----
    fin = attach_schedule_features(fin, schedule_features(fin))
    fin["home_b2b"], fin["away_b2b"] = fin["home_b2b"].astype(int), fin["away_b2b"].astype(int)

    # ---- possessions from box scores ----
    fin["poss_reg"] = np.nan
    coverage, box_on = 0.0, False
    if box is not None and not box.empty:
        bx = box.copy()
        for c in ("fga", "oreb", "tov", "fta"):
            bx[c] = pd.to_numeric(bx[c], errors="coerce")
        bx["poss"] = bx["fga"] - bx["oreb"] + bx["tov"] + 0.44 * bx["fta"]
        g = bx.groupby("game_id")["poss"].agg(["mean", "count"])
        g = g[(g["count"] == 2) & g["mean"].between(70, 160)]
        game_poss = fin["game_id"].map(g["mean"])
        fin["poss_reg"] = game_poss * REG_MIN / (REG_MIN + OT_MIN * fin["n_ot"])
        coverage = float(fin["poss_reg"].notna().mean())
        box_on = (use_box is True) or (use_box == "auto" and coverage >= min_box_coverage)
        log(f"box scores: {'ON' if box_on else 'OFF'} ({coverage:.1%} of games have possessions)")
    else:
        log("box scores: OFF - no game_team_box file (run nba_boxscores.py)")

    # ---- long format: one row per team-game ----
    cols = ["game_id", "season", "game_date", "team", "opponent", "is_home"] + TRAVEL_COLS + \
           ["type_weight", "pts", "h1", "poss"]
    home_flag = (~fin["neutral"]).astype(int)

    def side_feats(me, opp):
        d = {}
        for f in ["b2b"] + TRAVEL_FEATS:
            d[f], d[f"opp_{f}"] = fin[f"{me}_{f}"], fin[f"{opp}_{f}"]
        return d

    h = fin.assign(team=fin["home_team"], opponent=fin["away_team"], is_home=home_flag, **side_feats("home", "away"),
                   pts=fin["home_reg"], h1=fin["home_h1"], poss=fin["poss_reg"])[cols]
    a = fin.assign(team=fin["away_team"], opponent=fin["home_team"], is_home=0, **side_feats("away", "home"),
                   pts=fin["away_reg"], h1=fin["away_h1"], poss=fin["poss_reg"])[cols]
    long = pd.concat([h, a], ignore_index=True)
    long["season"] = long["season"].astype(int)
    long["eff"] = 100 * long["pts"] / long["poss"]
    teams = sorted(set(long["team"]) | set(long["opponent"]))
    log(f"games: {len(fin):,} finished | OT rate {ot_rate:.1%} | home wins OT {home_ot_win:.1%}")
    return {"games": games, "fin": fin, "long": long, "teams": teams, "box_on": box_on, "coverage": coverage,
            "players": players if players is not None and not players.empty else None, "snapshots": snapshots,
            "ot_pmf": ot_pmf, "home_ot_win": home_ot_win, "ot_rate": ot_rate, "ot_mean_total": ot_mean_total}


# ======================================================================================
# Fitting
# ======================================================================================

class TeamRidge:
    """y ~ intercept + team + opponent + is_home + extras, weighted least squares.
    The L2 penalty applies ONLY to team/opponent terms and is worth `prior_games` games of
    league-average play, so ratings shrink toward average (never toward a reference team)."""

    def __init__(self, teams, prior_games=DEFAULT_PRIOR_GAMES, extra_cols=REST_COLS):
        self.teams = list(teams)
        self.ix = {t: i for i, t in enumerate(self.teams)}
        self.prior = prior_games
        self.extra = list(extra_cols)

    def _X(self, df):
        n, T = len(df), len(self.teams)
        X = np.zeros((n, 1 + 2 * T + 1 + len(self.extra)))
        X[:, 0] = 1.0
        r = np.arange(n)
        X[r, 1 + df["team"].map(self.ix).values] = 1.0
        X[r, 1 + T + df["opponent"].map(self.ix).values] = 1.0
        X[:, 1 + 2 * T] = df["is_home"].values
        for j, c in enumerate(self.extra):
            X[:, 2 + 2 * T + j] = df[c].values
        return X

    def fit(self, df, y, w):
        X, y, w = self._X(df), np.asarray(y, float), np.asarray(w, float)
        A = X.T @ (X * w[:, None])
        T = len(self.teams)
        A[np.arange(1, 1 + 2 * T), np.arange(1, 1 + 2 * T)] += self.prior
        self.beta = np.linalg.solve(A + 1e-9 * np.eye(len(A)), X.T @ (w * y))
        return self

    def predict(self, df):
        return self._X(df) @ self.beta

    def coef(self, name):
        T = len(self.teams)
        return float(self.beta[1 + 2 * T] if name == "is_home" else self.beta[2 + 2 * T + self.extra.index(name)])


def _weights(df, as_of, half_life, carryover):
    days_ago = (as_of - df["game_date"]).dt.days.clip(lower=0)
    seasons_back = (season_of(as_of) - df["season"]).clip(lower=0)
    return (0.5 ** (days_ago / half_life) * df["type_weight"] * carryover ** seasons_back).values


def rest_cols(travel=DEFAULT_TRAVEL):
    return TRAVEL_COLS if travel else REST_COLS


def fit_ratings(long, teams, as_of, half_life=DEFAULT_HALF_LIFE, prior_games=DEFAULT_PRIOR_GAMES,
                carryover=DEFAULT_CARRYOVER, box_on=True, travel=DEFAULT_TRAVEL):
    """Fit all rating sets on `long` (rows strictly before as_of). Returns {name: beta} + diagnostics."""
    w = _weights(long, as_of, half_life, carryover)
    w_travel = _weights(long, as_of, TRAVEL_HALF_LIFE, 1.0)
    cols = rest_cols(travel)
    extra = [c for c in cols if c not in REST_COLS]
    out = {}
    models = {"pts": ("pts", long), "h1": ("h1", long)}
    if box_on:
        ok = long["eff"].notna().values
        models.update(eff=("eff", long[ok]), pace=("poss", long[ok]))
    pos = np.arange(len(long))
    for name, (ycol, df) in models.items():
        sel = pos if len(df) == len(long) else pos[long["eff"].notna().values]
        y = df[ycol].values.astype(float)
        if not extra:
            out[name] = TeamRidge(teams, prior_games, cols).fit(df, y, w[sel]).beta.tolist()
            continue
        # 1) travel effects on long-horizon weights, 2) ratings on recency weights around them
        t = TeamRidge(teams, prior_games, cols).fit(df, y, w_travel[sel])
        tc = np.array([t.coef(c) for c in extra])
        offset = df[extra].values.astype(float) @ tc
        r = TeamRidge(teams, prior_games, REST_COLS).fit(df, y - offset, w[sel])
        out[name] = r.beta.tolist() + tc.tolist()     # same layout as TeamRidge(..., TRAVEL_COLS)
    m = TeamRidge(teams, prior_games, cols)
    m.beta = np.array(out["eff" if box_on else "pts"])
    diag = {"home_coef": m.coef("is_home"), "b2b_coef": m.coef("b2b"), "opp_b2b_coef": m.coef("opp_b2b"),
            "travel_coefs": {c: m.coef(c) for c in cols if c not in REST_COLS},
            "units": "pts per 100 poss" if box_on else "points"}
    return out, diag


def _rows(games, is_home, cols):
    """Home-perspective and away-perspective rows for a batch of games. Missing feature columns = 0."""
    def col(name):
        return games[name].astype(float).values if name in games else np.zeros(len(games))

    h = {"team": games["home_team"].values, "opponent": games["away_team"].values, "is_home": is_home}
    a = {"team": games["away_team"].values, "opponent": games["home_team"].values, "is_home": 0}
    for c in cols:
        f = c[4:] if c.startswith("opp_") else c
        own_h, own_a = col(f"home_{f}"), col(f"away_{f}")
        h[c], a[c] = (own_a, own_h) if c.startswith("opp_") else (own_h, own_a)
    return pd.DataFrame(h), pd.DataFrame(a)


def _cols_for(beta, teams):
    """Which rest/travel columns a fitted beta vector uses (older fits: back-to-backs only)."""
    n_extra = len(beta) - 2 - 2 * len(teams)
    return TRAVEL_COLS if n_extra == len(TRAVEL_COLS) else REST_COLS


def project(betas, teams, games, box_on):
    """Expected REGULATION points and 1H points for a batch of games.
    games needs home_team, away_team, home_b2b, away_b2b, optionally neutral and home_/away_ travel
    features (three_in_four, tz, trip, alt; missing = 0)."""
    neutral = _truthy(games["neutral"]).values if "neutral" in games else np.zeros(len(games), bool)
    cols = _cols_for(betas["h1"], teams)
    h, a = _rows(games, (~neutral).astype(int), cols)
    m = TeamRidge(teams, extra_cols=cols)

    def pred(name, df):
        m.beta = np.asarray(betas[name])
        return m.predict(df)

    if box_on:
        pace = (pred("pace", h) + pred("pace", a)) / 2
        mu_h, mu_a = pace * pred("eff", h) / 100, pace * pred("eff", a) / 100
    else:
        pace = np.full(len(games), np.nan)
        mu_h, mu_a = pred("pts", h), pred("pts", a)
    return pd.DataFrame({"mu_h": mu_h, "mu_a": mu_a, "h1_h": pred("h1", h), "h1_a": pred("h1", a),
                         "pace": pace}, index=games.index)


# ======================================================================================
# Out-of-sample projections (backtest + calibration)
# ======================================================================================

def project_window(D, start, end, settings, refit_days=REFIT_DAYS, game_types=(2, 3, 5)):
    """Refit every `refit_days` on games strictly before each block; project the block."""
    fin, long = D["fin"], D["long"]
    sel = fin[(fin["game_date"] >= start) & (fin["game_date"] < end) & fin["game_type"].astype(int).isin(game_types)]
    if sel.empty:
        return pd.DataFrame()
    out = []
    edges = pd.date_range(sel["game_date"].min(), sel["game_date"].max() + pd.Timedelta(days=refit_days),
                          freq=f"{refit_days}D")
    for a, b in zip(edges[:-1], edges[1:]):
        chunk = sel[(sel["game_date"] >= a) & (sel["game_date"] < b)]
        train = long[long["game_date"] < a]
        if chunk.empty or len(train) < 1000:
            continue
        betas, _ = fit_ratings(train.reset_index(drop=True), D["teams"], a, settings["half_life_days"],
                               settings["prior_games"], settings["carryover"], D["box_on"],
                               settings.get("travel", DEFAULT_TRAVEL))
        p = project(betas, D["teams"], chunk, D["box_on"])
        out.append(pd.concat([chunk, p], axis=1))
    if not out:
        return pd.DataFrame()
    r = pd.concat(out, ignore_index=True)
    r["mu_tot"], r["mu_m"] = r["mu_h"] + r["mu_a"], r["mu_h"] - r["mu_a"]
    r["mu_h1"] = r["h1_h"] + r["h1_a"]
    r["mu_2h"] = r["mu_tot"] - r["mu_h1"] + D["ot_rate"] * D["ot_mean_total"]
    r["act_tot_reg"], r["act_m_reg"] = r["home_reg"] + r["away_reg"], r["home_reg"] - r["away_reg"]
    r["act_tot"] = r["home_score"] + r["away_score"]
    r["act_h1"] = r["home_h1"] + r["away_h1"]
    r["act_2h"] = r["act_tot"] - r["act_h1"]
    return r


def calibrate(r, as_of=None, half_life=DEFAULT_HALF_LIFE):
    """Score spread (SDs) and margin weights from out-of-sample projections."""
    sig = {
        "total_reg": float((r["act_tot_reg"] - r["mu_tot"]).std()),
        "margin_reg": float((r["act_m_reg"] - r["mu_m"]).std()),
        "h1_total": float((r["act_h1"] - r["mu_h1"]).std()),
        "h2_total": float((r["act_2h"] - r["mu_2h"]).std()),
        "team_reg": float(pd.concat([r["home_reg"] - r["mu_h"], r["away_reg"] - r["mu_a"]]).std()),
    }
    w = None
    if as_of is not None:
        w = 0.5 ** ((pd.Timestamp(as_of) - r["game_date"]).dt.days.clip(lower=0) / half_life)
    mw = fit_margin_weights(r["mu_m"].values, r["act_m_reg"].values, sig["margin_reg"], w)
    return sig, mw


HALFTIME_MIN_GAMES = 300


def _halftime_design(r):
    """Regressors for 2H total / 2H margin residuals given the first-half score.
      total : 1H total vs its projection (a fast or slow game so far), |1H margin| (blowouts slow down)
      margin: 1H margin vs its projection, and the 1H margin itself (leaders ease off)"""
    one = np.ones(len(r))
    h1m = (r["home_h1"] - r["away_h1"]).values.astype(float)
    xt = np.column_stack([one, (r["act_h1"] - r["mu_h1"]).values, np.abs(h1m)])
    xm = np.column_stack([one, h1m - (r["h1_h"] - r["h1_a"]).values, h1m])
    mu_2hm = ((r["mu_h"] - r["h1_h"]) - (r["mu_a"] - r["h1_a"])).values
    yt = (r["act_2h"] - r["mu_2h"]).values
    ym = ((r["home_score"] - r["away_score"]).values - h1m) - mu_2hm
    return xt, yt, xm, ym


def _ols(X, y):
    b = np.linalg.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ y)
    sd = float(np.sqrt(np.sum((y - X @ b) ** 2) / max(len(y) - X.shape[1], 1)))
    return b.tolist(), sd


def fit_halftime(r, sig):
    """Second half given the first: least squares on out-of-sample games. With too little history,
    falls back to what the score spreads imply (1H and 2H are correlated through the full-game total)."""
    if r is not None and len(r) >= HALFTIME_MIN_GAMES:
        xt, yt, xm, ym = _halftime_design(r)
        (ct, sdt), (cm, sdm) = _ols(xt, yt), _ols(xm, ym)
        return {"fitted": True, "n_fit": int(len(r)), "total": {"coef": ct, "sd": sdt},
                "margin": {"coef": cm, "sd": sdm}}
    s1, s2, t = sig["h1_total"], sig["h2_total"], sig["total_reg"]
    cov = (t ** 2 - s1 ** 2 - s2 ** 2) / 2
    b = float(np.clip(cov / s1 ** 2, -0.5, 0.5))
    sd = float(np.sqrt(max(s2 ** 2 - b ** 2 * s1 ** 2, 0.25 * s2 ** 2)))
    return {"fitted": False, "n_fit": 0, "total": {"coef": [0.0, b, 0.0], "sd": sd},
            "margin": {"coef": [0.0, 0.0, 0.0], "sd": float(sig["margin_reg"] * np.sqrt(0.5))}}


def halftime_projection(ht, pre, h1_home, h1_away, ot_rate, ot_mean_total):
    """2H expectations at halftime. pre: pre-game regulation means mu_h, mu_a and 1H means h1_h, h1_a."""
    h1t, h1m = h1_home + h1_away, h1_home - h1_away
    pre_2h = pre["mu_h"] + pre["mu_a"] - pre["h1_h"] - pre["h1_a"] + ot_rate * ot_mean_total
    pre_2hm = (pre["mu_h"] - pre["h1_h"]) - (pre["mu_a"] - pre["h1_a"])
    xt = np.array([1.0, h1t - pre["h1_h"] - pre["h1_a"], abs(h1m)])
    xm = np.array([1.0, h1m - (pre["h1_h"] - pre["h1_a"]), h1m])
    return {"h1_home": float(h1_home), "h1_away": float(h1_away), "pre_2h": float(pre_2h), "pre_2hm": float(pre_2hm),
            "mu_2h": float(pre_2h + xt @ np.asarray(ht["total"]["coef"])), "sd_2h": float(ht["total"]["sd"]),
            "mu_2hm": float(pre_2hm + xm @ np.asarray(ht["margin"]["coef"])), "sd_2hm": float(ht["margin"]["sd"])}


def discrete_normal_spread(mu, sd, home_spread):
    """Cover / push / fail for an integer margin ~ Normal(mu, sd). -2.5 = home lays 2.5."""
    v = np.arange(np.floor(mu - 6 * sd), np.ceil(mu + 6 * sd) + 1)
    p = norm.cdf(v + 0.5, mu, sd) - norm.cdf(v - 0.5, mu, sd)
    p /= p.sum()
    adj = v + home_spread
    cover, push = float(p[adj > 1e-9].sum()), float(p[np.abs(adj) < 1e-9].sum())
    return {"line": float(home_spread), "cover": cover, "push": push, "fail": float(max(0.0, 1 - cover - push))}


def halftime_backtest(r, tests):
    """2H error before the game vs at halftime, each test season scored with a fit on the other(s)."""
    parts = []
    for s in tests:
        fit_on, score = r[r["season"] != s], r[r["season"] == s]
        if len(fit_on) < HALFTIME_MIN_GAMES or score.empty:
            continue
        xt, yt, xm, ym = _halftime_design(fit_on)
        ct, _ = _ols(xt, yt)
        cm, _ = _ols(xm, ym)
        xt, yt, xm, ym = _halftime_design(score)
        parts.append(pd.DataFrame({"pre_t": yt, "ht_t": yt - xt @ np.asarray(ct),
                                   "pre_m": ym, "ht_m": ym - xm @ np.asarray(cm)}))
    if not parts:
        return None
    d = pd.concat(parts).abs()
    return {"n_games": int(len(d)), "h2_mae_pregame": float(d["pre_t"].mean()), "h2_mae_halftime": float(d["ht_t"].mean()),
            "h2_margin_mae_pregame": float(d["pre_m"].mean()), "h2_margin_mae_halftime": float(d["ht_m"].mean())}


def walk_forward(data_dir=DATA_DIR, n_seasons=2, half_life_days=DEFAULT_HALF_LIFE, prior_games=DEFAULT_PRIOR_GAMES,
                 carryover=DEFAULT_CARRYOVER, use_box="auto", margin_cal=True, refit_days=REFIT_DAYS, D=None, log=print,
                 travel=DEFAULT_TRAVEL):
    """Score the last `n_seasons` complete seasons; each block is predicted by a model trained only on earlier games."""
    D = D or load_data(data_dir, use_box=use_box, log=log)
    settings = dict(half_life_days=half_life_days, prior_games=prior_games, carryover=carryover, travel=int(travel))
    fin = D["fin"]
    counts = fin[fin["game_type"].astype(int) == 2].groupby("season").size()
    complete = [s for s, n in counts.items() if n >= 1000]
    first = int(fin["season"].min())
    tests = [s for s in complete[-n_seasons:] if s != first]
    parts = []
    for s in tests:
        sg = fin[fin["season"] == s]
        parts.append(project_window(D, sg["game_date"].min(), sg["game_date"].max() + pd.Timedelta(days=1),
                                    settings, refit_days, game_types=(2,)))
    r = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if r.empty:
        raise RuntimeError("Not enough history for a backtest (need 2+ seasons).")
    sig, mw = calibrate(r)
    r["exp_final"] = r["mu_tot"] + D["ot_rate"] * D["ot_mean_total"]
    r["p_win"] = p_home_win_fast(r["mu_m"].values, sig["margin_reg"], mw if margin_cal else None, D["home_ot_win"])
    r["y_win"] = (r["home_score"] > r["away_score"]).astype(int)
    first_day = r.groupby("season")["game_date"].transform("min")
    r["early"] = (r["game_date"] - first_day).dt.days < 30
    out = summarize_backtest(r, tests, sig)
    out["settings"] = settings
    ht = halftime_backtest(r, tests)
    if ht:
        out["halftime"] = ht
    if D["players"] is not None and len(tests) >= 2:
        av = nav.build(D["fin"], D["players"])
        ra = nav.attach_M(r, av["M"])
        adj = []
        for s in tests:  # coefficients fitted on the OTHER test season(s): no peeking
            c = nav.fit_coefs(ra[ra["season"] != s])
            if c:
                adj.append(nav.apply(ra[ra["season"] == s], c))
        if adj:
            ra = pd.concat(adj)
            ra["exp_final"] = ra["mu_tot"] + D["ot_rate"] * D["ot_mean_total"]
            ra["p_win"] = p_home_win_fast(ra["mu_m"].values, sig["margin_reg"], mw if margin_cal else None,
                                          D["home_ot_win"])
            out["with_availability"] = {
                "n_games": int(len(ra)),
                "win_brier": float(np.mean((ra["p_win"] - ra["y_win"]) ** 2)),
                "total_mae": float((ra["act_tot"] - ra["exp_final"]).abs().mean()),
                "margin_mae": float((ra["act_m_reg"] - ra["mu_m"]).abs().mean()),
                "team_mae": float(pd.concat([ra["home_reg"] - ra["mu_h"], ra["away_reg"] - ra["mu_a"]]).abs().mean()),
                "baseline_same_games": {
                    "win_brier": float(np.mean((r.loc[ra.index, "p_win"] - r.loc[ra.index, "y_win"]) ** 2)),
                    "margin_mae": float((r.loc[ra.index, "act_m_reg"] - r.loc[ra.index, "mu_m"]).abs().mean())}}
    return out


def summarize_backtest(r, tests, sig):
    def brier(p, y):
        return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))
    exp_final = r["exp_final"]
    out = {"seasons": [str(s) for s in tests], "n_games": int(len(r)),
           "win_brier": brier(r["p_win"], r["y_win"]),
           "const_brier": brier(np.full(len(r), r["y_win"].mean()), r["y_win"]),
           "win_accuracy": float(((r["p_win"] >= 0.5) == (r["y_win"] == 1)).mean()),
           "total_mae": float((r["act_tot"] - exp_final).abs().mean()),
           "margin_mae": float((r["act_m_reg"] - r["mu_m"]).abs().mean()),
           "team_mae": float(pd.concat([(r["home_reg"] - r["mu_h"]), (r["away_reg"] - r["mu_a"])]).abs().mean()),
           "h1_mae": float((r["act_h1"] - r["mu_h1"]).abs().mean()),
           "h2_mae": float((r["act_2h"] - r["mu_2h"]).abs().mean()),
           "early_win_brier": brier(r.loc[r["early"], "p_win"], r.loc[r["early"], "y_win"]),
           "early_n": int(r["early"].sum()), "sigma": sig}
    # vs the market, where pre-game lines were captured
    tl = pd.to_numeric(r.get("total_line"), errors="coerce") if "total_line" in r else pd.Series(dtype=float)
    if tl.notna().sum() >= 50:
        k = tl.notna()
        out["market_total_mae"] = float((r.loc[k, "act_tot"] - tl[k]).abs().mean())
        out["model_total_mae_same_games"] = float((r.loc[k, "act_tot"] - exp_final[k]).abs().mean())
        out["market_total_n"] = int(k.sum())
    bins = [0, .2, .3, .4, .5, .6, .7, .8, 1]
    cal = r.assign(bucket=pd.cut(r["p_win"], bins)).groupby("bucket", observed=True).agg(
        predicted=("p_win", "mean"), actual=("y_win", "mean"), n=("y_win", "size")).reset_index()
    cal["bucket"] = cal["bucket"].astype(str)
    out["calibration"] = cal.to_dict("records")
    out["raw"] = r
    return out


# ======================================================================================
# Training -> artifact
# ======================================================================================

def _lines_for(g, odds_rows):
    """ESPN line from the schedule, overridden by The Odds API when present."""
    keys = ["total_line", "spread_home", "ml_home", "ml_away", "spread_home_price", "spread_away_price",
            "total_over_price", "total_under_price", "home_tt_line", "home_tt_over_price", "home_tt_under_price",
            "away_tt_line", "away_tt_over_price", "away_tt_under_price", "h1_total_line", "h1_total_over_price",
            "h1_total_under_price", "h2_total_line", "h2_total_over_price", "h2_total_under_price"]
    lines = {k: g.get(k) for k in keys}
    lines["book"] = g.get("odds_provider")
    o = odds_rows.get((g["home_team"], g["away_team"], g["game_date"]))
    if o:
        for k in keys + ["book"]:
            if o.get(k) is not None:
                lines[k] = o[k]
    return lines


def build_slate(upcoming, last_played, odds_rows, fin=None):
    """Upcoming games with lines and rest/travel flags. With `fin` (finished games) the full travel
    features are computed from each team's real schedule, chaining through the upcoming days."""
    upcoming = sorted(upcoming, key=lambda g: (g["game_date"], g.get("start_utc") or ""))
    feats = None
    if fin is not None and upcoming:
        up = pd.DataFrame([{"game_id": str(g["game_id"]), "game_date": g["game_date"], "home_team": g["home_team"],
                            "away_team": g["away_team"], "neutral": bool(g.get("neutral"))} for g in upcoming])
        hist = fin[["game_id", "game_date", "home_team", "away_team", "neutral"]]
        hist = hist[(hist["game_date"] >= pd.Timestamp(up["game_date"].min()) - pd.Timedelta(days=20))
                    & ~hist["game_id"].astype(str).isin(up["game_id"])]
        feats = schedule_features(pd.concat([hist, up], ignore_index=True)).set_index(["game_id", "team"])
    out, played = [], dict(last_played)
    for g in upcoming:
        flags, travel = {}, {}
        for side in ("home", "away"):
            team = g[f"{side}_team"]
            if feats is not None:
                f = feats.loc[(str(g["game_id"]), team)]
                flags[f"{side}_b2b"] = bool(f["b2b"])
                travel[side] = {k: float(f[k]) for k in TRAVEL_FEATS}
            else:
                prev = played.get(team)
                flags[f"{side}_b2b"] = bool(prev and (pd.Timestamp(g["game_date"]) - pd.Timestamp(prev)).days == 1)
        out.append({"game_id": str(g["game_id"]), "date": g["game_date"], "start_utc": g.get("start_utc"),
                    "home": g["home_team"], "away": g["away_team"], "neutral": bool(g.get("neutral")),
                    **flags, "travel": travel or None, "lines": _lines_for(g, odds_rows)})
        played[g["home_team"]] = played[g["away_team"]] = g["game_date"]
    return out


def train_artifact(data_dir=DATA_DIR, half_life_days=DEFAULT_HALF_LIFE, prior_games=DEFAULT_PRIOR_GAMES,
                   carryover=DEFAULT_CARRYOVER, use_box="auto", as_of=None, upcoming=None, odds=None,
                   injuries=None, log=print, travel=DEFAULT_TRAVEL):
    D = load_data(data_dir, use_box=use_box, log=log)
    fin, long, teams = D["fin"], D["long"], D["teams"]
    last_date = fin["game_date"].max()
    as_of = pd.Timestamp(as_of) if as_of is not None else max(last_date, pd.Timestamp(date.today()))
    settings = dict(half_life_days=half_life_days, prior_games=prior_games, carryover=carryover, travel=int(travel))

    betas, diag = fit_ratings(long, teams, as_of, half_life_days, prior_games, carryover, D["box_on"], travel)
    log(f"fit: home {diag['home_coef']:+.2f} | b2b {diag['b2b_coef']:+.2f} | opp b2b {diag['opp_b2b_coef']:+.2f} "
        f"({diag['units']})")
    if diag["travel_coefs"]:
        log("travel: " + " | ".join(f"{k} {v:+.2f}" for k, v in diag["travel_coefs"].items()))
    else:
        log("travel: OFF (back-to-backs only)")

    # score spread + margin weights from the most recent year of out-of-sample predictions
    r = project_window(D, last_date - pd.Timedelta(days=CALIBRATION_DAYS), last_date + pd.Timedelta(days=1), settings)
    if len(r) >= 300:
        sig, mw = calibrate(r, as_of, half_life_days)
    else:  # too little history for out-of-sample calibration: typical NBA values
        log("  [warn] under 300 out-of-sample games - using default score spreads")
        sig = {"total_reg": 18.5, "margin_reg": 13.0, "h1_total": 11.0, "h2_total": 12.5, "team_reg": 11.5}
        mw = None
    log("sd: " + " | ".join(f"{k} {v:.1f}" for k, v in sig.items()) + f"  ({len(r):,} out-of-sample games)")
    if mw:
        log("margin weights |m| 0..15+: " + ", ".join(f"{x:.2f}" for x in mw))
    halftime = fit_halftime(r if len(r) else None, sig)
    ct, cm = halftime["total"]["coef"], halftime["margin"]["coef"]
    log(f"halftime: 2H total {ct[1]:+.2f} x 1H surprise {ct[2]:+.2f} x |1H margin| (sd {halftime['total']['sd']:.1f}) | "
        f"2H margin {cm[1]:+.2f} x 1H surprise {cm[2]:+.2f} x 1H margin (sd {halftime['margin']['sd']:.1f})"
        + ("" if halftime["fitted"] else "  [defaults: too few games]"))

    recent = fin[fin["game_date"] >= last_date - pd.Timedelta(days=365)]
    active = sorted(set(recent["home_team"]) | set(recent["away_team"]))
    cur = fin[(fin["season"] == season_of(as_of)) & (fin["game_type"].astype(int) == 2)]
    gp = pd.concat([cur["home_team"], cur["away_team"]]).value_counts()
    last_played = pd.concat([fin[["game_date", "home_team"]].rename(columns={"home_team": "t"}),
                             fin[["game_date", "away_team"]].rename(columns={"away_team": "t"})]
                            ).groupby("t")["game_date"].max().dt.strftime("%Y-%m-%d").to_dict()
    odds_rows = {(o["home_team"], o["away_team"], o["game_date"]): o for o in (odds or {}).get("games", [])}

    # ---- player availability ----
    availability = {"on": False}
    if D["players"] is not None and len(r) >= 300:
        av = nav.build(fin, D["players"])
        coefs = nav.fit_coefs(nav.attach_M(r, av["M"]))
        if coefs and coefs["b_own"] < 0:
            rates, counts = nav.play_rates(D["snapshots"], av["panel"], fin)
            availability = {"on": True, **coefs, "play_rates": rates, "status_counts": counts,
                            "replacement_value": av["prior"],
                            "injuries_fetched_at": (injuries or {}).get("fetched_at"),
                            "official_report": (injuries or {}).get("official_report"),
                            "teams": nav.live_teams(av, injuries, rates, active)}
            log(f"availability: ON | lost value -> own pts {coefs['b_own']:+.2f}, opp pts {coefs['b_opp']:+.2f} "
                f"| team MAE {coefs['team_mae_before']:.2f} -> {coefs['team_mae_after']:.2f} (known minutes)")
        else:
            log("availability: OFF - the fit didn't find a sensible effect (check game_players data)")
    else:
        log("availability: OFF - no player box scores (run nba_boxscores.py)")
    upcoming = [g for g in (upcoming or []) if pd.Timestamp(g["game_date"]) > last_date - pd.Timedelta(days=1)]

    art = {
        "version": ARTIFACT_VERSION,
        "meta": {"fitted_at": datetime.now().isoformat(timespec="seconds"), "trained_through": str(last_date.date()),
                 "as_of": str(as_of.date()), "n_games": int(len(fin)), **settings, "box_on": D["box_on"],
                 "box_coverage": D["coverage"], **diag, "active_teams": active,
                 "season_gp": {t: int(gp.get(t, 0)) for t in active}, "calibration_games": int(len(r)),
                 "odds_fetched_at": (odds or {}).get("fetched_at")},
        "teams": teams, "betas": betas, "box_on": D["box_on"], "sigma": sig, "margin_weights": mw,
        "ot_pmf": D["ot_pmf"], "home_ot_win": D["home_ot_win"], "ot_rate": D["ot_rate"],
        "ot_mean_total": D["ot_mean_total"], "last_played": last_played,
        "slate": build_slate(upcoming, last_played, odds_rows, fin),
        "availability": availability, "halftime": halftime,
    }
    return _clean(art)


def save_artifact(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, separators=(",", ":")))
    return path


def load_artifact(path):
    return json.loads(Path(path).read_text())


# ======================================================================================
# Prediction
# ======================================================================================

class Predictor:
    """Everything the app needs, from the artifact alone."""

    def __init__(self, art):
        if art.get("version", 0) < ARTIFACT_VERSION:
            raise ValueError("Model file is from an older version. Run `python refresh.py --skip-fetch`.")
        self.art, self.meta = art, art["meta"]
        self.teams = art["teams"]
        self.active_teams = self.meta.get("active_teams") or self.teams
        self.betas, self.box_on = art["betas"], art["box_on"]
        self.sig, self.mw = art["sigma"], art["margin_weights"]
        self.slate = art.get("slate", [])
        self.av = art.get("availability") or {"on": False}

    # ---------- availability ----------
    def roster(self, team):
        return (self.av.get("teams") or {}).get(team, []) if self.av.get("on") else []

    def availability(self, team, overrides=None):
        """Margin effect for `team` from tonight's availability, and the players worth naming."""
        if not self.av.get("on"):
            return {"M": 0.0, "adj": 0.0, "players": []}
        ps = self.roster(team)
        M = nav.team_lost_value(ps, overrides)
        k = self.av["b_own"] - self.av["b_opp"]
        named = [p for p in nav.player_impacts(ps, self.av, overrides)
                 if abs(p["impact"]) >= nav.IMPACT_SHOW and p["status"] not in ("Available", "Not on roster")]
        named.sort(key=lambda p: p["impact"])
        return {"M": M, "adj": float(k * M), "players": named}

    def resolve(self, team):
        if team not in self.teams:
            raise ValueError(f"Unknown team '{team}'.")
        return team

    def expected(self, home, away, home_b2b=False, away_b2b=False, neutral=False, overrides=None, use_av=True,
                 travel=None):
        """travel: {"home": {three_in_four, tz, trip, alt}, "away": {...}} from the slate (None = no travel)."""
        g = pd.DataFrame({"home_team": [self.resolve(home)], "away_team": [self.resolve(away)],
                          "home_b2b": [int(home_b2b)], "away_b2b": [int(away_b2b)], "neutral": [bool(neutral)]})
        for side in ("home", "away"):
            for f in TRAVEL_FEATS:
                g[f"{side}_{f}"] = float(((travel or {}).get(side) or {}).get(f, 0.0))
        e = project(self.betas, self.teams, g, self.box_on).iloc[0].copy()
        if use_av and self.av.get("on"):
            mh = nav.team_lost_value(self.roster(home), overrides)
            ma = nav.team_lost_value(self.roster(away), overrides)
            bo, bp = self.av["b_own"], self.av["b_opp"]
            for side, d in (("h", bo * mh + bp * ma), ("a", bo * ma + bp * mh)):
                base = e[f"mu_{side}"]
                e[f"mu_{side}"] = base + d
                e[f"h1_{side}"] = e[f"h1_{side}"] * (base + d) / base   # same share of the change in the 1st half
        return e

    def travel_effect(self, home, away, home_b2b=False, away_b2b=False, neutral=False, travel=None):
        """Margin points each team gains (+) or loses (-) from its OWN travel situation, plus short notes.
        Back-to-backs are shown separately on the cards, so they're not counted here."""
        out = {"home": {"adj": 0.0, "notes": []}, "away": {"adj": 0.0, "notes": []}}
        if not travel or _cols_for(self.betas["h1"], self.teams) is REST_COLS:
            return out
        args = (home, away, home_b2b, away_b2b, neutral, None, False)
        e0 = self.expected(*args)
        m0 = e0["mu_h"] - e0["mu_a"]
        for side, sign in (("home", 1), ("away", -1)):
            f = travel.get(side) or {}
            e = self.expected(*args, travel={side: f})
            out[side]["adj"] = float(sign * ((e["mu_h"] - e["mu_a"]) - m0))
            notes = []
            if f.get("three_in_four"):
                notes.append("3 in 4 nights")
            if f.get("tz"):
                notes.append(f"{int(f['tz'])} time zone{'s' if f['tz'] > 1 else ''}")
            if f.get("trip"):
                notes.append(f"road game {int(f['trip']) + 1}" + ("+" if f["trip"] >= TRIP_CAP else ""))
            if f.get("alt"):
                notes.append("altitude")
            out[side]["notes"] = notes
        return out

    def predict_game(self, home, away, home_b2b=False, away_b2b=False, neutral=False, total_line=225.5,
                     home_spread=-3.5, home_team_line=None, away_team_line=None, h1_line=None, h2_line=None,
                     overrides=None, use_av=True, travel=None):
        e = self.expected(home, away, home_b2b, away_b2b, neutral, overrides, use_av, travel)
        gd = GameDist(e["mu_h"], e["mu_a"], self.sig, self.mw, self.art["ot_pmf"])
        mu_h1 = e["h1_h"] + e["h1_a"]
        mu_2h = e["mu_h"] + e["mu_a"] - mu_h1 + self.art["ot_rate"] * self.art["ot_mean_total"]
        htl = home_team_line if home_team_line is not None else round(gd.exp_home) + 0.5
        atl = away_team_line if away_team_line is not None else round(gd.exp_away) + 0.5
        h1l = h1_line if h1_line is not None else round(mu_h1) + 0.5
        h2l = h2_line if h2_line is not None else round(mu_2h) + 0.5
        d, mp = gd.margin_pmf()
        keep = (np.abs(d) <= 30) & (d != 0)
        return {
            "home": home, "away": away, "pace": None if pd.isna(e["pace"]) else float(e["pace"]),
            "reg_home": float(e["mu_h"]), "reg_away": float(e["mu_a"]),
            "exp_home": gd.exp_home, "exp_away": gd.exp_away, "exp_total": gd.exp_home + gd.exp_away,
            "exp_margin": gd.exp_home - gd.exp_away, "p_home_win": gd.p_home_win, "p_ot": gd.p_ot,
            "exp_h1_home": float(e["h1_h"]), "exp_h1_away": float(e["h1_a"]), "exp_h1": float(mu_h1),
            "exp_h2": float(mu_2h),
            "total": gd.total(total_line), "spread": gd.spread(home_spread),
            "alt_spreads": {s: gd.spread(s) for s in sorted({home_spread - 4, home_spread - 2, home_spread + 2,
                                                              home_spread + 4})},
            "home_total": gd.team_total("home", htl), "away_total": gd.team_total("away", atl),
            "h1_total": discrete_normal_ou(mu_h1, self.sig["h1_total"], h1l),
            "h2_total": discrete_normal_ou(mu_2h, self.sig["h2_total"], h2l),
            "margin_pmf": {"margin": d[keep].tolist(), "p": mp[keep].tolist()},
            "avail": {"home": self.availability(home, overrides) if use_av else None,
                      "away": self.availability(away, overrides) if use_av else None},
            "travel": self.travel_effect(home, away, home_b2b, away_b2b, neutral, travel),
        }

    # ---------- joint outcomes (correlated stakes) ----------
    def sample_scores(self, home, away, home_b2b=False, away_b2b=False, neutral=False, n=20000, seed=0,
                      overrides=None, use_av=True, travel=None):
        """Simulated final scores (incl. OT) and first-half totals for one game, drawn from the same joint
        grid the markets are priced from. The 1H total is drawn given the full-game total, with the
        correlation implied by the fitted score spreads (2H = full game - 1H)."""
        e = self.expected(home, away, home_b2b, away_b2b, neutral, overrides, use_av, travel)
        gd = GameDist(e["mu_h"], e["mu_a"], self.sig, self.mw, self.art["ot_pmf"])
        rng = np.random.default_rng(seed)
        F = gd.F.ravel()
        idx = rng.choice(F.size, size=n, p=F / F.sum())
        i, j = np.unravel_index(idx, gd.F.shape)
        h, a = gd.pts[i].astype(float), gd.pts[j].astype(float)
        tv, tp = gd.total_pmf()
        mu_t = float(tv @ tp)
        var_t = float(((tv - mu_t) ** 2) @ tp)
        s1, s2 = self.sig["h1_total"], self.sig["h2_total"]
        cov = float(np.clip((var_t + s1 ** 2 - s2 ** 2) / 2, 0.0, 0.95 * s1 * np.sqrt(var_t)))
        mu1 = float(e["h1_h"] + e["h1_a"])
        h1 = np.round(rng.normal(mu1 + cov / var_t * (h + a - mu_t), np.sqrt(max(s1 ** 2 - cov ** 2 / var_t, 1.0))))
        return {"home": h, "away": a, "h1": h1}

    # ---------- halftime ----------
    def pregame(self, home, away, home_b2b=False, away_b2b=False, neutral=False, overrides=None, travel=None):
        e = self.expected(home, away, home_b2b, away_b2b, neutral, overrides, True, travel)
        return {"mu_h": float(e["mu_h"]), "mu_a": float(e["mu_a"]), "h1_h": float(e["h1_h"]), "h1_a": float(e["h1_a"])}

    def halftime(self, h1_home, h1_away, pre, total_line=None, home_spread=None):
        """Second-half markets once the first-half score is known. pre: Predictor.pregame(...) or the
        same four numbers saved in the prediction log before tip-off."""
        ht = self.art.get("halftime") or fit_halftime(None, self.sig)
        p = halftime_projection(ht, pre, h1_home, h1_away, self.art["ot_rate"], self.art["ot_mean_total"])
        tl = total_line if total_line is not None else round(p["mu_2h"]) + 0.5
        sp = home_spread if home_spread is not None else -(round(p["mu_2hm"]) + 0.5)
        h1m = h1_home - h1_away
        fin = discrete_normal_spread(p["mu_2hm"] + h1m, p["sd_2hm"], 0.0)
        p.update(total=discrete_normal_ou(p["mu_2h"], p["sd_2h"], tl), spread=discrete_normal_spread(
            p["mu_2hm"], p["sd_2hm"], sp), p_home_win=float(fin["cover"] + fin["push"] * self.art["home_ot_win"]),
            fitted=bool(ht.get("fitted")))
        return p

    def team_ratings(self):
        T = len(self.teams)
        t = pd.DataFrame(index=self.teams)
        if self.box_on:
            e, p = np.asarray(self.betas["eff"]), np.asarray(self.betas["pace"])
            t["ORtg"] = e[0] + e[1:1 + T]
            t["DRtg"] = e[0] + e[1 + T:1 + 2 * T]
            t["Net"] = t["ORtg"] - t["DRtg"]
            t["Pace"] = p[0] + (p[1:1 + T] + p[1 + T:1 + 2 * T]) / 2
        else:
            b = np.asarray(self.betas["pts"])
            t["Pts for"] = b[0] + b[1:1 + T]
            t["Pts against"] = b[0] + b[1 + T:1 + 2 * T]
            t["Net"] = t["Pts for"] - t["Pts against"]
        h1 = np.asarray(self.betas["h1"])
        t["1H net"] = h1[1:1 + T] - h1[1 + T:1 + 2 * T]
        gp = self.meta.get("season_gp", {})
        t["GP"] = [gp.get(x, 0) for x in t.index]
        return t.loc[[x for x in self.teams if x in self.active_teams]].sort_values("Net", ascending=False)
