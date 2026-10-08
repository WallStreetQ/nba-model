"""
nba_availability.py
--------------------
Player availability: how much a team's projection moves when players sit (the NBA version of the
NHL goalie module). Everything is leak-free: each game only uses information from earlier games.

Ingredients (from data/game_players.parquet, the box scores)
  value     each player's production per minute (Hollinger Game Score / minute), recency-weighted
            and shrunk toward replacement level, so a 3-game rookie isn't treated as a star.
  baseline  his share of the team's minutes over recent team games, counting games he missed as
            zero. This is the role the team RATING already assumes.
  tonight   minutes he's expected to play tonight = P(plays) x his usual minutes when he plays,
            then rescaled so the team's minutes add up to 240 (sitting players' minutes go to teammates).

  lost value  M = sum over players of  value x (baseline minutes - tonight's minutes)

  Projection adjustment (regulation points), fitted on out-of-sample games:
      team points     += b_own x M_team        (b_own < 0: losing value costs points)
      opponent points += b_opp x M_team        (b_opp > 0: weaker defense)

Status -> P(plays): measured from your own injury snapshots joined to box scores once there are
enough of them (150+ per status); until then sensible defaults.
"""

import numpy as np
import pandas as pd

BASE_SPAN = 20            # team games: minutes baseline the rating assumes
PLAY_SPAN = 10            # games played: usual minutes when he plays
VALUE_SPAN = 60           # games played: per-minute value
VALUE_PRIOR_MIN = 400     # minutes of replacement-level play mixed into every value
TEAM_MIN = 240
DEFAULT_PLAY_RATES = {"Available": 1.0, "Probable": 0.93, "Questionable": 0.55, "Day-To-Day": 0.6,
                      "Doubtful": 0.12, "Out": 0.0, "Suspension": 0.0}
MIN_STATUS_SAMPLES = 150
IMPACT_SHOW = 0.5         # margin points before a player is named in the app


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df else pd.Series(0.0, index=df.index)


def game_score(p):
    """Hollinger Game Score from a box-score line."""
    n = lambda c: _num(p, c)  # noqa: E731
    return (n("pts") + 0.4 * n("fgm") - 0.7 * n("fga") - 0.4 * (n("fta") - n("ftm")) + 0.7 * n("oreb")
            + 0.3 * n("dreb") + n("stl") + 0.7 * n("ast") + 0.7 * n("blk") - 0.4 * n("pf") - n("tov"))


def _decayed_sum(x, alpha):
    """sum_k (1-alpha)^k x_{t-k} for a sorted series (includes the current row)."""
    n = np.arange(1, len(x) + 1)
    return x.ewm(alpha=alpha, adjust=True).mean() * (1 - (1 - alpha) ** n) / alpha


def build(fin, players):
    """Panel of every (team, player, team-game) between his first and last appearance for that team,
    with leak-free baseline minutes and value, plus per team-game lost value M (actual minutes)."""
    pl = players.copy()
    pl["player_id"] = pl["player_id"].astype(str)
    pl = pl.merge(fin[["game_id", "game_date", "n_ot"]], on="game_id")
    pl["minutes"] = _num(pl, "minutes")
    pl["reg_min"] = pl["minutes"] * TEAM_MIN / (TEAM_MIN + 25 * pl["n_ot"])
    pl["gmsc"] = game_score(pl)

    # replacement level: per-minute Game Score of low-minute players
    per = pl[pl["minutes"] > 0].groupby("player_id").agg(m=("minutes", "mean"), g=("gmsc", "sum"), t=("minutes", "sum"))
    low = per[per["m"] < 12]
    prior = float(low["g"].sum() / low["t"].sum()) if low["t"].sum() > 1000 else 0.2

    # value per minute, by player across teams (post-game state, then shifted to pre-game)
    played = pl[pl["minutes"] > 0].sort_values(["player_id", "game_date", "game_id"]).copy()
    a = 2 / (VALUE_SPAN + 1)
    g = played.groupby("player_id")
    sg = g["gmsc"].transform(lambda x: _decayed_sum(x, a))
    sm = g["minutes"].transform(lambda x: _decayed_sum(x, a))
    played["v_post"] = (sg + VALUE_PRIOR_MIN * prior) / (sm + VALUE_PRIOR_MIN)
    ap = 2 / (PLAY_SPAN + 1)
    played["mwp_post"] = g["reg_min"].transform(lambda x: x.ewm(alpha=ap, adjust=True).mean())

    # team schedule and the zero-filled panel
    sched = pd.concat([fin[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"}),
                       fin[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"})])
    sched = sched.sort_values(["team", "game_date", "game_id"])
    sched["tg"] = sched.groupby("team").cumcount()
    pl = pl.merge(sched[["game_id", "team", "tg"]], on=["game_id", "team"])
    span = pl.groupby(["team", "player_id"])["tg"].agg(["min", "max"]).reset_index()
    full = span.merge(sched, on="team")
    full = full[(full["tg"] >= full["min"]) & (full["tg"] <= full["max"])]
    panel = full[["team", "player_id", "tg", "game_id", "game_date"]].merge(
        pl[["game_id", "team", "player_id", "reg_min", "name"]], on=["game_id", "team", "player_id"], how="left")
    panel["reg_min"] = panel["reg_min"].fillna(0.0)
    panel = panel.sort_values(["team", "player_id", "tg"]).reset_index(drop=True)
    ab = 2 / (BASE_SPAN + 1)
    panel["base_post"] = panel.groupby(["team", "player_id"])["reg_min"].transform(
        lambda x: x.ewm(alpha=ab, adjust=False).mean())
    panel["base"] = panel.groupby(["team", "player_id"])["base_post"].shift(1).fillna(0.0)

    # pre-game value: latest value strictly before this game date
    vv = played[["player_id", "game_date", "v_post"]].sort_values("game_date")
    panel = pd.merge_asof(panel.sort_values("game_date"), vv, on="game_date", by="player_id",
                          allow_exact_matches=False).rename(columns={"v_post": "v"})
    panel["v"] = panel["v"].fillna(prior)
    panel["lost"] = panel["v"] * (panel["base"] - panel["reg_min"])
    M = panel.groupby(["game_id", "team"])["lost"].sum().rename("M").reset_index()
    return {"panel": panel, "played": played, "sched": sched, "M": M, "prior": prior}


def attach_M(r, M):
    """Add M_home / M_away (lost value) to rows with home_team / away_team / game_id."""
    m = M.set_index(["game_id", "team"])["M"]
    r = r.copy()
    r["M_home"] = m.reindex(list(zip(r["game_id"], r["home_team"]))).values
    r["M_away"] = m.reindex(list(zip(r["game_id"], r["away_team"]))).values
    return r


def fit_coefs(r):
    """Least squares on out-of-sample residuals:  resid_team = b_own * M_team + b_opp * M_opponent."""
    ok = r["M_home"].notna() & r["M_away"].notna()
    r = r[ok]
    if len(r) < 300:
        return None
    y = np.concatenate([r["home_reg"] - r["mu_h"], r["away_reg"] - r["mu_a"]])
    X = np.column_stack([np.concatenate([r["M_home"], r["M_away"]]), np.concatenate([r["M_away"], r["M_home"]])])
    b = np.linalg.solve(X.T @ X + 1e-6 * np.eye(2), X.T @ y)
    before = np.abs(y).mean()
    after = np.abs(y - X @ b).mean()
    return {"b_own": float(b[0]), "b_opp": float(b[1]), "n_fit": int(len(r)),
            "team_mae_before": float(before), "team_mae_after": float(after)}


def apply(r, coefs):
    """Adjust projections with (actual) availability; used by the backtest."""
    r = r.copy()
    mh, ma = r["M_home"].fillna(0), r["M_away"].fillna(0)
    dh = coefs["b_own"] * mh + coefs["b_opp"] * ma
    da = coefs["b_own"] * ma + coefs["b_opp"] * mh
    r["mu_h"], r["mu_a"] = r["mu_h"] + dh, r["mu_a"] + da
    r["mu_tot"], r["mu_m"] = r["mu_h"] + r["mu_a"], r["mu_h"] - r["mu_a"]
    return r


def play_rates(snapshots, panel, fin):
    """P(plays | status) from injury snapshots joined to who actually logged minutes."""
    rates = dict(DEFAULT_PLAY_RATES)
    counts = {}
    if snapshots is None or snapshots.empty:
        return rates, counts
    s = snapshots.copy()
    s["player_id"] = s["player_id"].astype(str)
    games = panel[["game_id", "team", "player_id", "reg_min", "game_date"]].copy()
    games["date"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")
    j = s.merge(games, on=["date", "team", "player_id"], how="inner")
    for status, grp in j.groupby("status"):
        counts[status] = int(len(grp))
        if len(grp) >= MIN_STATUS_SAMPLES:
            rates[status] = float((grp["reg_min"] > 0).mean())
    return rates, counts


def live_teams(av, injuries, rates, carry_teams):
    """Current availability inputs per team for the artifact."""
    panel, played = av["panel"], av["played"]
    ab = 2 / (BASE_SPAN + 1)
    last = panel.sort_values("tg").groupby(["team", "player_id"]).tail(1)
    team_last_tg = av["sched"].groupby("team")["tg"].max()
    last = last.assign(base_now=last["base_post"] * (1 - ab) ** (last["team"].map(team_last_tg) - last["tg"]))
    latest = played.sort_values("game_date").groupby("player_id").tail(1).set_index("player_id")
    names = played.sort_values("game_date").groupby("player_id")["name"].last()
    listed = {str(p["player_id"]): p for p in (injuries or {}).get("players", []) if p.get("player_id")}
    rosters = (injuries or {}).get("rosters") or {}

    out = {}
    for team in carry_teams:
        roster = {str(p["player_id"]): p.get("name") for p in rosters.get(team, [])}
        tl = last[last["team"] == team]
        if not roster:  # no roster feed: players who appeared in the team's latest game
            roster = {pid: None for pid in tl.loc[tl["tg"] == team_last_tg.get(team, -1), "player_id"]}
        base = dict(zip(tl["player_id"], tl["base_now"]))
        rows = []
        for pid in set(roster) | {p for p, b in base.items() if b > 0.5}:
            on_roster = pid in roster
            inj = listed.get(pid)
            status = inj["status"] if inj else ("Available" if on_roster else "Not on roster")
            p_play = 0.0 if not on_roster else rates.get(status, DEFAULT_PLAY_RATES.get(status, 0.6))
            mwp = float(latest.loc[pid, "mwp_post"]) if pid in latest.index else 0.0
            v = float(latest.loc[pid, "v_post"]) if pid in latest.index else av["prior"]
            b = float(base.get(pid, 0.0))
            if b < 0.05 and mwp < 0.5:
                continue
            rows.append({"player_id": pid, "name": (roster.get(pid) or (inj or {}).get("name") or names.get(pid, pid)),
                         "base": round(b, 3), "mwp": round(mwp, 3), "v": round(v, 4), "status": status,
                         "p_play": round(float(p_play), 3), "on_roster": on_roster,
                         "detail": (inj or {}).get("reason") or (inj or {}).get("detail"),
                         "return_date": (inj or {}).get("return_date")})
        out[team] = rows
    return out


# ------------------------------------------------------------------ live math (used by Predictor)
def team_lost_value(players, overrides=None):
    """M for one team tonight. overrides: {player_id: 'plays' | 'sits'}."""
    if not players:
        return 0.0
    ov = overrides or {}
    p = np.array([1.0 if ov.get(x["player_id"]) == "plays" else 0.0 if ov.get(x["player_id"]) == "sits"
                  else x["p_play"] for x in players])
    mwp = np.array([x["mwp"] for x in players])
    exp = p * mwp
    tot = exp.sum()
    scale = np.clip(TEAM_MIN / tot, 0.8, 1.4) if tot > 0 else 1.0
    tonight = exp * scale
    base = np.array([x["base"] for x in players])
    v = np.array([x["v"] for x in players])
    return float((v * (base - tonight)).sum())


def player_impacts(players, coefs, overrides=None):
    """Margin swing for the team between the player sitting and playing (others as given)."""
    out = []
    k = coefs["b_own"] - coefs["b_opp"]
    for x in players:
        if not x["on_roster"]:
            continue
        ov = dict(overrides or {})
        ov[x["player_id"]] = "sits"
        m_sit = team_lost_value(players, ov)
        ov[x["player_id"]] = "plays"
        m_play = team_lost_value(players, ov)
        out.append({**x, "impact": float(k * (m_sit - m_play))})   # negative = team worse when he sits
    return out
