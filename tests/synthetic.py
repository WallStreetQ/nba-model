"""Synthetic league in the exact ESPN schema, for offline tests (no network)."""
import datetime as dt

import numpy as np
import pandas as pd

from nba_common import NBA_TEAMS

TEAMS = sorted(NBA_TEAMS)


ROLE_MIN = np.array([35, 33, 31, 29, 27, 23, 20, 17, 13, 8, 4, 0, 0], float)   # sums to 240
ROLE_VAL = np.array([.62, .5, .42, .38, .35, .32, .3, .28, .25, .22, .2, .2, .2])  # Game Score per minute
EFFECT_OWN, EFFECT_OPP = -0.6, 0.3   # true points effect of one unit of lost value


def _rosters(rng):
    return {t: [{"player_id": f"{t}{i}", "name": f"{t} Player{i}", "v": float(ROLE_VAL[i] * rng.uniform(0.85, 1.15)),
                 "min": float(ROLE_MIN[i]), "p_sit": 0.12 if i < 2 else 0.07} for i in range(13)] for t in TEAMS}


def _minutes(roster, rng):
    """Who sits tonight, and everyone else's minutes rescaled to 240. Returns (minutes, lost value)."""
    sit = np.array([rng.random() < p["p_sit"] for p in roster])
    base = np.array([p["min"] for p in roster])
    v = np.array([p["v"] for p in roster])
    mins = np.where(sit, 0.0, base)
    mins = mins * 240 / mins.sum()
    return mins, float((v * (base - mins)).sum())


def make_league(data_dir, seasons=(2024, 2025, 2026), games_per_day=8, days=155, seed=3, with_lines=True):
    rng = np.random.default_rng(seed)
    rosters = _rosters(rng)
    games, box, players = [], [], []
    gid = 400000000
    for season in seasons:
        off = dict(zip(TEAMS, rng.normal(0, 3.5, 30)))
        de = dict(zip(TEAMS, rng.normal(0, 3.5, 30)))
        pace = dict(zip(TEAMS, rng.normal(0, 2.0, 30)))
        start = dt.date(season - 1, 10, 22)
        for d in range(days):
            day = start + dt.timedelta(days=d)
            order = rng.permutation(TEAMS)
            for k in range(0, 2 * games_per_day, 2):
                h, a = order[k], order[k + 1]
                poss = rng.normal(99 + pace[h] + pace[a], 4)
                eh = 114 + off[h] - de[a] + 1.5
                ea = 114 + off[a] - de[h] - 1.5
                (mh, lh), (ma, la) = _minutes(rosters[h], rng), _minutes(rosters[a], rng)
                hp = int(round(poss * eh / 100 + rng.normal(0, 9.5) + EFFECT_OWN * lh + EFFECT_OPP * la))
                ap = int(round(poss * ea / 100 + rng.normal(0, 9.5) + EFFECT_OWN * la + EFFECT_OPP * lh))
                for team, mins in ((h, mh), (a, ma)):
                    for pl, m in zip(rosters[team], mins):
                        players.append({"game_id": str(gid + 1), "team": team, "player_id": pl["player_id"],
                                        "name": pl["name"], "starter": False, "did_not_play": m == 0,
                                        "minutes": round(m), "pts": max(0.0, pl["v"] * m + rng.normal(0, 3)) if m else 0.0})
                hq = rng.multinomial(hp, [.25] * 4)
                aq = rng.multinomial(ap, [.25] * 4)
                lines, n_periods, hot, aot = {}, 4, 0, 0
                while hp + hot == ap + aot:
                    oh, oa = rng.poisson(10.5), rng.poisson(10.5)
                    hot, aot, n_periods = hot + oh, aot + oa, n_periods + 1
                if with_lines and season == seasons[-1]:
                    lines = {"total_line": round((eh + ea) * (99 + pace[h] + pace[a]) / 100 * 2) / 2,
                             "spread_home": -round((eh - ea) * 2) / 2, "ml_home": -150.0, "ml_away": 130.0,
                             "odds_provider": "TestBook"}
                gid += 1
                games.append({"game_id": str(gid), "game_date": day.isoformat(), "start_utc": f"{day}T23:30Z",
                              "season": season, "game_type": 2, "game_state": "post", "completed": True,
                              "status_detail": "Final", "neutral": False, "home_team": h, "away_team": a,
                              "home_score": float(hp + hot), "away_score": float(ap + aot),
                              **{f"home_q{i + 1}": float(hq[i]) for i in range(4)},
                              **{f"away_q{i + 1}": float(aq[i]) for i in range(4)},
                              "home_ot": float(hot), "away_ot": float(aot), "n_periods": n_periods,
                              "total_line": lines.get("total_line"), "spread_home": lines.get("spread_home"),
                              "ml_home": lines.get("ml_home"), "ml_away": lines.get("ml_away"),
                              "odds_provider": lines.get("odds_provider")})
                p_full = poss * (48 + 5 * (n_periods - 4)) / 48
                for team, opp, is_home in ((h, a, True), (a, h, False)):
                    tov, oreb, fta = rng.normal(14, 2), rng.normal(10, 2), rng.normal(22, 4)
                    box.append({"game_id": str(gid), "team": team, "opponent": opp, "is_home": is_home,
                                "fga": p_full - tov + oreb - 0.44 * fta, "oreb": oreb, "tov": tov, "fta": fta})
    g = pd.DataFrame(games)
    b = pd.DataFrame(box)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(players).to_parquet(data_dir / "game_players.parquet", index=False)
    data_dir.mkdir(parents=True, exist_ok=True)
    g.to_parquet(data_dir / "games.parquet", index=False)
    b.to_parquet(data_dir / "game_team_box.parquet", index=False)
    return g, b
