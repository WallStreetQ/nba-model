"""Offline tests for travel features, correlated stakes and halftime pricing (no data files needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import nba_model as nm  # noqa: E402
from nba_picks import grade_one, halftime_rows, joint_kelly, outcomes, price, size_together


# ------------------------------------------------------------------ travel
def test_schedule_features_trip_tz_altitude_and_reset():
    g = pd.DataFrame([
        {"game_id": "1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "NY"},
        {"game_id": "2", "game_date": "2026-11-02", "home_team": "DEN", "away_team": "BOS"},   # b2b, 2 tz, altitude
        {"game_id": "3", "game_date": "2026-11-04", "home_team": "UTAH", "away_team": "BOS"},  # 3 in 4, trip game 2
        {"game_id": "4", "game_date": "2026-11-11", "home_team": "LAL", "away_team": "BOS"},   # long gap: trip resets
        {"game_id": "5", "game_date": "2026-11-12", "home_team": "SAC", "away_team": "BOS", "neutral": True},
    ])
    g["neutral"] = g["neutral"].fillna(False).astype(bool)
    f = nm.schedule_features(g).set_index(["game_id", "team"])
    assert f.loc[("1", "BOS")].tolist() == [0, 0, 0, 0, 0]
    assert f.loc[("2", "BOS"), "b2b"] == 1 and f.loc[("2", "BOS"), "tz"] == 2 and f.loc[("2", "BOS"), "alt"] == 1
    assert f.loc[("2", "BOS"), "trip"] == 0                       # first road game
    assert f.loc[("2", "DEN"), "alt"] == 0                        # the home side is never "visiting at altitude"
    assert f.loc[("3", "BOS"), "three_in_four"] == 1 and f.loc[("3", "BOS"), "trip"] == 1
    assert f.loc[("3", "BOS"), "tz"] == 0                         # Denver -> Utah, same zone
    assert f.loc[("4", "BOS"), "trip"] == 0 and f.loc[("4", "BOS"), "tz"] == 0   # 7-day gap
    assert f.loc[("5", "BOS"), "tz"] == 0                         # neutral site: location unknown -> 0
    assert f.loc[("5", "SAC"), "alt"] == 0


def test_rows_and_cols_layout_back_compat():
    teams = ["A", "B"]
    old_beta = np.zeros(2 + 2 * len(teams) + len(nm.REST_COLS))
    new_beta = np.zeros(2 + 2 * len(teams) + len(nm.TRAVEL_COLS))
    assert nm._cols_for(old_beta, teams) is nm.REST_COLS
    assert nm._cols_for(new_beta, teams) is nm.TRAVEL_COLS
    games = pd.DataFrame({"home_team": ["A"], "away_team": ["B"], "home_b2b": [1], "away_b2b": [0], "away_alt": [1.0]})
    h, a = nm._rows(games, np.array([1]), nm.TRAVEL_COLS)
    assert h.loc[0, "b2b"] == 1 and h.loc[0, "opp_b2b"] == 0 and a.loc[0, "opp_b2b"] == 1
    assert a.loc[0, "alt"] == 1 and h.loc[0, "alt"] == 0 and h.loc[0, "three_in_four"] == 0   # missing -> 0


# ------------------------------------------------------------------ correlated exposure
def _R(win, dec):
    o = np.asarray(win, float)
    return np.where(o > 0, dec - 1, np.where(o < 0, -1.0, 0.0))


def test_joint_kelly_single_bet_matches_formula():
    rng = np.random.default_rng(1)
    p, dec = 0.56, 1.909
    o = np.where(rng.random(200000) < p, 1, -1)
    f = joint_kelly(_R(o[:, None], dec))[0]
    b = dec - 1
    assert f == pytest.approx((b * p - (1 - p)) / b, abs=0.006)


def test_joint_kelly_duplicates_share_one_stake_and_independent_keep_theirs():
    rng = np.random.default_rng(2)
    p, dec, n = 0.56, 1.909, 200000
    x = np.where(rng.random(n) < p, 1, -1)
    y = np.where(rng.random(n) < p, 1, -1)
    solo = joint_kelly(_R(x[:, None], dec))[0]
    dup = joint_kelly(_R(np.column_stack([x, x]), dec))
    assert dup.sum() == pytest.approx(solo, abs=0.006)              # same bet twice = one bet's stake
    ind = joint_kelly(_R(np.column_stack([x, y]), dec))
    assert ind == pytest.approx([solo, solo], abs=0.01)              # unrelated bets barely change


def test_joint_kelly_never_upsizes_a_hedge():
    rng = np.random.default_rng(3)
    n = 100000
    x = np.where(rng.random(n) < 0.6, 1, -1)
    y = -x * np.where(rng.random(n) < 0.9, 1, -1)                    # mostly the opposite outcome
    R = _R(np.column_stack([x, y]), 1.95)
    solo = np.array([joint_kelly(R[:, :1])[0], max(joint_kelly(R[:, 1:])[0], 0.0)])
    f = joint_kelly(R, solo, upper=solo)
    assert (f <= solo + 1e-9).all()


def test_outcomes_grades_every_market_type():
    sims = {"home": np.array([110., 100.]), "away": np.array([100., 104.]), "h1": np.array([105., 98.])}
    rows = [{"type": "Spread", "side": "home", "team": "H", "line": -9.5},
            {"type": "Moneyline", "side": "away", "team": "A", "line": None},
            {"type": "Total", "side": "over", "team": None, "line": 205.0},
            {"type": "Team Total", "side": "under", "team": "A", "line": 101.5},
            {"type": "1H Total", "side": "over", "team": None, "line": 100.5},
            {"type": "2H Total", "side": "under", "team": None, "line": 106.0}]
    O = outcomes(rows, sims, "H")
    assert O[:, 0].tolist() == [1, -1]          # 10 - 9.5 > 0 ; -4 - 9.5 < 0
    assert O[:, 1].tolist() == [-1, 1]
    assert O[:, 2].tolist() == [1, -1]          # 210 vs 205 ; 204 vs 205
    assert O[:, 3].tolist() == [1, -1]          # 100 < 101.5 ; 104 > 101.5
    assert O[:, 4].tolist() == [1, -1]
    assert O[:, 5].tolist() == [1, 0]           # 2H = 105 ; 106 -> push


def test_size_together_caps_the_game_and_names_the_partner():
    rng = np.random.default_rng(4)
    tot = rng.normal(225, 18, 20000).round()
    sims = {"home": (tot / 2 + rng.normal(3, 6, 20000)).round(), "away": None, "h1": (tot / 2).round()}
    sims["away"] = tot - sims["home"]
    rows = pd.DataFrame([
        {"type": "Total", "side": "under", "team": None, "line": 235.5, "label": "Under 235.5", "dec": 1.909, "kelly": 0.06},
        {"type": "Team Total", "side": "under", "team": "H", "line": 120.5, "label": "H TT Under 120.5", "dec": 1.909,
         "kelly": 0.05}])
    stakes, notes = size_together(rows, sims, "H", cap=0.05)
    assert stakes.sum() <= 0.05 + 1e-9 and (stakes <= rows["kelly"].values + 1e-9).all()
    assert "moves with" in notes[0] and "H TT Under" in notes[0]


# ------------------------------------------------------------------ halftime
def _halftime_frame(n, b_tot, b_abs, c_surp, c_lead, seed=5):
    rng = np.random.default_rng(seed)
    mu_h, mu_a = rng.normal(112, 4, n), rng.normal(110, 4, n)
    h1_h, h1_a = mu_h * 0.505, mu_a * 0.505
    home_h1 = (h1_h + rng.normal(0, 7, n)).round()
    away_h1 = (h1_a + rng.normal(0, 7, n)).round()
    mu_h1 = h1_h + h1_a
    act_h1 = home_h1 + away_h1
    h1m = home_h1 - away_h1
    mu_2h = mu_h + mu_a - mu_h1
    act_2h = mu_2h + b_tot * (act_h1 - mu_h1) + b_abs * np.abs(h1m) + rng.normal(0, 9, n)
    mu_2hm = (mu_h - h1_h) - (mu_a - h1_a)
    act_2hm = mu_2hm + c_surp * (h1m - (h1_h - h1_a)) + c_lead * h1m + rng.normal(0, 8, n)
    home_score = home_h1 + (act_2h + act_2hm) / 2
    away_score = away_h1 + (act_2h - act_2hm) / 2
    return pd.DataFrame({"mu_h": mu_h, "mu_a": mu_a, "h1_h": h1_h, "h1_a": h1_a, "mu_h1": mu_h1, "act_h1": act_h1,
                         "home_h1": home_h1, "away_h1": away_h1, "mu_2h": mu_2h, "act_2h": act_2h,
                         "home_score": home_score, "away_score": away_score, "season": np.repeat([1, 2], n // 2)})


def test_fit_halftime_recovers_known_relationship():
    r = _halftime_frame(6000, b_tot=0.30, b_abs=-0.12, c_surp=0.10, c_lead=-0.15)
    ht = nm.fit_halftime(r, {"h1_total": 10, "h2_total": 11, "total_reg": 17, "margin_reg": 13})
    assert ht["fitted"]
    assert ht["total"]["coef"][1] == pytest.approx(0.30, abs=0.04)
    assert ht["total"]["coef"][2] == pytest.approx(-0.12, abs=0.04)
    assert ht["margin"]["coef"][2] == pytest.approx(-0.15, abs=0.05)
    bt = nm.halftime_backtest(r, [1, 2])
    assert bt["h2_mae_halftime"] < bt["h2_mae_pregame"]


def test_fit_halftime_fallback_and_projection():
    sig = {"h1_total": 10.0, "h2_total": 11.0, "total_reg": 17.0, "margin_reg": 13.0}
    ht = nm.fit_halftime(None, sig)
    assert not ht["fitted"] and 0 < ht["total"]["coef"][1] <= 0.5
    pre = {"mu_h": 112.0, "mu_a": 108.0, "h1_h": 56.5, "h1_a": 54.5}
    even = nm.halftime_projection(ht, pre, 56.5, 54.5, 0.06, 21.0)
    hot = nm.halftime_projection(ht, pre, 66.5, 64.5, 0.06, 21.0)
    assert even["mu_2h"] == pytest.approx(even["pre_2h"])          # on-script first half: no change
    assert hot["mu_2h"] > even["mu_2h"]                             # fast first half carries over


def test_discrete_spread_and_halftime_rows_price():
    s = nm.discrete_normal_spread(0.0, 8.0, 0.0)
    assert s["cover"] == pytest.approx(s["fail"], abs=1e-9) and s["push"] > 0
    ht = {"total": nm.discrete_normal_ou(110, 9, 104.5), "spread": nm.discrete_normal_spread(3, 8, -1.5)}
    mk = price(halftime_rows(ht, "H", "A", {"h2_total_line": 104.5, "h2_spread_home": -1.5}))
    assert set(mk["type"]) == {"Live 2H Total", "Live 2H Spread"}
    best = mk.sort_values("edge").iloc[-1]
    assert best["label"] == "2H Over 104.5" and best["edge"] > 0.2


def test_grade_live_markets():
    g = {"home_score": 112.0, "away_score": 104.0, "home_q1": 30, "home_q2": 25, "away_q1": 28, "away_q2": 30,
         "home_team": "H", "away_team": "A"}
    # 2H: home 57, away 46 -> total 103, home margin +11
    assert grade_one({"market": "Live 2H Total", "side": "under", "line": 105.5, "price": -110, "team": None}, g)[:2] == ("W", 103.0)
    assert grade_one({"market": "Live 2H Spread", "side": "home", "line": -10.5, "price": -110, "team": "H"}, g)[0] == "W"
    assert grade_one({"market": "Live 2H Spread", "side": "away", "line": 10.5, "price": -110, "team": "A"}, g)[0] == "L"
