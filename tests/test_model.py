"""Offline tests (no network). Run: pytest -q"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nba_model as nm  # noqa: E402
from synthetic import make_league  # noqa: E402


def test_train_backtest_and_predict(tmp_path):
    make_league(tmp_path)
    bt = nm.walk_forward(tmp_path, log=lambda *a: None)
    assert bt["n_games"] > 2000
    assert bt["win_brier"] < bt["const_brier"]           # the ratings carry real signal
    assert 10 < bt["sigma"]["total_reg"] < 25 and 9 < bt["sigma"]["margin_reg"] < 20
    for row in bt["calibration"]:                          # roughly calibrated where there's data
        if row["n"] > 300:
            assert abs(row["predicted"] - row["actual"]) < 0.06

    art = nm.train_artifact(tmp_path, as_of="2026-03-25", log=lambda *a: None)
    p = nm.Predictor(art)
    r = p.predict_game("BOS", "NY", total_line=224.0, home_spread=-3.0)
    assert 180 < r["exp_total"] < 270
    assert abs(r["total"]["over"] + r["total"]["under"] + r["total"]["push"] - 1) < 1e-9
    assert r["total"]["push"] > 0 and r["spread"]["push"] > 0    # whole-number lines can push
    flip = p.predict_game("NY", "BOS")
    assert (r["p_home_win"] - 0.5) * (flip["p_home_win"] - 0.5) <= 0 or abs(r["p_home_win"] - flip["p_home_win"]) < 0.2
    tr = p.team_ratings()
    assert {"ORtg", "DRtg", "Pace"} <= set(tr.columns) and len(tr) == 30


def test_game_dist_has_no_final_ties():
    gd = nm.GameDist(112, 110, {"total_reg": 18, "margin_reg": 13}, None, nm.default_ot_pmf())
    d, p = gd.margin_pmf()
    assert abs(p.sum() - 1) < 1e-9
    assert p[d == 0].sum() < 1e-12
    assert 0.02 < gd.p_ot < 0.05
    assert 0.5 < gd.p_home_win < 0.6


def test_margin_weights_raise_tie_rate():
    rng = np.random.default_rng(0)
    mu = rng.normal(0, 6, 4000)
    act = np.round(mu + rng.normal(0, 13, 4000))
    act[rng.random(4000) < 0.04] = 0                       # inject extra regulation ties
    w = nm.fit_margin_weights(mu, act, 13.0)
    assert w[0] > 1.5


def test_grading_rules():
    from nba_picks import grade_one
    g = {"home_team": "BOS", "away_team": "NY", "home_score": 110, "away_score": 104,
         "home_q1": 28, "home_q2": 27, "away_q1": 25, "away_q2": 26}
    assert grade_one({"market": "Spread", "side": "home", "line": -5.5, "price": -110}, g)[0] == "W"
    assert grade_one({"market": "Spread", "side": "away", "line": 6, "price": -110}, g)[0] == "P"
    assert grade_one({"market": "Total", "side": "over", "line": 214.5, "price": -110}, g)[0] == "L"
    assert grade_one({"market": "1H Total", "side": "over", "line": 106.5, "price": -110}, g)[0] == "L"
    assert grade_one({"market": "2H Total", "side": "under", "line": 108.5, "price": -110}, g)[0] == "W"
    assert grade_one({"market": "Team Total", "side": "over", "team": "NY", "line": 103.5, "price": -110}, g)[0] == "W"
    assert grade_one({"market": "Moneyline", "side": "away", "line": None, "price": 150}, g)[:3:2] == ("L", -1.0)
    assert grade_one({"market": "Moneyline", "side": "home", "line": None, "price": 150}, g)[2] == 1.5


def test_pricing_devig_and_kelly():
    from nba_picks import price
    mk = pd.DataFrame([{"group": "tot", "label": "Over", "p": 0.55, "push": 0.0, "price": -110},
                       {"group": "tot", "label": "Under", "p": 0.45, "push": 0.0, "price": -110}])
    out = price(mk)
    assert np.allclose(out["book"], 0.5)
    assert out.loc[0, "edge"] > 0.04 and out.loc[0, "kelly"] > 0 and out.loc[1, "kelly"] == 0


def test_espn_parsing_and_line_persistence():
    from nba_common import parse_scoreboard, upsert_games
    js = {"events": [{"id": "1", "date": "2026-10-21T23:30Z", "season": {"year": 2027, "type": 2},
                      "competitions": [{"neutralSite": False,
                                        "status": {"type": {"state": "pre", "completed": False}},
                                        "competitors": [
                                            {"homeAway": "home", "score": "0", "team": {"abbreviation": "BOS"}},
                                            {"homeAway": "away", "score": "0", "team": {"abbreviation": "NY"}}],
                                        "odds": [{"details": "NY -2.5", "overUnder": 224.5,
                                                  "homeTeamOdds": {"moneyLine": 110}, "awayTeamOdds": {"moneyLine": -130}}]}]}]}
    pre = pd.DataFrame(parse_scoreboard(js, dt.date(2026, 10, 21)))
    assert pre.loc[0, "spread_home"] == 2.5 and pre.loc[0, "ml_away"] == -130
    js["events"][0]["competitions"][0].pop("odds")      # ESPN drops the line after tip-off
    js["events"][0]["competitions"][0]["status"]["type"] = {"state": "post", "completed": True}
    post = pd.DataFrame(parse_scoreboard(js, dt.date(2026, 10, 21)))
    merged = upsert_games(pre, post)
    assert merged.loc[0, "total_line"] == 224.5 and bool(merged.loc[0, "completed"])


def test_box_score_parser():
    from nba_boxscores import parse_summary
    stat = lambda n, v: {"name": n, "displayValue": v}  # noqa: E731
    team = lambda ab, ha: {"team": {"abbreviation": ab}, "homeAway": ha, "statistics": [  # noqa: E731
        stat("fieldGoalsMade-fieldGoalsAttempted", "45-90"), stat("freeThrowsMade-freeThrowsAttempted", "12-17"),
        stat("offensiveRebounds", "8"), stat("totalTurnovers", "13")]}
    js = {"boxscore": {"teams": [team("POR", "away"), team("BOS", "home")], "players": [
        {"team": {"abbreviation": "POR"}, "statistics": [{"keys": ["minutes", "points", "fieldGoalsMade-fieldGoalsAttempted"],
         "athletes": [{"athlete": {"id": 7, "displayName": "A"}, "starter": True, "didNotPlay": False,
                       "stats": ["31", "12", "3-11"]}]}]}]}}
    t, p = parse_summary("1", js)
    assert t[0]["fga"] == 90 and t[0]["tov"] == 13 and t[1]["is_home"] is True
    assert p[0]["minutes"] == 31 and p[0]["fga"] == 11


def test_availability_recovers_absence_effects(tmp_path):
    from synthetic import EFFECT_OWN, EFFECT_OPP
    make_league(tmp_path)
    inj = {"fetched_at": "t", "rosters": {}, "players": [
        {"team": "BOS", "player_id": "BOS0", "name": "BOS Player0", "status": "Questionable"}]}
    art = nm.train_artifact(tmp_path, as_of="2026-03-25", injuries=inj, log=lambda *a: None)
    av = art["availability"]
    assert av["on"]
    assert av["b_own"] < 0 < av["b_opp"]                       # losing value hurts you and helps them
    assert abs(av["b_own"] - EFFECT_OWN) < 0.35 and abs(av["b_opp"] - EFFECT_OPP) < 0.35
    assert av["team_mae_after"] < av["team_mae_before"]

    p = nm.Predictor(art)
    plays = p.predict_game("BOS", "NY", overrides={"BOS0": "plays"})
    sits = p.predict_game("BOS", "NY", overrides={"BOS0": "sits"})
    auto = p.predict_game("BOS", "NY")
    assert plays["exp_margin"] > auto["exp_margin"] > sits["exp_margin"]
    assert plays["exp_margin"] - sits["exp_margin"] > 3            # the synthetic star matters
    named = [x["name"] for x in auto["avail"]["home"]["players"]]
    assert named == ["BOS Player0"]                                 # healthy players aren't listed

    bt = nm.walk_forward(tmp_path, log=lambda *a: None)
    wa = bt["with_availability"]
    assert wa["win_brier"] < wa["baseline_same_games"]["win_brier"]


def test_depends_on_flag(tmp_path):
    from nba_picks import select
    make_league(tmp_path)
    inj = {"fetched_at": "t", "rosters": {}, "players": [
        {"team": "BOS", "player_id": "BOS0", "name": "BOS Player0", "status": "Questionable"}]}
    up = [{"game_id": "1", "game_date": "2026-03-26", "home_team": "BOS", "away_team": "NY", "neutral": False}]
    art = nm.train_artifact(tmp_path, as_of="2026-03-25", injuries=inj, upcoming=up, log=lambda *a: None)
    p = nm.Predictor(art)
    g = dict(art["slate"][0])
    mid = p.predict_game("BOS", "NY")["exp_margin"]
    g["lines"] = {"spread_home": -round(mid * 2) / 2 - 2.5, "ml_home": None, "ml_away": None}   # away side looks good
    picks = select(p, g)
    flagged = [x for x in picks if x["market"] == "Spread"]
    assert flagged, "expected an away spread pick"     # it exists only because the star might sit
    assert flagged[0]["depends_on"] == "BOS Player0 sitting"


def test_official_report_parser():
    from nba_injuries import norm_name, parse_report_lines
    lines = ["Injury Report: 02/18/26 05:30 PM", "Game Date Game Time Matchup Team Player Name Current Status Reason",
             "02/19/2026 07:00 (ET) ATL@PHI Atlanta Hawks Johnson, Jalen Out Injury/Illness - Left Shoulder; Surgery",
             "Hunter, De'Andre Questionable Injury/Illness - Right Knee; Soreness",
             "Philadelphia 76ers Embiid, Joel Doubtful Injury/Illness - Left Knee; Injury Management",
             "Oubre Jr., Kelly Probable Injury/Illness - Right Ankle; Sprain",
             "07:30 (ET) BOS@NYK Boston Celtics NOT YET SUBMITTED", "Page 1 of 2"]
    rows = parse_report_lines(lines)
    got = [(r["team"], norm_name(r["name"]), r["official_status"]) for r in rows]
    assert got == [("ATL", "jalen johnson", "Out"), ("ATL", "deandre hunter", "Questionable"),
                   ("PHI", "joel embiid", "Doubtful"), ("PHI", "kelly oubre", "Probable")]
    assert norm_name("Nikola Jokić") == norm_name("Jokic, Nikola")


def test_preseason_and_summer_league_are_excluded(tmp_path):
    from nba_common import parse_scoreboard

    def event(gid, stype, slug):
        return {"id": gid, "date": "2026-10-10T23:30Z", "season": {"year": 2027, "type": stype, "slug": slug},
                "competitions": [{"status": {"type": {"state": "post", "completed": True}}, "competitors": [
                    {"homeAway": "home", "score": "110", "team": {"abbreviation": "BOS"}},
                    {"homeAway": "away", "score": "100", "team": {"abbreviation": "NY"}}]}]}
    js = {"events": [event("1", 1, "preseason"), event("2", 2, "regular-season"), event("3", 5, "play-in-season")]}
    kept = [r["game_id"] for r in parse_scoreboard(js, dt.date(2026, 10, 10))]
    assert kept == ["2", "3"]
    assert parse_scoreboard({"events": [event("4", 2, "regular-season")]}, dt.date(2026, 7, 12)) == []  # summer

    # rows already stored by older code are dropped on load, with their box scores and player minutes
    g, _ = make_league(tmp_path)
    junk = g.head(40).copy()
    junk["game_id"] = "pre" + junk["game_id"]
    junk.loc[:20, "game_type"] = 1                                      # preseason
    junk.loc[20:, "game_date"] = "2025-07-15"                           # summer league date
    pd.concat([g, junk]).to_parquet(tmp_path / "games.parquet", index=False)
    D = nm.load_data(tmp_path, log=lambda *a: None)
    assert not D["fin"]["game_id"].str.startswith("pre").any()
    assert len(D["fin"]) == len(g)
