"""
streamlit_app.py - NBA game predictor

Reads only files in model/ (written by refresh.py and nba_live.py): model_artifact.json,
prediction_log.csv, picks.csv, live.json, halftime_log.csv, live_picks.csv.
No raw data, no API calls.  Run locally:  streamlit run streamlit_app.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from nba_availability import player_impacts
from nba_model import TRAVEL_SHOW, Predictor, load_artifact
from nba_picks import MAX_GAME_EXPOSURE, game_lines_args, halftime_rows, market_rows, price, size_together

ROOT = Path(__file__).parent
ART_PATH = ROOT / "model" / "model_artifact.json"
LOG_PATH = ROOT / "model" / "prediction_log.csv"
PICKS_PATH = ROOT / "model" / "picks.csv"
LIVE_PATH = ROOT / "model" / "live.json"
HT_LOG_PATH = ROOT / "model" / "halftime_log.csv"
LIVE_PICKS_PATH = ROOT / "model" / "live_picks.csv"
ET = ZoneInfo("America/New_York")
CUSTOM = "Custom matchup"
EARLY_GP = 10
MARKETS = ["Spread", "Moneyline", "Total", "Team Total", "1H Total", "2H Total", "Live 2H Total", "Live 2H Spread"]

st.set_page_config(page_title="NBA Game Predictor", page_icon="🏀", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 1.4rem; padding-bottom: 3rem;}
@media (max-width: 640px) { .block-container {padding-left: .75rem; padding-right: .75rem;} }
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 13px 15px; margin-bottom: 12px;
       background: rgba(128,128,128,.06);}
.head {display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;}
.matchup {font-weight:700; font-size:1.05rem;}
.muted {opacity:.68; font-size:.82rem;}
.chip {display:inline-block; padding:2px 9px; border-radius:999px; font-size:.74rem; font-weight:600;
       border:1px solid rgba(128,128,128,.35); white-space:nowrap;}
.chip.mk {background:rgba(249,115,22,.14); color:#f97316; border-color:rgba(249,115,22,.4);}
.chip.W {background:#16a34a; color:#fff; border-color:#16a34a;}
.chip.L {background:#dc2626; color:#fff; border-color:#dc2626;}
.chip.P {background:#6b7280; color:#fff; border-color:#6b7280;}
.chip.b2b {background:rgba(234,179,8,.15); color:#ca8a04; border-color:rgba(234,179,8,.45);}
.score {font-size:1.3rem; font-weight:700; margin:6px 0 2px;}
.sel {font-size:1.22rem; font-weight:800; margin:6px 0 2px;}
.grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(118px, 1fr)); gap:8px; margin-top:10px;}
.tile {border-radius:10px; padding:7px 10px; background:rgba(128,128,128,.08);}
.tile .l {font-size:.7rem; opacity:.65; text-transform:uppercase; letter-spacing:.03em;}
.tile .v {font-size:1.02rem; font-weight:700;}
.tile .s {font-size:.74rem; opacity:.8;}
.lean {color:#f97316; font-weight:600;}
.stars {color:#f59e0b; letter-spacing:1px;}
.chip.warn {background:rgba(234,179,8,.15); color:#ca8a04; border-color:rgba(234,179,8,.45);}
.inj {margin-top:8px; font-size:.82rem; line-height:1.45;}
.inj b.neg {color:#ef4444;} .inj b.pos {color:#22c55e;}
.chip.corr {background:rgba(99,102,241,.12); color:#6366f1; border-color:rgba(99,102,241,.4);}
.chip.live {background:rgba(239,68,68,.12); color:#ef4444; border-color:rgba(239,68,68,.4);}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner=False)
def get_predictor(mtime: float) -> Predictor:      # mtime in the key => reload if the file changes
    return Predictor(load_artifact(ART_PATH))


@st.cache_data(show_spinner=False)
def get_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"game_id": str, "pick_id": str})


@st.cache_data(show_spinner=False)
def slate_predictions(mtime: float) -> list[dict]:
    pred = get_predictor(mtime)
    out = []
    for g in pred.slate:
        try:
            r = pred.predict_game(g["home"], g["away"], g["home_b2b"], g["away_b2b"], g.get("neutral", False),
                                  **game_lines_args(g.get("lines") or {}), travel=g.get("travel"))
        except ValueError:
            continue
        out.append({"g": g, "r": r})
    return out


@st.cache_data(show_spinner=False)
def get_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def csv_or_empty(path):
    return get_csv(str(path), path.stat().st_mtime) if path.exists() else pd.DataFrame()


# ------------------------------------------------------------------ formatting
def pct(p):
    return "-" if p is None or pd.isna(p) else f"{100 * p:.1f}%"


def american(p):
    if p is None or pd.isna(p) or p <= 0.0005 or p >= 0.9995:
        return "-"
    us = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    return f"{us:+.0f}"


def fair(p):
    return "-" if p <= 0.0005 or p >= 0.9995 else f"{1 / p:.2f} ({american(p)})"


def fmt(x, nd=1, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def price_txt(x):
    return "—" if x is None or pd.isna(x) else f"{int(x):+d}"


def tip(start_utc):
    try:
        return datetime.fromisoformat(str(start_utc).replace("Z", "+00:00")).astimezone(ET).strftime("%a %-I:%M %p")
    except ValueError:
        return ""


def tile(label, value, sub=""):
    return f'<div class="tile"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


STATUS_SHORT = {"Out": "out", "Doubtful": "doubtful", "Questionable": "Q", "Probable": "probable",
                "Day-To-Day": "day-to-day", "Suspension": "suspended"}


def injury_line(r, home, away):
    """'Injury adj: BOS -4.2 · Tatum out · Holiday Q (55%)' - only what moves the line."""
    parts = []
    for side, team in (("away", away), ("home", home)):
        av = (r.get("avail") or {}).get(side) or {}
        names = []
        for pl in av.get("players", [])[:3]:
            st_ = STATUS_SHORT.get(pl["status"], pl["status"])
            prob = f" ({pl['p_play']:.0%} plays)" if 0.05 < pl["p_play"] < 0.95 else ""
            names.append(f"{pl['name']} {st_}{prob}")
        if abs(av.get("adj", 0)) >= 0.5 or names:
            cls = "neg" if av.get("adj", 0) < 0 else "pos"
            parts.append(f'<b class="{cls}">{team} {av.get("adj", 0):+.1f}</b>' + (" · " + " · ".join(names) if names else ""))
    return '<div class="inj">🩹 Injury adj: ' + " &nbsp;|&nbsp; ".join(parts) + "</div>" if parts else ""


def travel_line(r, home, away):
    """'✈ Travel: LAL −1.2 (3 in 4 nights · 2 time zones)' - only teams whose travel moves the margin ≥ 0.5."""
    parts = []
    for side, team in (("away", away), ("home", home)):
        t = (r.get("travel") or {}).get(side) or {}
        if abs(t.get("adj", 0)) >= TRAVEL_SHOW:
            cls = "neg" if t["adj"] < 0 else "pos"
            parts.append(f'<b class="{cls}">{team} {t["adj"]:+.1f}</b>' + (f' ({" · ".join(t["notes"])})' if t.get("notes") else ""))
    return '<div class="inj">✈ Travel: ' + " &nbsp;|&nbsp; ".join(parts) + "</div>" if parts else ""


def lean(proj, line, over="Over", under="Under"):
    if line is None or pd.isna(line):
        return ""
    d = proj - line
    return f'<span class="lean">{over if d > 0 else under} {abs(d):.1f}</span>'


# ------------------------------------------------------------------ load
if not ART_PATH.exists():
    st.error("model/model_artifact.json not found. Run `python refresh.py` locally, then commit and push "
             "(or `python refresh.py --push`).")
    st.stop()
mtime = ART_PATH.stat().st_mtime
try:
    pred = get_predictor(mtime)
except ValueError as e:
    st.error(str(e))
    st.stop()
meta = pred.meta
gp = meta.get("season_gp", {})
slate = slate_predictions(mtime)
picks = pd.concat([csv_or_empty(PICKS_PATH), csv_or_empty(LIVE_PICKS_PATH)], ignore_index=True)
plog = csv_or_empty(LOG_PATH)
htlog = csv_or_empty(HT_LOG_PATH)
live = get_json(str(LIVE_PATH), LIVE_PATH.stat().st_mtime) if LIVE_PATH.exists() else {"games": {}}

st.title("🏀 NBA Game Predictor")
odds_note = f" · lines {meta['odds_fetched_at'][5:16].replace('T', ' ')}" if meta.get("odds_fetched_at") else ""
av_meta = pred.av
inj_note = (f" · injuries **{av_meta['injuries_fetched_at'][5:16].replace('T', ' ')}**"
            if av_meta.get("on") and av_meta.get("injuries_fetched_at") else
            (" · availability **OFF**" if not av_meta.get("on") else ""))
st.caption(f"Model trained through **{meta['trained_through']}** · {meta['n_games']:,} games · "
           f"box scores **{'ON' if meta['box_on'] else 'OFF'}**{inj_note}{odds_note}")

tab_sel, tab_slate, tab_game, tab_half, tab_track, tab_model = st.tabs(
    ["🎯 Selections", "📅 Today's Slate", "🧮 Game Markets", "⏱ Halftime", "📈 Track Record", "📊 Model"])

# ================================================================== Selections
with tab_sel:
    view = st.segmented_control("View", ["Open", "Recent results"], default="Open", label_visibility="collapsed")
    c1, c2 = st.columns([3, 2])
    chosen = c1.pills("Markets", MARKETS, selection_mode="multi", default=MARKETS, label_visibility="collapsed")
    min_stars = c2.select_slider("Minimum confidence", [1, 2, 3], 1, format_func=lambda s: "★" * s)
    if picks.empty:
        st.info("No selections yet. They're written by refresh.py when the slate has games with posted lines.")
    else:
        if view == "Open":
            pk = picks[picks["status"] == "pending"]
        else:
            pk = picks[picks["status"].isin(["W", "L", "P"])]
            pk = pk[pd.to_datetime(pk["date"]) >= pd.to_datetime(pk["date"]).max() - pd.Timedelta(days=6)]
        pk = pk[pk["market"].isin(chosen or MARKETS) & (pk["stars"] >= min_stars)]
        if pk.empty:
            st.info("Nothing clears the edge threshold here." if view == "Open" else "No graded selections in the last week.")
        else:
            pk = pk.sort_values(["date", "stars", "edge"], ascending=[view == "Open", False, False])
            st.caption(f"{len(pk)} selection(s) · edge = model win chance (pushes excluded) minus the no-vig book "
                       f"chance · stake is ¼ Kelly, with picks in the same game sized together (they share risk)")
            cols = st.columns(2)
            for i, (_, p) in enumerate(pk.iterrows()):
                is_live = str(p["market"]).startswith("Live")
                chip = (f'<span class="chip {p["status"]}">{p["status"]} · {p["profit"]:+.2f}u</span>'
                        if p["status"] in ("W", "L", "P") else
                        ('<span class="chip live">⏱ at halftime</span>' if is_live
                         else f'<span class="chip">{tip(p["start_utc"])}</span>'))
                if p["market"] == "Moneyline":
                    proj = tile("Win prob", pct(p["model_proj"]), f"Fair {american(p['model_proj'])}")
                elif p["market"] == "Spread":
                    proj = tile("Proj margin", fmt(p["model_proj"], 1, True), f"Line {p['line']:+g}")
                else:
                    proj = tile("Projection", fmt(p["model_proj"]), f"Line {p['line']:g}")
                push = f" · push {pct(p['push_prob'])}" if p.get("push_prob", 0) > 0.001 else ""
                dep = p.get("depends_on")
                dep = (f'<div style="margin-top:6px"><span class="chip warn">⚠ depends on {dep}</span></div>'
                       if isinstance(dep, str) and dep else "")
                cw = p.get("corr_with")
                if isinstance(cw, str) and cw:
                    dep += f'<div style="margin-top:6px"><span class="chip corr">📎 {cw}</span></div>'
                solo = p.get("kelly_solo")
                k_sub = "of bankroll"
                if isinstance(solo, (int, float)) and not pd.isna(solo) and abs(solo - p["kelly"]) > 0.0005:
                    n_other = int(p.get("n_same_game", 1) or 1) - 1
                    k_sub = (f"solo {100 * solo:.1f}% · sized with {n_other} other pick{'s' if n_other != 1 else ''}"
                             if n_other > 0 else f"solo {100 * solo:.1f}% · game cap")
                cols[i % 2].markdown(f"""
                <div class="card">
                  <div class="head"><span class="chip mk">{p['market']}</span>{chip}</div>
                  <div class="sel">{p['selection']} <span class="muted">({price_txt(p['price'])})</span></div>
                  <div class="muted">{p['matchup']} · <span class="stars">{'★' * int(p['stars'])}</span>{push}</div>
                  <div class="grid">
                    {proj}
                    {tile("Model / book", f"{pct(p['model_prob'])}", f"book {pct(p['book_prob'])}")}
                    {tile("Edge", f"{100 * p['edge']:+.1f} pts", f"EV {100 * p['ev']:+.1f}%")}
                    {tile("¼ Kelly", f"{100 * p['kelly']:.1f}%", k_sub)}
                  </div>{dep}
                </div>""", unsafe_allow_html=True)

# ================================================================== Slate
with tab_slate:
    if not slate:
        st.info("No upcoming games in this model file. `refresh.py` pulls today and the next 3 days.")
    else:
        dates = sorted({s["g"]["date"] for s in slate})
        day = st.segmented_control("Day", dates, default=dates[0], label_visibility="collapsed",
                                   format_func=lambda d: pd.Timestamp(d).strftime("%a %b %-d"))
        rows = [s for s in slate if s["g"]["date"] == (day or dates[0])]
        st.caption(f"{len(rows)} games · projections are final scores including overtime · lines: "
                   f"{rows[0]['g']['lines'].get('book') or 'ESPN'}" if rows else "")
        cols = st.columns(2)
        for i, s in enumerate(rows):
            g, r, L = s["g"], s["r"], s["g"].get("lines") or {}
            h, a = g["home"], g["away"]
            fav, pf = (h, r["p_home_win"]) if r["p_home_win"] >= 0.5 else (a, 1 - r["p_home_win"])
            b2b = "".join(f' <span class="chip b2b">{t} B2B</span>' for t, on in ((a, g["away_b2b"]), (h, g["home_b2b"])) if on)
            early = [t for t in (h, a) if gp.get(t, 99) < EARLY_GP]
            sh = L.get("spread_home")
            cols[i % 2].markdown(f"""
            <div class="card">
              <div class="head"><span class="matchup">{a} @ {h}{b2b}</span><span class="chip">{tip(g.get('start_utc'))}</span></div>
              <div class="score">{a} {r['exp_away']:.1f} – {r['exp_home']:.1f} {h}</div>
              <div class="muted">{'Pace ' + fmt(r['pace']) + ' · ' if r['pace'] else ''}OT chance {pct(r['p_ot'])}
                {(' · early season: ' + ', '.join(early)) if early else ''}</div>
              {injury_line(r, h, a)}{travel_line(r, h, a)}
              <div class="grid">
                {tile("Win prob", f"{fav} {pf:.0%}", f"Fair {american(pf)} · ML {price_txt(L.get('ml_away'))}/{price_txt(L.get('ml_home'))}")}
                {tile("Spread", f"{h} {-r['exp_margin']:+.1f}", (f"Line {sh:+g} · {h} covers {pct(r['spread']['cover'])}" if sh is not None else "no line"))}
                {tile("Total", fmt(r['exp_total']), (f"Line {L['total_line']:g} {lean(r['exp_total'], L['total_line'])} · O {pct(r['total']['over'])}" if L.get('total_line') is not None else "no line"))}
                {tile(f"{a} total", fmt(r['exp_away']), f"Line {fmt(L.get('away_tt_line'))} {lean(r['exp_away'], L.get('away_tt_line'))}")}
                {tile(f"{h} total", fmt(r['exp_home']), f"Line {fmt(L.get('home_tt_line'))} {lean(r['exp_home'], L.get('home_tt_line'))}")}
                {tile("1H total", fmt(r['exp_h1']), f"Line {fmt(L.get('h1_total_line'))} {lean(r['exp_h1'], L.get('h1_total_line'))}")}
                {tile("2H total", fmt(r['exp_h2']), f"Line {fmt(L.get('h2_total_line'))} {lean(r['exp_h2'], L.get('h2_total_line'))}")}
              </div>
            </div>""", unsafe_allow_html=True)
        st.caption("Team totals and 1H/2H lines appear when The Odds API key is set with --period-odds. "
                   "Use Game Markets to price any game at your own book's numbers.")

# ================================================================== Game markets (NHL-style calculator)
with tab_game:
    teams = pred.active_teams
    labels = {CUSTOM: None}
    for s in slate:
        g = s["g"]
        tag = " · ".join(f"{t} on B2B" for t, on in ((g["home"], g["home_b2b"]), (g["away"], g["away_b2b"])) if on)
        labels[f"{g['date'][5:]}  {g['away']} @ {g['home']}" + (f"  ({tag})" if tag else "")] = g
    pick = st.selectbox("Game", list(labels), index=1 if len(labels) > 1 else 0)
    sg = labels[pick]
    k = str(sg["game_id"]) if sg else "custom"
    L = (sg or {}).get("lines") or {}

    c1, c2, c3, c4 = st.columns(4)
    def_away = sg["away"] if sg else ("BOS" if "BOS" in teams else teams[0])
    def_home = sg["home"] if sg else ("NY" if "NY" in teams else teams[1])
    away = c1.selectbox("Away team", teams, index=teams.index(def_away) if def_away in teams else 0, key=f"away_{k}")
    home = c2.selectbox("Home team", teams, index=teams.index(def_home) if def_home in teams else 1, key=f"home_{k}")
    away_b2b = c3.checkbox(f"{away} played yesterday", value=bool(sg and sg["away_b2b"]), key=f"ab_{k}_{away}")
    home_b2b = c4.checkbox(f"{home} played yesterday", value=bool(sg and sg["home_b2b"]), key=f"hb_{k}_{home}")
    if home == away:
        st.warning("Pick two different teams.")
        st.stop()
    early = [t for t in (home, away) if gp.get(t, 99) < EARLY_GP]
    if early:
        st.info(f"Early season: {', '.join(f'{t} ({gp.get(t, 0)} GP)' for t in early)}. Ratings still lean on last "
                "season, pulled toward league average. Roster changes since then aren't reflected yet.")

    # rest/travel from the real schedule when this is the slate game (a custom matchup has none)
    travel = sg.get("travel") if sg and sg["home"] == home and sg["away"] == away else None

    # ---- availability: automatic, with a switch only for players who move the line ----
    overrides = {}
    if pred.av.get("on"):
        auto = pred.predict_game(home, away, home_b2b, away_b2b, travel=travel)
        listed = [(t, pl) for side, t in (("away", away), ("home", home))
                  for pl in (auto["avail"][side] or {}).get("players", [])]
        adj_txt = " · ".join(f"{t} {(auto['avail'][s_] or {}).get('adj', 0):+.1f}" for s_, t in (("away", away), ("home", home)))
        with st.expander(f"🩹 Availability · {adj_txt}" + (f" · {len(listed)} player(s) that matter" if listed else ""),
                         expanded=bool(listed)):
            if not listed:
                st.caption("No listed injuries that move this game by half a point or more. Adjustments shown come "
                           "from recent role and minutes changes.")
            for t, pl in listed:
                fixed = pl["p_play"] in (0.0, 1.0)
                label = (f"**{pl['name']}** ({t}) · {pl['status']}"
                         + (f" · plays {pl['p_play']:.0%}" if not fixed else "")
                         + f" · worth {abs(pl['impact']):.1f} pts to {t}")
                choice = st.segmented_control(label, ["Auto", "Plays", "Sits"], default="Auto",
                                              key=f"ov_{k}_{pl['player_id']}")
                if choice in ("Plays", "Sits"):
                    overrides[pl["player_id"]] = choice.lower()
            st.caption("Auto uses tonight's injury report: listed players count by how often players with that "
                       "status actually play, and their minutes go to teammates. Override when news breaks.")

    base = pred.predict_game(home, away, home_b2b, away_b2b, overrides=overrides, travel=travel)
    rnd = lambda x: round(x * 2) / 2  # noqa: E731

    def dv(key, fallback):
        v = L.get(key)
        return float(v) if v is not None and sg and sg["home"] == home and sg["away"] == away else float(fallback)

    with st.expander("Lines", expanded=True):
        l1, l2, l3 = st.columns(3)
        total_line = l1.number_input("Game total", 150.0, 290.0, dv("total_line", rnd(base["exp_total"])), 0.5, key=f"tl_{k}")
        home_spread = l2.number_input(f"{home} spread", -30.0, 30.0, dv("spread_home", rnd(-base["exp_margin"])), 0.5,
                                      key=f"sp_{k}", help="-5.5 = home lays 5.5")
        h1_line = l3.number_input("1H total", 80.0, 150.0, dv("h1_total_line", rnd(base["exp_h1"])), 0.5, key=f"h1_{k}")
        l4, l5, l6 = st.columns(3)
        home_tt = l4.number_input(f"{home} team total", 70.0, 160.0, dv("home_tt_line", rnd(base["exp_home"])), 0.5, key=f"htt_{k}")
        away_tt = l5.number_input(f"{away} team total", 70.0, 160.0, dv("away_tt_line", rnd(base["exp_away"])), 0.5, key=f"att_{k}")
        h2_line = l6.number_input("2H total", 80.0, 150.0, dv("h2_total_line", rnd(base["exp_h2"])), 0.5, key=f"h2_{k}")

    r = pred.predict_game(home, away, home_b2b, away_b2b, False, total_line, home_spread, home_tt, away_tt, h1_line, h2_line,
                          overrides=overrides, travel=travel)
    tl_txt = travel_line(r, home, away)
    if tl_txt:
        st.markdown(tl_txt, unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{away} @ {home}", f"{r['exp_away']:.1f} – {r['exp_home']:.1f}", help="Expected final score incl. OT")
    m2.metric("Expected total", f"{r['exp_total']:.1f}")
    m3.metric(f"{home} win probability", pct(r["p_home_win"]), help="Includes overtime")
    m4.metric("Goes to OT", pct(r["p_ot"]))

    lines_now = {"spread_home": home_spread, "total_line": total_line, "home_tt_line": home_tt, "away_tt_line": away_tt,
                 "h1_total_line": h1_line, "h2_total_line": h2_line, "ml_home": -110, "ml_away": -110}
    mk = market_rows(r, lines_now)
    for s in r["alt_spreads"].items():
        line, alt = s
        mk = pd.concat([mk, pd.DataFrame([
            {"group": f"alt{line}", "type": "Spread", "side": "home", "team": home, "label": f"{home} {line:+g} (alt)",
             "line": line, "p": alt["cover"], "push": alt["push"], "price": None},
            {"group": f"alt{line}", "type": "Spread", "side": "away", "team": away, "label": f"{away} {-line:+g} (alt)",
             "line": -line, "p": alt["fail"], "push": alt["push"], "price": None}])], ignore_index=True)
    # pre-fill "Your odds" with the posted prices for this game, when we have them
    same = sg and sg["home"] == home and sg["away"] == away
    posted = {("ml", "home"): L.get("ml_home"), ("ml", "away"): L.get("ml_away"),
              ("spread", "home"): L.get("spread_home_price"), ("spread", "away"): L.get("spread_away_price"),
              ("tot", "over"): L.get("total_over_price"), ("tot", "under"): L.get("total_under_price"),
              ("htt", "over"): L.get("home_tt_over_price"), ("htt", "under"): L.get("home_tt_under_price"),
              ("att", "over"): L.get("away_tt_over_price"), ("att", "under"): L.get("away_tt_under_price"),
              ("h1", "over"): L.get("h1_total_over_price"), ("h1", "under"): L.get("h1_total_under_price"),
              ("h2", "over"): L.get("h2_total_over_price"), ("h2", "under"): L.get("h2_total_under_price")}
    prefill = [posted.get((g_, s_)) if same else None for g_, s_ in zip(mk["group"], mk["side"])]

    st.caption("Type your sportsbook's American odds (e.g. -115, +140) in **Your odds**. Price both sides of a market "
               "and the book's margin is removed before comparing. On whole-number lines the push chance is shown "
               "separately and fair odds exclude it, matching how books refund a push.")
    shown = pd.DataFrame({"Market": mk["label"], "Model": mk["p"].map(pct),
                          "Push": mk["push"].map(lambda x: pct(x) if x > 1e-9 else ""),
                          "Fair odds": (mk["p"] / (1 - mk["push"])).map(lambda p: fair(p) if p > 0 else "-"),
                          "Your odds": prefill})
    edit_key = (f"odds_{home}_{away}_{total_line}_{home_spread}_{home_tt}_{away_tt}_{h1_line}_{h2_line}_"
                f"{sorted(overrides.items())}")
    edited = st.data_editor(shown, hide_index=True, width="stretch", key=edit_key,
                            disabled=["Market", "Model", "Push", "Fair odds"],
                            column_config={"Your odds": st.column_config.NumberColumn(
                                "Your odds", help="American odds, e.g. -110 or +125", step=1, format="%+d")})
    priced = price(mk, edited["Your odds"].values)
    priced = priced[priced["dec"].notna()]
    if len(priced):
        # bets with a positive stake are sized together: they settle on the same game
        bets = priced[priced["kelly"] > 0]
        joint = pd.Series(0.0, index=priced.index)
        if len(bets) > 1:
            sims = pred.sample_scores(home, away, home_b2b, away_b2b, n=20000, seed=0, overrides=overrides, travel=travel)
            joint[bets.index] = size_together(bets, sims, home)[0]
        else:
            joint[bets.index] = np.minimum(bets["kelly"], MAX_GAME_EXPOSURE)
        st.markdown("**Value check**")
        st.dataframe(pd.DataFrame({
            "Market": priced["label"], "Book (no-vig)": priced["book"].map(pct), "Model": priced["p_cond"].map(pct),
            "Edge": priced["edge"].map(lambda x: f"{100 * x:+.1f} pts"),
            "EV per $100": priced["ev"].map(lambda x: f"{100 * x:+.2f}"),
            "¼ Kelly (alone)": priced["kelly"].map(lambda x: f"{100 * x:.1f}%" if x > 0 else "no bet"),
            "¼ Kelly (together)": joint.map(lambda x: f"{100 * x:.1f}%" if x > 0 else "—")}),
            hide_index=True, width="stretch")
        st.caption(f"'Alone' sizes each bet as if it were the only one. 'Together' sizes every positive-EV bet in this "
                   f"table as one package, since they all settle on this game: an Over plus a team-total Over share most "
                   f"of their risk, so the package stakes less than the sum. A bet is never sized up for hedging another, "
                   f"and the package is capped at {MAX_GAME_EXPOSURE:.0%} of bankroll. Small edges (under ~3 points) "
                   f"are within this model's error. Stakes assume the model is right; ¼ Kelly is already cautious.")
    mp = pd.DataFrame(r["margin_pmf"])
    st.markdown(f"**Final margin** ({home} minus {away})")
    st.bar_chart(mp.set_index("margin")["p"], x_label=f"{home} margin", y_label="Probability", height=230)

# ================================================================== Halftime
def pre_for(game_id, home, away):
    """Pre-tip means for a game: saved by the last refresh before tip-off, else from the model now."""
    if game_id and not plog.empty and "reg_home" in plog:
        row = plog[plog["game_id"] == str(game_id)]
        need = ["reg_home", "reg_away", "exp_h1_home", "exp_h1_away"]
        if not row.empty and row[need].notna().all(axis=1).iloc[-1]:
            r_ = row.iloc[-1]
            return {"mu_h": r_["reg_home"], "mu_a": r_["reg_away"], "h1_h": r_["exp_h1_home"], "h1_a": r_["exp_h1_away"]}
    sg_ = next((s["g"] for s in slate if s["g"]["game_id"] == str(game_id)), None)
    if sg_ and sg_["home"] == home and sg_["away"] == away:
        return pred.pregame(home, away, sg_["home_b2b"], sg_["away_b2b"], sg_.get("neutral", False), travel=sg_.get("travel"))
    return pred.pregame(home, away)


with tab_half:
    priced_tonight = sorted((live.get("games") or {}).values(), key=lambda v: (v.get("date", ""), v.get("priced_at", "")),
                            reverse=True)
    if priced_tonight:
        st.caption("Each game is priced the moment it reaches halftime, from its first-half score and the pre-game "
                   "projection. Book lines and edges appear when the job buys live 2H lines (The Odds API).")
        cols = st.columns(2)
        for i, v in enumerate(priced_tonight):
            h, a = v["home"], v["away"]
            fin_ = v.get("final")
            chip = (f'<span class="chip">Final {a} {fin_["away"]:.0f}–{fin_["home"]:.0f} {h}</span>' if fin_ else
                    ('<span class="chip warn">priced after the half</span>' if v.get("late") else
                     '<span class="chip live">⏱ at halftime</span>'))
            res = (f" · actual 2H {v['act_2h']:.0f}" if v.get("act_2h") is not None and not pd.isna(v.get("act_2h")) else "")
            bl = ""
            if v.get("markets"):
                best_m = max(v["markets"], key=lambda m: m["edge"])
                bl = tile("Best vs book", best_m["label"], f"edge {100 * best_m['edge']:+.1f} pts · {price_txt(best_m['price'])}")
            picks_html = "".join(f'<span class="chip mk" style="margin-right:6px">{x}</span>' for x in (v.get("picks") or []))
            cols[i % 2].markdown(f"""
            <div class="card">
              <div class="head"><span class="matchup">{a} @ {h}</span>{chip}</div>
              <div class="score">{v['h1_line']} <span class="muted">at half</span></div>
              <div class="muted">2H projection {v['mu_2h']:.1f} (pre-game {v['pre_2h']:.1f}){res}</div>
              <div class="grid">
                {tile("2H total", fmt(v['mu_2h']), f"Line {fmt(v.get('h2_total_line'))} {lean(v['mu_2h'], v.get('h2_total_line'))}")}
                {tile("2H margin", f"{h} {v['mu_2hm']:+.1f}", f"Line {fmt(v.get('h2_spread_home'), 1, True)}" if v.get('h2_spread_home') is not None else "no line")}
                {tile("Win prob now", f"{h} {v['p_home_win']:.0%}", f"Fair {american(v['p_home_win'])}")}
                {bl}
              </div>
              {('<div style="margin-top:8px">' + picks_html + '</div>') if picks_html else ''}
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No games priced at halftime yet. On game nights the evening job (`nba_live.py`, GitHub Action "
                "`live.yml`) fills this in as each game reaches the half. You can price any game by hand below.")

    st.markdown("#### Price a second half")
    opts = {CUSTOM: None}
    for v in priced_tonight:
        opts[f"{v['away']} @ {v['home']}  (priced at half)"] = {"game_id": v["game_id"], "home": v["home"], "away": v["away"],
                                                                "h1": (v["h1_home"], v["h1_away"]), "v": v}
    recent_log = plog[pd.to_datetime(plog["date"]) >= pd.Timestamp(datetime.now(ET).date()) - pd.Timedelta(days=1)] \
        if not plog.empty and "date" in plog else pd.DataFrame()
    for _, row in recent_log.iterrows():
        opts.setdefault(f"{row['away']} @ {row['home']}  ({row['date'][5:]})",
                        {"game_id": row["game_id"], "home": row["home"], "away": row["away"], "h1": None, "v": None})
    for s in slate:
        g = s["g"]
        opts.setdefault(f"{g['away']} @ {g['home']}  ({g['date'][5:]})",
                        {"game_id": g["game_id"], "home": g["home"], "away": g["away"], "h1": None, "v": None})
    choice = st.selectbox("Game", list(opts), index=1 if len(opts) > 1 else 0, key="ht_game")
    hg = opts[choice]
    kk = str(hg["game_id"]) if hg else "custom"
    teams_ = pred.active_teams
    c1, c2 = st.columns(2)
    if hg:
        ht_home, ht_away = hg["home"], hg["away"]
        c1.markdown(f"**{ht_away} @ {ht_home}**")
    else:
        ht_away = c1.selectbox("Away team", teams_, index=0, key="ht_away")
        ht_home = c2.selectbox("Home team", teams_, index=1, key="ht_home")
    if ht_home == ht_away:
        st.warning("Pick two different teams.")
    else:
        pre = pre_for(hg["game_id"] if hg else None, ht_home, ht_away)
        d_h1 = hg["h1"] if hg and hg["h1"] else (round(pre["h1_h"]), round(pre["h1_a"]))
        s1, s2, s3, s4 = st.columns(4)
        h1_away_pts = s1.number_input(f"{ht_away} 1H points", 20, 120, int(d_h1[1]), 1, key=f"ht_a1_{kk}")
        h1_home_pts = s2.number_input(f"{ht_home} 1H points", 20, 120, int(d_h1[0]), 1, key=f"ht_h1_{kk}")
        base_ht = pred.halftime(h1_home_pts, h1_away_pts, pre)
        v = (hg or {}).get("v") or {}
        rnd = lambda x: round(x * 2) / 2  # noqa: E731
        tl_ = s3.number_input("2H total line", 60.0, 160.0, float(v.get("h2_total_line") or rnd(base_ht["mu_2h"])), 0.5,
                              key=f"ht_tl_{kk}")
        sp_ = s4.number_input(f"{ht_home} 2H spread", -30.0, 30.0,
                              float(v.get("h2_spread_home") if v.get("h2_spread_home") is not None else rnd(-base_ht["mu_2hm"])),
                              0.5, key=f"ht_sp_{kk}", help="-2.5 = home lays 2.5 in the second half")
        p1, p2, p3, p4 = st.columns(4)
        po = p1.number_input("Over price", -300, 300, -110, 5, key=f"ht_po_{kk}")
        pu = p2.number_input("Under price", -300, 300, -110, 5, key=f"ht_pu_{kk}")
        ph = p3.number_input(f"{ht_home} price", -300, 300, -110, 5, key=f"ht_ph_{kk}")
        pa = p4.number_input(f"{ht_away} price", -300, 300, -110, 5, key=f"ht_pa_{kk}")
        ht = pred.halftime(h1_home_pts, h1_away_pts, pre, tl_, sp_)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("2H total", f"{ht['mu_2h']:.1f}", f"{ht['mu_2h'] - ht['pre_2h']:+.1f} vs pre-game", delta_color="off")
        m2.metric(f"2H margin ({ht_home})", f"{ht['mu_2hm']:+.1f}")
        m3.metric(f"{ht_home} wins the game", pct(ht["p_home_win"]))
        m4.metric("Projected final", f"{h1_away_pts + (ht['mu_2h'] - ht['mu_2hm']) / 2:.0f}–"
                                     f"{h1_home_pts + (ht['mu_2h'] + ht['mu_2hm']) / 2:.0f}")
        rows_ = halftime_rows(ht, ht_home, ht_away, {
            "h2_total_line": tl_, "h2_total_over_price": po, "h2_total_under_price": pu,
            "h2_spread_home": sp_, "h2_spread_home_price": ph, "h2_spread_away_price": pa})
        pr = price(rows_)
        st.dataframe(pd.DataFrame({
            "Market": pr["label"], "Model": pr["p_cond"].map(pct),
            "Push": pr["push"].map(lambda x: pct(x) if x > 1e-9 else ""),
            "Fair odds": pr["p_cond"].map(fair), "Book (no-vig)": pr["book"].map(pct),
            "Edge": pr["edge"].map(lambda x: f"{100 * x:+.1f} pts"), "EV per $100": pr["ev"].map(lambda x: f"{100 * x:+.2f}"),
            "¼ Kelly": pr["kelly"].map(lambda x: f"{100 * x:.1f}%" if x > 0 else "no bet")}), hide_index=True, width="stretch")
        ht_meta = pred.art.get("halftime") or {}
        st.caption("The second half is projected from the pre-game numbers, adjusted for how the first half went: a "
                   "faster or slower first half than expected carries partly into the second, and big leads tend to "
                   "shrink as starters sit. Fitted on "
                   + (f"{ht_meta.get('n_fit', 0):,} out-of-sample games." if ht_meta.get("fitted")
                      else "default relationships (not enough history yet).")
                   + " It doesn't know about in-game injuries or foul trouble.")

# ================================================================== Track record
with tab_track:
    st.markdown("#### Selections")
    done = picks[picks["status"].isin(["W", "L", "P"])].copy() if not picks.empty else pd.DataFrame()
    if done.empty:
        st.info("No graded selections yet. Results fill in on the first refresh after each game.")
    else:
        done["date"] = pd.to_datetime(done["date"])
        f1, f2 = st.columns([2, 3])
        window = f1.segmented_control("Period", ["7D", "30D", "Season", "All"], default="All", label_visibility="collapsed")
        mks = f2.pills("Market filter", MARKETS, selection_mode="multi", default=MARKETS, label_visibility="collapsed",
                       key="rec_mk")
        last = done["date"].max()
        if window in ("7D", "30D"):
            done = done[done["date"] > last - pd.Timedelta(days=int(window[:-1]))]
        elif window == "Season":
            done = done[done["date"] >= pd.Timestamp(last.year if last.month >= 8 else last.year - 1, 8, 1)]
        done = done[done["market"].isin(mks or MARKETS)]

        def summary(df):
            w, l, p = (df["status"] == "W").sum(), (df["status"] == "L").sum(), (df["status"] == "P").sum()
            u = df["profit"].sum()
            return {"Picks": len(df), "Record": f"{w}-{l}-{p}", "Win %": w / (w + l) if w + l else np.nan,
                    "Units": u, "ROI": u / len(df) if len(df) else np.nan}

        s = summary(done)
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Record", s["Record"])
        t2.metric("Win %", pct(s["Win %"]), help="Break-even at -110 is 52.4%")
        t3.metric("Units (1u flat)", f"{s['Units']:+.2f}")
        t4.metric("ROI", pct(s["ROI"]))
        if not done.empty:
            st.area_chart(done.groupby("date")["profit"].sum().cumsum().rename("Cumulative units"), height=210)
            cfg = {"Win %": st.column_config.NumberColumn(format="percent"),
                   "Units": st.column_config.NumberColumn(format="%+.2f"),
                   "ROI": st.column_config.NumberColumn(format="percent")}
            st.dataframe(pd.DataFrame([{"Market": m, **summary(done[done["market"] == m])} for m in MARKETS
                                       if (done["market"] == m).any()]), hide_index=True, width="stretch", column_config=cfg)
            st.dataframe(pd.DataFrame([{"Confidence": "★" * k_, **summary(done[done["stars"] == k_])} for k_ in (1, 2, 3)
                                       if (done["stars"] == k_).any()]), hide_index=True, width="stretch", column_config=cfg)
            with st.expander("Pick history"):
                hist = done.sort_values("date", ascending=False)
                st.dataframe(pd.DataFrame({
                    "Date": hist["date"].dt.strftime("%b %-d"), "Game": hist["matchup"], "Pick": hist["selection"],
                    "Price": hist["price"].map(price_txt), "Edge": hist["edge"].map(lambda x: f"{100 * x:+.1f}"),
                    "Result": hist["actual"].map(lambda x: fmt(x, 0)), "W/L": hist["status"],
                    "Units": hist["profit"].map(lambda x: f"{x:+.2f}")}), hide_index=True, width="stretch")

    st.markdown("#### Every game the model projected")
    settled = plog[plog["home_score"].notna()].copy() if not plog.empty and "home_score" in plog else pd.DataFrame()
    if settled.empty:
        st.info(f"{len(plog)} games logged so far; none settled yet." if not plog.empty
                else "No prediction log yet. It starts the first time refresh.py runs with games on the schedule.")
    else:
        y = (settled["home_score"] > settled["away_score"]).astype(int)
        p = settled["p_home_win"]
        act_tot = settled["home_score"] + settled["away_score"]
        act_m = settled["home_score"] - settled["away_score"]
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Games settled", f"{len(settled):,}")
        a2.metric("Model's winner right", pct(((p >= 0.5) == (y == 1)).mean()))
        a3.metric("Moneyline Brier", f"{((p - y) ** 2).mean():.4f}",
                  delta=f"{((p - y) ** 2).mean() - ((y.mean() - y) ** 2).mean():+.4f} vs constant", delta_color="inverse",
                  help="Lower is better. 0.25 = always saying 50%.")
        a4.metric("Total MAE", fmt((settled["exp_total"] - act_tot).abs().mean(), 2), help="Average miss on the full-game total")

        rows = []
        tl = settled["total_line"].notna()
        if tl.any():
            side_over = settled.loc[tl, "exp_total"] > settled.loc[tl, "total_line"]
            res = act_tot[tl] - settled.loc[tl, "total_line"]
            won = np.where(side_over, res > 0, res < 0)
            rows.append({"Check": "Total vs market", "Games": int(tl.sum()),
                         "Model MAE": (settled.loc[tl, "exp_total"] - act_tot[tl]).abs().mean(),
                         "Market MAE": res.abs().mean(),
                         "Model side record": f"{int(won[res != 0].sum())}-{int((~won)[res != 0].sum())}-{int((res == 0).sum())}"})
        sp = settled["spread_home"].notna()
        if sp.any():
            side_home = settled.loc[sp, "exp_margin"] > -settled.loc[sp, "spread_home"]
            res = act_m[sp] + settled.loc[sp, "spread_home"]
            won = np.where(side_home, res > 0, res < 0)
            rows.append({"Check": "Spread vs market", "Games": int(sp.sum()),
                         "Model MAE": (settled.loc[sp, "exp_margin"] - act_m[sp]).abs().mean(),
                         "Market MAE": (-settled.loc[sp, "spread_home"] - act_m[sp]).abs().mean(),
                         "Model side record": f"{int(won[res != 0].sum())}-{int((~won)[res != 0].sum())}-{int((res == 0).sum())}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Model MAE": st.column_config.NumberColumn(format="%.2f"),
                "Market MAE": st.column_config.NumberColumn(format="%.2f")})
            st.caption("'Model side record' grades the model's lean on every game with a line, edge or not. "
                       "Beating the market's error over hundreds of games is the real test.")
        if len(settled) >= 40:
            cal = settled.assign(y=y, bucket=pd.cut(p, [0, .3, .4, .5, .6, .7, 1])).groupby("bucket", observed=True).agg(
                Predicted=("p_home_win", "mean"), Actual=("y", "mean"), Games=("y", "size")).reset_index()
            cal["bucket"] = cal["bucket"].astype(str)
            st.markdown("**Calibration (home win probability)**")
            st.dataframe(cal.round(3), hide_index=True, width="stretch")

    hs_ = htlog[htlog["act_2h"].notna()] if not htlog.empty and "act_2h" in htlog else pd.DataFrame()
    if not hs_.empty:
        st.markdown("#### Halftime projections")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Games priced at half", f"{len(hs_):,}", help="Includes games first seen after the half; those count "
                  "for accuracy but never produce selections")
        b2.metric("2H total MAE", fmt((hs_["act_2h"] - hs_["mu_2h"]).abs().mean(), 2),
                  delta=f"{(hs_['act_2h'] - hs_['mu_2h']).abs().mean() - (hs_['act_2h'] - hs_['pre_2h']).abs().mean():+.2f} vs pre-game",
                  delta_color="inverse")
        b3.metric("2H margin MAE", fmt((hs_["act_2hm"] - hs_["mu_2hm"]).abs().mean(), 2),
                  delta=f"{(hs_['act_2hm'] - hs_['mu_2hm']).abs().mean() - (hs_['act_2hm'] - hs_['pre_2hm']).abs().mean():+.2f} vs pre-game",
                  delta_color="inverse")
        wl = hs_[hs_["h2_total_line"].notna()] if "h2_total_line" in hs_ else pd.DataFrame()
        if not wl.empty:
            res = wl["act_2h"] - wl["h2_total_line"]
            won = np.where(wl["mu_2h"] > wl["h2_total_line"], res > 0, res < 0)
            b4.metric("Lean vs 2H line", f"{int(won[res != 0].sum())}-{int((~won)[res != 0].sum())}-{int((res == 0).sum())}",
                      help="The model's side of the book's halftime 2H total, every game with a line, edge or not")

    bt = meta.get("backtest")
    if bt:
        st.markdown(f"#### Walk-forward backtest ({bt['n_games']:,} games, seasons {', '.join(bt['seasons'])})")
        rows = [{"Check": "Moneyline Brier", "Model": f"{bt['win_brier']:.4f}", "Reference": f"{bt['const_brier']:.4f} (constant home rate)"},
                {"Check": "Winner accuracy", "Model": pct(bt["win_accuracy"]), "Reference": ""},
                {"Check": "Full-game total MAE", "Model": f"{bt['total_mae']:.2f}", "Reference": (
                    f"{bt['market_total_mae']:.2f} market on {bt['market_total_n']:,} games (model {bt['model_total_mae_same_games']:.2f})"
                    if bt.get("market_total_mae") else "")},
                {"Check": "Margin MAE (regulation)", "Model": f"{bt['margin_mae']:.2f}", "Reference": ""},
                {"Check": "1H total MAE", "Model": f"{bt['h1_mae']:.2f}", "Reference": ""},
                {"Check": "2H total MAE", "Model": f"{bt['h2_mae']:.2f}", "Reference": ""},
                {"Check": "Early-season Brier (first 30 days)", "Model": f"{bt['early_win_brier']:.4f}", "Reference": f"{bt['early_n']:,} games"}]
        bh = bt.get("halftime")
        if bh:
            rows += [{"Check": "2H total MAE at halftime", "Model": f"{bh['h2_mae_halftime']:.2f}",
                      "Reference": f"{bh['h2_mae_pregame']:.2f} pre-game ({bh['n_games']:,} games)"},
                     {"Check": "2H margin MAE at halftime", "Model": f"{bh['h2_margin_mae_halftime']:.2f}",
                      "Reference": f"{bh['h2_margin_mae_pregame']:.2f} pre-game"}]
        wa = bt.get("with_availability")
        if wa:
            rows += [{"Check": "With known availability: Brier", "Model": f"{wa['win_brier']:.4f}",
                      "Reference": f"{wa['baseline_same_games']['win_brier']:.4f} without"},
                     {"Check": "With known availability: margin MAE", "Model": f"{wa['margin_mae']:.2f}",
                      "Reference": f"{wa['baseline_same_games']['margin_mae']:.2f} without"}]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if wa:
            st.caption("'Known availability' uses who actually played, with effect sizes fitted on the other test "
                       "season. It's the upper bound of what injury news is worth; live projections use the injury "
                       "report's probabilities instead.")
        if bt.get("calibration"):
            st.dataframe(pd.DataFrame(bt["calibration"]).rename(columns={"bucket": "Home win prob.", "predicted": "Predicted",
                         "actual": "Actual", "n": "Games"}).round(3), hide_index=True, width="stretch")

# ================================================================== Model
with tab_model:
    st.markdown("**Team ratings**" + (" (per 100 possessions; DRtg = points allowed, lower is better)" if pred.box_on else ""))
    tr = pred.team_ratings()
    st.dataframe(tr.round(1), width="stretch", height=560)
    if pred.av.get("on"):
        st.markdown("**Who matters most right now** (margin swing if he sits, minutes going to teammates)")
        rows = []
        for t in pred.active_teams:
            for pl in player_impacts(pred.roster(t), pred.av):
                rows.append({"Player": pl["name"], "Team": t, "Status": pl["status"], "Plays": pl["p_play"],
                             "Usual min": pl["mwp"], "Swing (pts)": pl["impact"]})
        if rows:
            imp = pd.DataFrame(rows).sort_values("Swing (pts)").head(40)
            st.dataframe(imp, hide_index=True, width="stretch", height=420, column_config={
                "Plays": st.column_config.NumberColumn(format="percent"),
                "Usual min": st.column_config.NumberColumn(format="%.0f"),
                "Swing (pts)": st.column_config.NumberColumn(format="%+.1f")})
        a = pred.av
        rates = ", ".join(f"{k} {v:.0%}" + (f" (n={a['status_counts'][k]})" if a.get("status_counts", {}).get(k) else "")
                          for k, v in a["play_rates"].items() if k in ("Probable", "Questionable", "Doubtful", "Day-To-Day"))
        st.caption(f"Availability: each unit of lost player value costs the team {abs(a['b_own']):.2f} pts and gives "
                   f"the opponent {a['b_opp']:+.2f} (fitted on {a['n_fit']:,} out-of-sample games). Chance a listed "
                   f"player plays: {rates}. Rates switch from defaults to your own measured ones as injury "
                   f"snapshots build up.")
    tc = meta.get("travel_coefs") or {}
    if tc:
        names = {"three_in_four": "3 games in 4 nights", "opp_three_in_four": "opponent on 3 in 4",
                 "tz": "per time zone crossed", "opp_tz": "opponent, per time zone", "trip": "per road-trip game",
                 "opp_trip": "opponent, per road-trip game", "alt": "visiting at altitude"}
        st.caption("Rest & travel (" + meta["units"] + ", fitted on ~2-year weights): "
                   + " · ".join(f"{names.get(k, k)} {v:+.1f}" for k, v in tc.items())
                   + f". A slate card names a team's travel only when it moves the margin {TRAVEL_SHOW:g}+ pts.")
    sd = pred.sig
    st.caption(
        f"Recency half-life {meta['half_life_days']:g} days · prior {meta['prior_games']:g} games · past-season weight "
        f"{meta['carryover']:g} · home edge {meta['home_coef']:+.1f} {meta['units']} · team on a back-to-back "
        f"{meta['b2b_coef']:+.1f} · score spread (SD): total {sd['total_reg']:.1f}, margin {sd['margin_reg']:.1f}, "
        f"1H {sd['h1_total']:.1f}, 2H {sd['h2_total']:.1f} · fitted {meta['fitted_at'][:16].replace('T', ' ')}. "
        "The model predicts regulation scoring; ties are settled with real overtime scoring, and the margin "
        "distribution is calibrated to actual close-game frequencies. For information and entertainment only - "
        "not betting advice.")
