# 🏀 NBA Game Predictor

Same design as the NHL predictor: a local (or scheduled) `refresh.py` fetches data, retrains, and writes one small model file plus logs. The Streamlit app reads only those files — no API calls — so it loads fast on phone and desktop.

**Markets:** full-game total · team totals · 1H total · 2H total · spread · moneyline (plus alt spreads).
**App tabs:** 🎯 Selections · 📅 Today's Slate · 🧮 Game Markets (enter your odds → edge, EV, ¼ Kelly) · 📈 Track Record · 📊 Model.

## How it maps to the NHL version

| NHL | NBA | Notes |
|---|---|---|
| `nhl_backfill.py` | `nba_backfill.py` | ESPN daily scoreboards: final + quarter scores, OT points, posted line. Resumable. |
| `nhl_periods.py` | (in backfill) | Quarter scores come with the scoreboard, no extra calls. |
| `nhl_goalies.py` | `nba_boxscores.py` | ESPN box scores → **true possessions** (FGA − OREB + TOV + 0.44·FTA), plus every player's minutes and DNP reason. Resumable, checkpoints every 250 games. |
| `nhl_update.py` / `nhl_schedule.py` | `nba_update.py` / `nba_schedule.py` | Settle recent games; pull the next 4 days with ESPN's spread/total/moneyline. Pre-game lines are kept after ESPN drops them, so line history builds up over the season. |
| — | `nba_odds.py` | Optional The Odds API: prices, team totals, 1H/2H lines. |
| `nhl_model.py` | `nba_model.py` | Ratings → joint score grid → every market. Details below. |
| `goalie_features.py` | `nba_availability.py` + `nba_injuries.py` | Player availability (the NBA version of the goalie module): who's out, what they're worth, applied automatically. |
| `refresh.py`, `tune_model.py` | same names | Same flags and behaviour: `--push`, `--skip-fetch`, `--backtest`; paired-bootstrap tuning. |
| `streamlit_app.py` | same name | NHL calculator kept as **Game Markets**; Selections, Slate cards and pick record added. |

## What never enters the model

Only **regular season, play-in and playoff** games count. **Preseason** (ESPN season type 1), **summer league**, All-Star and exhibitions are excluded at every step: when scoreboards are parsed, again when stored data is loaded (anything older code saved is dropped), when box scores are fetched, and therefore from player values, minutes, injury-rate learning, the slate and the track record. July–September dates are rejected outright. Preseason *injury news* is still used, since who's out on opening night matters; preseason *results and minutes* are not.

## The model (`nba_model.py`)

- **Team ratings** by weighted ridge regression with team (offense) and opponent (defense) terms, home court (zero at neutral sites) and back-to-back flags:
  - *efficiency* — points per 100 possessions, and *pace* — possessions. Regulation points = pace × efficiency / 100. This separates "plays fast" (moves totals) from "scores well" (moves spreads).
  - a separate rating set for **first-half points**.
  - Without enough box-score coverage (<95 % of games) it falls back to a direct points rating, like the NHL goalie switch.
- **Recency half-life**, **season carryover** and a **ridge prior worth N games of league-average play**: ratings regress toward average when there's little current-season data, never toward a reference team.
- **Regulation target + real overtime.** A tie after 48 minutes is settled with the league's actual OT scoring, so totals, team totals, spread and moneyline all come from one joint final-score grid with correct push probabilities.
- **Margin calibration.** Normal margins under-predict regulation ties and close-game margins; the grid is re-weighted by |margin| with factors fitted to out-of-sample games (same idea as your NHL margin/total weights).
- **Uncertainty from out-of-sample errors only.** Score spreads (SDs) are measured on the last year of walk-forward predictions, refit every 14 days.
- **1H / 2H:** 1H from its own ratings; 2H = full game − 1H, including OT (how books grade it).

Tunable defaults at the top of `nba_model.py`: `DEFAULT_HALF_LIFE`, `DEFAULT_PRIOR_GAMES`, `DEFAULT_CARRYOVER`. Selection thresholds at the top of `nba_picks.py`.

## Player availability (injuries)

Works like the NHL "usual goalie mix", except it's automatic and only shows what matters.

- **`nba_injuries.py`** pulls ESPN's injury list and current rosters (catches trades and off-season moves), and, when `pdfplumber` is installed, the official NBA injury report PDF for the finer statuses (Out / Doubtful / Questionable / Probable). ESPN player ids match the box scores, so no name-matching guesswork except for the official PDF.
- **Player value** = recency-weighted Game Score per minute from the box scores, shrunk toward replacement level. **Baseline** = the player's share of team minutes over recent games (games missed count as zero), which is what the team rating already assumes.
- **Tonight:** each player's usual minutes × the chance he plays, rescaled so the team plays 240 minutes (a sitting star's minutes go to teammates). Lost value = Σ value × (baseline − tonight). How many points a unit of lost value costs the team, and gives the opponent, is fitted on out-of-sample games.
- **Chance a listed player plays** starts from defaults (Questionable 55 %, Doubtful 12 %, Probable 93 %, Day-To-Day 60 %). Every refresh saves an injury snapshot; once 150+ snapshots per status have finished games, the model switches to the rates measured from your own data.
- **In the app:** slate cards get one line, e.g. `🩹 Injury adj: BOS −4.2 · Tatum out · Holiday Q (55% plays)`, naming only players worth ≥ 0.5 pts. Game Markets has an Auto / Plays / Sits switch for each of those players, and everything recalculates. A selection that only clears the edge threshold in one outcome of a questionable player says **⚠ depends on Holiday sitting**.
- **Backtest:** "with known availability" re-scores the backtest using who actually played, with effects fitted on the other test season. It's the ceiling for what injury news is worth.

## Selections

For every slate game with a posted line, each market's sides are priced with the posted odds (−110 when a spread/total has no price). The book's margin is removed (no-vig), pushes are excluded from the model's win chance, and a side is selected when **edge ≥ 4 pts (5 for team totals and halves), EV > 0 and edge < 25 pts** (bigger "edges" are almost always stale lines or injury news). Stars: ★ 4+, ★★ 7+, ★★★ 10+ pts. Each card shows ¼ Kelly. Picks stay live until tip-off, then lock and are graded at the stored price.

## 1. Set up in VS Code

Use **Python 3.11** (same as the GitHub Action and Streamlit Cloud).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
pytest -q                            # offline tests on a synthetic league, a few seconds
```

`.vscode/launch.json` has run configs for the app, refresh, backtest and tuning.

## 2. First-time data + model

```bash
python nba_backfill.py               # 4 seasons of scoreboards, ~5 min
python nba_boxscores.py              # ~5,000 box scores, ~10-15 min, resumable (Ctrl-C safe)
python nba_injuries.py               # injuries + rosters (also run by refresh.py)
python refresh.py --backtest         # update, schedule, retrain, backtest, logs
python tune_model.py                 # optional, ~1-2 min: tells you which DEFAULT_ lines to change
streamlit run streamlit_app.py
```

## 3. Day to day

Either run it yourself, exactly like the NHL app:

```bash
python refresh.py --push             # fetch, retrain, log predictions + selections, commit and push
python refresh.py --push --period-odds   # evening run: also buy team totals + 1H/2H lines
```

…or let **GitHub Actions** do it (`.github/workflows/refresh.yml`): 6 AM ET settles last night, noon ET posts the slate, 6:30 PM ET refreshes lines before tip-off. If you mix both, `refresh.py` tells you to `git pull --rebase` when the Action pushed first.

## 4. Optional: sportsbook lines (The Odds API)

ESPN only posts spread, total and moneyline. For prices, **team totals and 1H/2H lines**, get a key at the-odds-api.com and set `ODDS_API_KEY` (shell for local runs; repo → Settings → Secrets → Actions for the workflow). Cost: 3 credits per run for spread/total/moneyline on every game; `--period-odds` adds ~3 credits per game today. Evening-only `--period-odds` is ~1,000 credits/month in season — above the free 500, so on the free tier drop `--period-odds` from the workflow. Without period lines, those markets still show projections, and you can price them by hand in **Game Markets**.

## 5. Deploy

1. Push the repo with `model/` and `data/*.parquet` (CSV twins are git-ignored).
2. Repo → Settings → Actions → General → Workflow permissions → **Read and write** (only if you use the Action).
3. share.streamlit.io → New app → this repo, branch `main`, file `streamlit_app.py`, Python 3.11.

Every push (yours or the Action's) redeploys the app in about a minute.

## Limits worth knowing

- **Late scratches.** Statuses are as of the last refresh (the official game-day report is due by 1 p.m. local); news in the last hour before tip needs the Plays/Sits switch.
- **Value is box-score based.** Game Score captures scoring and playmaking better than defense, so elite defenders are under-valued. On/off plus-minus is the natural upgrade once a season of data is in.
- **Early season** leans on last season (shrunk by carryover) until teams have ~10 games; the app flags it.
- **Historical lines aren't in ESPN's archive.** The backtest scores projections and probabilities; betting results start the day you begin running `refresh.py`, and the market comparison grows from there.

*For information and entertainment only — not betting advice.*
