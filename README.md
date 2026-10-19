# 🏀 NBA Game Predictor

Same design as the NHL predictor: a local (or scheduled) `refresh.py` fetches data, retrains, and writes one small model file plus logs. The Streamlit app reads only those files — no API calls — so it loads fast on phone and desktop.

**Markets:** full-game total · team totals · 1H total · 2H total · spread · moneyline (plus alt spreads) · **2H total and 2H spread at halftime**.
**App tabs:** 🎯 Selections · 📅 Today's Slate · 🧮 Game Markets (enter your odds → edge, EV, ¼ Kelly) · ⏱ Halftime · 📈 Track Record · 📊 Model.

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
| — | `nba_live.py` + `.github/workflows/live.yml` | Halftime: prices each game's second half from its first-half score as it happens. |
| `streamlit_app.py` | same name | NHL calculator kept as **Game Markets**; Selections, Slate cards and pick record added. |

## What never enters the model

Only **regular season, play-in and playoff** games count. **Preseason** (ESPN season type 1), **summer league**, All-Star and exhibitions are excluded at every step: when scoreboards are parsed, again when stored data is loaded (anything older code saved is dropped), when box scores are fetched, and therefore from player values, minutes, injury-rate learning, the slate and the track record. July–September dates are rejected outright. Preseason *injury news* is still used, since who's out on opening night matters; preseason *results and minutes* are not.

## The model (`nba_model.py`)

- **Team ratings** by weighted ridge regression with team (offense) and opponent (defense) terms, home court (zero at neutral sites), back-to-back flags and travel (below):
  - *efficiency* — points per 100 possessions, and *pace* — possessions. Regulation points = pace × efficiency / 100. This separates "plays fast" (moves totals) from "scores well" (moves spreads).
  - a separate rating set for **first-half points**.
  - Without enough box-score coverage (<95 % of games) it falls back to a direct points rating, like the NHL goalie switch.
- **Recency half-life**, **season carryover** and a **ridge prior worth N games of league-average play**: ratings regress toward average when there's little current-season data, never toward a reference team.
- **Regulation target + real overtime.** A tie after 48 minutes is settled with the league's actual OT scoring, so totals, team totals, spread and moneyline all come from one joint final-score grid with correct push probabilities.
- **Margin calibration.** Normal margins under-predict regulation ties and close-game margins; the grid is re-weighted by |margin| with factors fitted to out-of-sample games (same idea as your NHL margin/total weights).
- **Uncertainty from out-of-sample errors only.** Score spreads (SDs) are measured on the last year of walk-forward predictions, refit every 14 days.
- **1H / 2H:** 1H from its own ratings; 2H = full game − 1H, including OT (how books grade it).

Tunable defaults at the top of `nba_model.py`: `DEFAULT_HALF_LIFE`, `DEFAULT_PRIOR_GAMES`, `DEFAULT_CARRYOVER`, `DEFAULT_TRAVEL`. Selection thresholds at the top of `nba_picks.py`.

## Rest and travel

On top of back-to-backs, each team's schedule gives four more inputs, for the team and its opponent:
**3 games in 4 nights**, **time zones crossed** since its last game (only if that was within 2 days),
**road-trip game number** (resets after 5+ days off) and **visiting at altitude** (DEN, UTAH; visitor side only,
since two teams can't separate "visitors score less" from "the home side scores more").

- These are league-wide effects that change slowly, so they're fitted on ~2-year weights (`TRAVEL_HALF_LIFE`).
  The team ratings are then fitted around them with the usual recency half-life. Fitting them on the 120-day
  ratings weights made them mostly noise.
- Neutral sites (NBA Cup, international games) get no time-zone or altitude value, because the location isn't in
  ESPN's scoreboard data.
- **In the app** they only show up when they matter: a slate card gets `✈ Travel: LAL −1.2 (3 in 4 nights · 2 time zones)`
  when a team's travel moves the margin by 0.5+ pts. The fitted effects are listed under 📊 Model.
- **Check they earn their place:** `python tune_model.py --only travel --metric margin_abs` compares travel on vs off
  on the same backtest games. If "off" wins, set `DEFAULT_TRAVEL = 0` (or run `refresh.py --travel 0`).

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

### Correlated exposure

Selections in the same game share risk: a moneyline and a spread on the same team, or an Under plus a team-total
Under, mostly win and lose together. So a game's selections are **sized as a package**:

- Outcomes are simulated from the model's joint final-score grid (20,000 games; the 1H total is drawn given the
  full-game total, using the correlation implied by the fitted score spreads).
- Stakes maximise expected log bankroll for the set (joint Kelly), then ×¼. A bet is **never sized up** for
  hedging another, because that benefit would rest on the model's correlations. All of a game's stakes together are capped at
  `MAX_GAME_EXPOSURE` (5 % of bankroll).
- **Cards** show the package stake, with the single-bet stake underneath (`solo 2.4% · sized with 2 other picks`),
  and name the pick it moves with (`📎 moves with BOS −4.5 (+0.76)`) when the correlation is 0.3+.
- **Game Markets** shows `¼ Kelly (alone)` and `¼ Kelly (together)` for whatever you price.

The track record still grades every pick at 1 unit, so win rate and ROI aren't affected by the sizing.

## Halftime (second-half pricing)

`nba_live.py` watches ESPN's live scoreboard. When a game reaches halftime it prices the second half from the
first-half score and the pre-game projection (saved by the last refresh before tip-off, so with that evening's
injury news):

- **2H total** = pre-game 2H + a share of how far the first half ran over or under its projection (pace carries
  over), adjusted for the size of the lead (blowouts slow down). **2H margin** = pre-game 2H margin, adjusted
  for the first-half margin (leaders ease off). Both are fitted on the same out-of-sample games as the score spreads,
  with defaults until 300 games exist. The backtest shows 2H error before the game vs at halftime.
- Also gives the **live win probability** (first-half margin + the 2H margin distribution, ties settled with OT).
- Writes `model/live.json`, `model/halftime_log.csv` (graded when games end → Track Record) and, when it buys
  live lines, `model/live_picks.csv` (2H selections, same thresholds as the 2H pre-game market: 5 pts).
- A game first seen after the 3rd quarter starts (e.g. a late GitHub run) is still logged for accuracy, marked
  late, with no line and no selection.
- **App:** the ⏱ Halftime tab shows tonight's priced games and has a **manual pricer** (type the 1H score and your
  book's 2H line), so it works even when the job hasn't run.
- **Live 2H lines** (optional): with `--odds` (or repo variable `LIVE_ODDS=1` for the Action) it buys each game's
  2H total + 2H spread from The Odds API at halftime: ~2 credits per game. Not every book posts these, so some
  games will have no line.

```bash
python nba_live.py                    # one pass
python nba_live.py --watch 345 --push # what the Action runs: poll every 90 s, push when something changes
```

`nba_live.py` owns those three files and `refresh.py` never writes them, so the evening Action and a refresh can
both push. Either one rebases onto the other automatically if it lands second.

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

…or let **GitHub Actions** do it (`.github/workflows/refresh.yml`): 6 AM ET settles last night, noon ET posts the slate, 6:30 PM ET refreshes lines before tip-off. If you mix both, `refresh.py` now rebases onto the Action's commit and pushes again by itself.

**Halftime job** (`.github/workflows/live.yml`): starts 7:30 PM EDT / 6:30 PM EST, prices each game at the half,
exits early on nights with no games and as soon as the last game is final. Nothing to run by hand. A full night is
~5–6 runner hours, **free on a public repo**. On a private repo that would use up the free 2,000 Action minutes a
month in about a week, so the workflow file includes a commented-out "every 10 minutes" schedule instead
(~1,300 min/month). That version only catches a game's halftime when a run lands inside the ~15-minute break.
You can also start it by hand: repo → Actions → Halftime pricing → Run workflow.

## 4. Optional: sportsbook lines (The Odds API)

ESPN only posts spread, total and moneyline. For prices, **team totals and 1H/2H lines**, get a key at the-odds-api.com and set `ODDS_API_KEY` (shell for local runs; repo → Settings → Secrets → Actions for the workflow). Cost: 3 credits per run for spread/total/moneyline on every game; `--period-odds` adds ~3 credits per game today. Evening-only `--period-odds` is ~1,000 credits/month in season — above the free 500, so on the free tier drop `--period-odds` from the workflow. Without period lines, those markets still show projections, and you can price them by hand in **Game Markets**.

## 5. Deploy

1. Push the repo with `model/` and `data/*.parquet` (CSV twins are git-ignored).
2. Repo → Settings → Actions → General → Workflow permissions → **Read and write** (only if you use the Action).
3. share.streamlit.io → New app → this repo, branch `main`, file `streamlit_app.py`, Python 3.11.

Every push (yours or the Action's) redeploys the app in about a minute.

**After updating the code** (e.g. the travel / correlated-exposure / halftime release), the model file format changed
(artifact version 2), so retrain once before or right after pushing, or the app asks you to:

```bash
python refresh.py --skip-fetch --backtest --push
```

## Limits worth knowing

- **Late scratches.** Statuses are as of the last refresh (the official game-day report is due by 1 p.m. local); news in the last hour before tip needs the Plays/Sits switch.
- **Value is box-score based.** Game Score captures scoring and playmaking better than defense, so elite defenders are under-valued. On/off plus-minus is the natural upgrade once a season of data is in.
- **Early season** leans on last season (shrunk by carryover) until teams have ~10 games; the app flags it.
- **Halftime knows the score, not the story.** In-game injuries, ejections and foul trouble aren't seen; check before
  betting a 2H number. GitHub can start the evening job 10–20+ minutes late; that only delays its start.
- **Travel effects are averages.** They don't know a team flew in the night before or rested a star by choice.
- **Historical lines aren't in ESPN's archive.** The backtest scores projections and probabilities; betting results start the day you begin running `refresh.py`, and the market comparison grows from there.

*For information and entertainment only — not betting advice.*
