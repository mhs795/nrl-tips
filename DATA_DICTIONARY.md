# NRL Source Data Dictionary

Each row = one game. Fill in home/away stats AS THEY STOOD BEFORE the game was played.

---

## Game Info

| Column | Description | Example |
|--------|-------------|---------|
| season | NRL season year | 2025 |
| round | Round number (use 25+ for finals) | 12 |
| date | Match date YYYY-MM-DD | 2025-06-05 |
| venue | Ground name | Accor Stadium |
| home_team | Home team full name | South Sydney Rabbitohs |
| away_team | Away team full name | Melbourne Storm |

---

## Result (fill AFTER game)

| Column | Description |
|--------|-------------|
| home_score | Final score – home team |
| away_score | Final score – away team |
| winner | `home` or `away` |

---

## Team Stats — fill for BOTH home_ and away_ prefixes

| Column suffix | Description | Where to find |
|---------------|-------------|---------------|
| `_ladder_pos` | Current ladder position (1=top) | nrl.com ladder |
| `_season_wins/losses/draws` | Season record so far | nrl.com ladder |
| `_season_pts_for_avg` | Season avg points scored per game | nrl.com stats |
| `_season_pts_against_avg` | Season avg points conceded per game | nrl.com stats |
| `_last5_wins` | Wins in last 5 games (0–5) | nrl.com results |
| `_last5_pts_for_avg` | Avg points scored in last 5 games | nrl.com results |
| `_last5_pts_against_avg` | Avg points conceded in last 5 games | nrl.com results |
| `_last5_pts_diff_avg` | Avg margin in last 5 (pts_for − pts_against) | calculated |
| `_home_record_wins` | Home team: W at home this season | nrl.com results |
| `_home_record_played` | Home team: games played at home | nrl.com results |
| `_away_record_wins` | Away team: W away this season | nrl.com results |
| `_away_record_played` | Away team: games played away | nrl.com results |
| `_win_streak` | +N = winning streak, −N = losing streak | calculated |
| `_days_rest` | Days since last game | calculated |
| `_travel_km` | Distance from home city to venue (km) | Google Maps estimate |

---

## Advanced Stats (last 5 game averages)
Source: nrl.com/stats or rugby-league-project.com

| Column suffix | Description | Typical range |
|---------------|-------------|---------------|
| `_completion_rate` | Set completion % | 68–82% |
| `_errors_pg` | Handling errors per game | 7–16 |
| `_penalties_pg` | Penalties conceded per game | 4–10 |
| `_tackle_eff` | Tackle efficiency % | 84–93% |
| `_line_breaks_pg` | Line breaks per game | 2–8 |
| `_tries_pg` | Tries scored per game | 2–6 |
| `_post_contact_metres_pg` | Post-contact metres per game | 80–140 |
| `_kick_metres_pg` | Kick metres per game | 300–600 |

---

## Squad

| Column | Description |
|--------|-------------|
| `_key_players_out` | Count of key players out (0–5). Count: halves, fullback, starting props, hooker, captain if they are unavailable |
| `_origin_players_out` | Players missing for State of Origin |

---

## Head to Head

| Column | Description |
|--------|-------------|
| h2h_home_wins | Home team's wins in last 10 H2H meetings |
| h2h_total_meetings | Total H2H meetings counted (usually 10) |
| h2h_home_pts_for_avg | Avg points home team scores in H2H |
| h2h_away_pts_for_avg | Avg points away team scores in H2H |
| h2h_last3yr_home_wins | Home team wins in H2H last 3 years |
| h2h_last3yr_meetings | Meetings in last 3 years |

---

## Market (betting lines — helps calibrate model)

| Column | Description |
|--------|-------------|
| market_home_win_odds | Home team win odds (decimal, e.g. 1.72) |
| market_away_win_odds | Away team win odds (decimal) |
| market_home_handicap | Line/handicap — negative means home favoured (e.g. -6.5) |
| market_open_home_odds | Opening home odds (captures line movement) |
| market_open_away_odds | Opening away odds |

---

## Weather

| Column | Description |
|--------|-------------|
| weather_temp_c | Temperature °C at kick-off |
| weather_rain_mm | Rainfall (mm) on day |
| weather_wind_kmh | Wind speed (km/h) |

---

## Flags

| Column | Description |
|--------|-------------|
| is_finals | 1 if finals game |
| is_elimination_final | 1 if a loss means you're out |
| is_neutral_venue | 1 if neither team's home ground |
| crowd | Attendance |

---

## Tips for data quality

- Fill stats as they stood **before** the game (no future leakage)
- Use at least **3 full seasons** of historical data before trusting predictions
- For rounds 1–3 (low sample), use previous season data for form stats
- Weather data: Bureau of Meteorology (bom.gov.au) for Australian venues; MetService for NZ
- Advanced stats: nrl.com stats centre has most of these by team per round
