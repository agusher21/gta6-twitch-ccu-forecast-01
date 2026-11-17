# GTA VI Twitch CCU Forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/agusher21/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/agusher21/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty ($250): forecast how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (19 Nov 2026, UTC), from the first hour's viewer count, as five quantiles (5/25/50/75/95%) per hour for hours 2–24. Scored by pinball loss on the log scale.

## Method

`main.py` (Python 3.13, stdlib only, deterministic, no network) fits a **spline template** of `log(viewers(h) / viewers(1))` over past big game launches, by weighted least squares:

- **Pool:** the case's comparables, excluding early-access launches (their release-day curve is contaminated by pre-launch play).
- **Weights:** similarity in hour-1 size, `exp(-|log p1 - log p1_target| / 1.0)`, times ×4 when the launch's unlock pattern (rolling midnight local vs simultaneous global) matches the target's, parsed from the case notes (negation-aware: "no early access" is not early access).
- **Amplitude:** each comparable's curve is a scaled copy of the template; the scale `a(log p1)` is regressed on hour-1 size — small hour-1 launches rise steeply through the day, huge ones stay flat (correlation −0.93 between log p1 and the evening surge).
- **Spread:** weighted residual standard deviation per hour around the scaled template.

## Backtest (practice cases, answers recovered from SullyGnome and validated: starter baseline reproduces the kit's 0.387 loss, 82.8% typical miss, 0.783 coverage exactly)

| model | practice pinball loss |
|---|---|
| House baseline (persistence) | 0.387 |
| This entry (spline template) | **0.158** |
| per case: Wolverine 0.161, RDR2 0.131, God of War Ragnarök 0.140, Spider-Man 2 0.200 | |

Cross-checked on leave-one-out over the 20 big (hour-1 ≥ 50k) comparables: 0.286 vs 0.415 for the unweighted median model.

## Run

```sh
MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
```

Reads each `<id>.in`, writes `<id>.out` with `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` — 23 rows, quantiles finite, > 0, non-decreasing.

## License

MIT — see [LICENSE](LICENSE).
