# Results — wait_per_vehicle_s (mean ± 95% CI over 10 test seeds; RL averaged over training seeds)


## Grid 2x2 (seen in training)

| scenario | Fixed-time | Actuated | Max-pressure | Max-pressure + shield | RL, no shield | RL local-only + shield | GridFlow-RL (ours) |
|---|---|---|---|---|---|---|---|
| light | 13.8 ± 0.1 | 9.8 ± 0.2 | 16.0 ± 0.1 | 16.0 ± 0.1 | **9.5 ± 0.1** | 9.5 ± 0.1 | 9.5 ± 0.1 |
| uniform | **20.6 ± 0.7** | 31.8 ± 0.6 | 22.4 ± 0.4 | 22.4 ± 0.4 | 20.6 ± 0.6 | 20.6 ± 0.6 | 20.7 ± 0.6 |
| arterial | 169.4 ± 14.3 | 22.4 ± 0.7 | 24.1 ± 0.3 | 24.2 ± 0.3 | **16.4 ± 0.5** | 16.7 ± 0.6 | 16.4 ± 0.5 |
| rush_hour | 30.6 ± 2.4 | 32.0 ± 1.7 | **28.1 ± 2.1** | 28.1 ± 2.2 | 29.8 ± 2.6 | 31.4 ± 3.1 | 29.6 ± 2.6 |
| surge | 81.7 ± 3.3 | 38.2 ± 2.3 | 31.6 ± 2.6 | 31.2 ± 1.8 | 31.0 ± 6.0 | 28.9 ± 1.5 | **28.2 ± 1.4** |
| incident | 40.9 ± 5.4 | 21.0 ± 0.5 | 21.1 ± 0.2 | 21.1 ± 0.2 | 14.3 ± 0.3 | 14.4 ± 0.3 | **14.3 ± 0.3** |

## Grid 3x3 (seen in training)

| scenario | Fixed-time | Actuated | Max-pressure | Max-pressure + shield | RL, no shield | RL local-only + shield | GridFlow-RL (ours) |
|---|---|---|---|---|---|---|---|
| light | 20.1 ± 0.2 | 14.9 ± 0.2 | 23.7 ± 0.3 | 23.4 ± 0.2 | **14.0 ± 0.1** | 14.0 ± 0.1 | 14.0 ± 0.1 |
| uniform | **29.6 ± 0.5** | 46.7 ± 0.7 | 33.3 ± 0.4 | 33.3 ± 0.4 | 30.5 ± 0.5 | 30.5 ± 0.5 | 30.5 ± 0.5 |
| arterial | 176.9 ± 9.2 | 39.9 ± 1.0 | 37.6 ± 0.6 | 37.5 ± 0.5 | 29.8 ± 0.7 | 30.2 ± 0.8 | **29.7 ± 0.7** |
| rush_hour | 38.8 ± 3.1 | 45.6 ± 2.3 | **38.4 ± 2.8** | 39.3 ± 4.0 | 38.5 ± 3.9 | 39.8 ± 4.5 | 38.5 ± 3.8 |
| surge | 88.9 ± 2.5 | 51.1 ± 2.3 | 78.0 ± 60.1 | 48.9 ± 4.3 | 55.0 ± 12.8 | **40.2 ± 1.6** | 40.6 ± 1.9 |
| incident | 44.5 ± 4.5 | 33.9 ± 0.9 | 30.7 ± 0.3 | 30.5 ± 0.4 | 22.0 ± 0.4 | 22.0 ± 0.4 | **22.0 ± 0.4** |

## Grid 4x4 (NEVER seen in training (zero-shot))

| scenario | Fixed-time | Actuated | Max-pressure | Max-pressure + shield | RL, no shield | RL local-only + shield | GridFlow-RL (ours) |
|---|---|---|---|---|---|---|---|
| light | 26.2 ± 0.2 | 20.0 ± 0.3 | 31.1 ± 0.3 | 30.5 ± 0.3 | 18.5 ± 0.2 | 18.5 ± 0.2 | **18.5 ± 0.2** |
| uniform | **38.4 ± 0.3** | 61.3 ± 0.7 | 43.6 ± 0.3 | 43.5 ± 0.2 | 40.2 ± 0.4 | 40.3 ± 0.4 | 40.2 ± 0.4 |
| arterial | 187.3 ± 10.1 | 59.2 ± 1.1 | 49.5 ± 1.3 | 49.2 ± 1.3 | **43.3 ± 2.2** | 48.1 ± 6.2 | 43.5 ± 2.3 |
| rush_hour | 47.0 ± 2.3 | 55.7 ± 2.0 | 46.9 ± 2.0 | 46.9 ± 2.3 | 45.5 ± 2.6 | 45.7 ± 2.4 | **45.3 ± 2.4** |
| surge | 97.4 ± 3.4 | 65.8 ± 2.2 | 308.7 ± 20.1 | 97.4 ± 13.5 | 119.6 ± 29.4 | **56.6 ± 4.2** | 59.4 ± 5.4 |
| incident | 40.2 ± 2.7 | 44.0 ± 0.6 | 39.4 ± 0.4 | 39.2 ± 0.4 | **28.9 ± 0.4** | 29.0 ± 0.4 | 29.0 ± 0.4 |

## Grid 6x6 (NEVER seen in training (zero-shot))

| scenario | Fixed-time | Actuated | Max-pressure | Max-pressure + shield | RL, no shield | RL local-only + shield | GridFlow-RL (ours) |
|---|---|---|---|---|---|---|---|
| light | 37.8 ± 0.2 | 28.9 ± 0.3 | 45.3 ± 0.2 | 44.3 ± 0.2 | 26.7 ± 0.1 | 26.7 ± 0.2 | **26.7 ± 0.2** |
| uniform | **55.3 ± 0.5** | 87.3 ± 0.7 | 63.4 ± 0.5 | 63.4 ± 0.6 | 58.2 ± 0.7 | 58.1 ± 0.7 | 58.2 ± 0.6 |
| arterial | 203.5 ± 6.8 | 100.5 ± 3.0 | 79.0 ± 2.9 | **78.0 ± 2.6** | 129.2 ± 32.1 | 120.2 ± 21.8 | 106.4 ± 18.4 |
| rush_hour | **64.9 ± 2.1** | 81.4 ± 1.9 | 69.1 ± 1.9 | 68.9 ± 2.3 | 83.2 ± 18.5 | 73.7 ± 8.9 | 74.5 ± 10.7 |
| surge | 111.4 ± 2.8 | 84.7 ± 1.3 | 330.7 ± 8.9 | 198.4 ± 13.3 | 163.7 ± 18.5 | **83.6 ± 5.8** | 93.5 ± 5.7 |
| incident | 51.4 ± 2.3 | 64.1 ± 0.3 | 55.8 ± 0.3 | 55.4 ± 0.3 | **42.4 ± 0.2** | 42.4 ± 0.3 | 42.4 ± 0.2 |

## GridFlow-RL vs the best classical controller in each cell

Paired difference in wait per vehicle (negative = ours better), bootstrap 95% CI over the 10 test seeds. 'Best classical' is chosen per cell, after the fact — the strictest possible comparison.

| grid | scenario | best classical | classical | ours | diff | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| 2x2 | light | Actuated | 9.8 | 9.5 | -0.4 | [-0.5, -0.2] | ours better |
| 2x2 | uniform | Fixed-time | 20.6 | 20.7 | +0.1 | [-0.2, +0.3] | no clear difference |
| 2x2 | arterial | Actuated | 22.4 | 16.4 | -6.0 | [-6.2, -5.8] | ours better |
| 2x2 | rush_hour | Max-pressure | 28.1 | 29.6 | +1.5 | [+0.6, +2.5] | ours worse |
| 2x2 | surge | Max-pressure + shield | 31.2 | 28.2 | -2.9 | [-3.9, -2.0] | ours better |
| 2x2 | incident | Actuated | 21.0 | 14.3 | -6.8 | [-7.0, -6.5] | ours better |
| 3x3 | light | Actuated | 14.9 | 14.0 | -0.9 | [-1.0, -0.8] | ours better |
| 3x3 | uniform | Fixed-time | 29.6 | 30.5 | +0.9 | [+0.7, +1.1] | ours worse |
| 3x3 | arterial | Max-pressure + shield | 37.5 | 29.7 | -7.7 | [-8.1, -7.4] | ours better |
| 3x3 | rush_hour | Max-pressure | 38.4 | 38.5 | +0.1 | [-1.1, +1.5] | no clear difference |
| 3x3 | surge | Max-pressure + shield | 48.9 | 40.6 | -8.3 | [-11.2, -5.6] | ours better |
| 3x3 | incident | Max-pressure + shield | 30.5 | 22.0 | -8.5 | [-8.6, -8.4] | ours better |
| 4x4 | light | Actuated | 20.0 | 18.5 | -1.5 | [-1.7, -1.3] | ours better |
| 4x4 | uniform | Fixed-time | 38.4 | 40.2 | +1.8 | [+1.6, +2.0] | ours worse |
| 4x4 | arterial | Max-pressure + shield | 49.2 | 43.5 | -5.7 | [-6.6, -4.7] | ours better |
| 4x4 | rush_hour | Max-pressure | 46.9 | 45.3 | -1.6 | [-2.3, -0.8] | ours better |
| 4x4 | surge | Actuated | 65.8 | 59.4 | -6.4 | [-9.6, -2.6] | ours better |
| 4x4 | incident | Max-pressure + shield | 39.2 | 29.0 | -10.2 | [-10.4, -10.0] | ours better |
| 6x6 | light | Actuated | 28.9 | 26.7 | -2.2 | [-2.4, -2.0] | ours better |
| 6x6 | uniform | Fixed-time | 55.3 | 58.2 | +2.9 | [+2.6, +3.1] | ours worse |
| 6x6 | arterial | Max-pressure + shield | 78.0 | 106.4 | +28.4 | [+13.6, +45.4] | ours worse |
| 6x6 | rush_hour | Fixed-time | 64.9 | 74.5 | +9.6 | [+2.5, +19.2] | ours worse |
| 6x6 | surge | Actuated | 84.7 | 93.5 | +8.8 | [+3.7, +14.1] | ours worse |
| 6x6 | incident | Fixed-time | 51.4 | 42.4 | -9.0 | [-11.3, -7.6] | ours better |

Ours clearly better in 15, clearly worse in 7, no clear difference in 2 of 24 cells.


## Robustness: no single classical controller is good everywhere

For each of the 24 (grid, scenario) cells, wait divided by the best controller's wait in that cell (1.00 = best). 'Worst case' is the controller's worst cell.

| controller | mean ratio | worst case | worst cell | cells where it is best |
|---|---|---|---|---|
| GridFlow-RL (ours) | 1.03 | 1.36x | 6x6 arterial | 15 |
| Actuated | 1.31 | 1.60x | 4x4 uniform | 1 |
| Max-pressure + shield | 1.33 | 2.34x | 6x6 surge | 1 |
| Max-pressure | 1.58 | 5.19x | 4x4 surge | 2 |
| Fixed-time | 2.19 | 10.30x | 2x2 arterial | 5 |

## Ablations: what does each ingredient contribute?

Mean wait per vehicle over all scenarios (s). Positive change = worse without it.

| grid | full method | Remove the shield | Remove the neighbour view |
|---|---|---|---|
| 2x2 | 19.8 | 20.3 (+3%) | 20.2 (+2%) |
| 3x3 | 29.2 | 31.6 (+8%) | 29.5 (+1%) |
| 4x4 | 39.3 | 49.3 (+25%) | 39.7 (+1%) |
| 6x6 | 67.0 | 83.9 (+25%) | 67.5 (+1%) |

## Spread across training seeds (all grids and scenarios pooled)

| controller | training seed | mean wait per vehicle (s) |
|---|---|---|
| RL local-only + shield | 0 | 40.8 |
| RL local-only + shield | 1 | 39.2 |
| RL local-only + shield | 2 | 42.5 |
| RL local-only + shield | 3 | 36.7 |
| RL local-only + shield | 4 | 36.9 |
| RL, no shield | 0 | 59.7 |
| RL, no shield | 1 | 36.4 |
| RL, no shield | 2 | 52.7 |
| RL, no shield | 3 | 41.8 |
| RL, no shield | 4 | 40.7 |
| GridFlow-RL (ours) | 0 | 42.8 |
| GridFlow-RL (ours) | 1 | 36.4 |
| GridFlow-RL (ours) | 2 | 40.6 |
| GridFlow-RL (ours) | 3 | 37.6 |
| GridFlow-RL (ours) | 4 | 36.7 |

## Secondary metrics, 3x3, all scenarios pooled

| controller | mean travel time (s) | throughput (veh/h) | max cars queued outside | switches per junction per hour |
|---|---|---|---|---|
| Fixed-time | 141.1 | 9321 | 282 | 200 |
| Actuated | 114.7 | 9487 | 73 | 155 |
| Max-pressure | 110.9 | 9446 | 62 | 153 |
| Max-pressure + shield | 111.4 | 9491 | 34 | 155 |
| RL, no shield | 105.1 | 9485 | 70 | 275 |
| RL local-only + shield | 105.3 | 9507 | 60 | 274 |
| GridFlow-RL (ours) | 105.1 | 9508 | 56 | 276 |