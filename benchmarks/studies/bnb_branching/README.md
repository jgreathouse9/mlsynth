# Branching rules, incumbents and node order in the certified branch and bounds (#839)

#839 asks whether the two certified branch and bounds split their nodes and
find their incumbents well, against the rules the literature compares:
BiqMac's easy first and difficult first (Rendl, Rinaldi and Wiegele 2010),
reliability branching on pseudo-costs (Achterberg, Koch and Martin 2005),
seeds for the incumbent, and the order nodes are taken in. This study runs both
searches under each choice and keeps everything else as the library has it.

This is a study, not a benchmark case. Nothing is pinned. The node counts,
the sets priced and the modelled work are deterministic; the seconds are one
machine's (four cores visible, two to four runs at a time, BLAS on one
thread) and moved with load.

## The instruments

`syndes_rules.search` is the loop of
`mlsynth.utils.syndes_helpers.branch.branch_and_bound` with the branching unit
and the incumbent's seed opened up. `marex_rules.search` is the loop of
`mlsynth.utils.marex_helpers.branch.branch_and_bound` with the branching
market, the guess at each bounded node and the node order opened up. The
bounds, their certificates, the cost model and the work limit are the
library's own functions, imported. A test holds each copy to the library count
for count under the library's choices (`test_*_rules.py`), and a property
holds every choice to the enumeration's optimum, certified, on panels small
enough to enumerate. The costs and rules decide where the time goes, never
what is proved (rule 6 of `agents/agents_combinatorial.md`).

Both copies record the modelled work at which the incumbent first came within
0.1% of the optimum (`near`). The two-way optima were certified once with the
library's search and no work limit (`results/optimum_*.json`); the MAREX
optima are those of `benchmarks/studies/marex_bnb/results/persp_n.md`.

```
python -m benchmarks.studies.bnb_branching.run optimum geolift 40 8
python -m benchmarks.studies.bnb_branching.run syndes geolift 40 8 root,easy,hard,pseudo swap
python -m benchmarks.studies.bnb_branching.run marex walmart 8 largest,pseudo on,off best
```

## SYNDES two-way

The branching unit:

- `root`, the library's: every node splits on the first free unit in the order
  of the root trust region's maximiser;
- `easy`: the free unit whose entry in the node's own trust-region maximiser
  is largest in magnitude;
- `hard`: the one whose entry is smallest in magnitude;
- `pseudo`: reliability branching. A unit's pseudo-cost is its mean bound gain
  per split, up and down; a unit with fewer than eight gains a side is tried on
  both children's trust-region bounds, at most four trials in a row without a
  better score, each trial charged as a bound.

The seed of the incumbent: `swap`, the library's (the swap search's local
optima from 32 restarts); `none`; `ascent` (the `K` largest entries of the
root maximiser, then the concave ascent of Qian, Shen and Xu until the set
stops changing).

The #834 panels at 40 units, at the default work limit of 120 modelled
seconds. `work` is modelled seconds; `gap` is (design − bound) / design at
stop, zero when the search certifies.

| panel | K | rule | seed | nodes | lifts | sets priced | work | near | certified | gap | seconds |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| GeoLift | 6 | root | swap | 97 | 4 | 3,838,380 | 15.8 | 0.0 | yes | | 7.5 |
| GeoLift | 6 | easy | swap | 78 | 4 | 1,890,588 | 10.5 | 0.0 | yes | | 4.8 |
| GeoLift | 6 | hard | swap | 58 | 3 | 1,077,699 | 7.9 | 0.0 | yes | | 3.8 |
| GeoLift | 6 | pseudo | swap | 293 | 4 | 686,639 | 8.8 | 0.0 | yes | | 4.7 |
| GeoLift | 6 | root | none | 97 | 4 | 3,838,380 | 15.8 | 2.7 | yes | | 7.1 |
| GeoLift | 6 | root | ascent | 97 | 4 | 3,838,380 | 15.8 | 2.7 | yes | | 6.3 |
| GeoLift | 8 | root | swap | 143 | 36 | 6,570,025 | 54.5 | 0.0 | yes | | 24.3 |
| GeoLift | 8 | easy | swap | 161 | 36 | 12,133,928 | 68.4 | 0.0 | yes | | 30.2 |
| GeoLift | 8 | hard | swap | 102 | 33 | 1,947,792 | 35.4 | 0.0 | yes | | 16.4 |
| GeoLift | 8 | pseudo | swap | 653 | 40 | 765,652 | 34.3 | 0.0 | yes | | 16.7 |
| GeoLift | 8 | root | none | 202 | 29 | 23,934,241 | 90.9 | 14.5 | yes | | 34.1 |
| GeoLift | 8 | root | ascent | 202 | 29 | 23,934,241 | 90.9 | 14.5 | yes | | 34.4 |
| GeoLift | 10 | root | swap | 158 | 33 | 11,403,040 | 59.6 | 46.4 | yes | | 24.7 |
| GeoLift | 10 | easy | swap | 147 | 53 | 6,940,396 | 59.7 | 44.5 | yes | | 25.5 |
| GeoLift | 10 | hard | swap | 124 | 33 | 4,665,375 | 36.4 | 26.5 | yes | | 15.1 |
| GeoLift | 10 | pseudo | swap | 423 | 37 | 123,585 | 25.0 | 21.3 | yes | | 11.1 |
| GeoLift | 10 | root | none | 193 | 25 | 22,399,608 | 78.6 | 54.6 | yes | | 29.3 |
| GeoLift | 10 | root | ascent | 194 | 25 | 22,399,608 | 78.6 | 54.6 | yes | | 29.4 |
| noise | 6 | root | swap | 78 | 3 | 1,890,588 | 11.2 | 0.0 | yes | | 4.9 |
| noise | 6 | easy | swap | 97 | 3 | 3,838,380 | 17.4 | 0.0 | yes | | 7.1 |
| noise | 6 | hard | swap | 78 | 3 | 1,890,588 | 12.0 | 0.0 | yes | | 5.2 |
| noise | 6 | pseudo | swap | 500 | 3 | 3,623,236 | 17.0 | 0.0 | yes | | 6.8 |
| noise | 6 | root | none | 78 | 3 | 1,890,588 | 11.2 | 4.0 | yes | | 4.2 |
| noise | 6 | root | ascent | 78 | 3 | 1,890,588 | 11.2 | 4.0 | yes | | 4.3 |
| noise | 8 | root | swap | 634 | 42 | 7,315,182 | 69.5 | 61.6 | yes | | 30.1 |
| noise | 8 | easy | swap | 15,885 | 11 | 6,008,145 | 67.1 | 24.9 | yes | | 32.9 |
| noise | 8 | hard | swap | 100 | 29 | 2,383,689 | 50.8 | 42.6 | yes | | 23.2 |
| noise | 8 | pseudo | swap | 1,814 | 31 | 1,369,838 | 38.0 | 28.0 | yes | | 16.5 |
| noise | 8 | root | none | 222 | 12 | 40,661,082 | 119.4 | – | no | 0.780% | 40.1 |
| noise | 8 | root | ascent | 224 | 11 | 41,799,021 | 119.3 | – | no | 0.780% | 41.2 |
| noise | 10 | root | swap | 256 | 76 | 78,059 | 67.3 | 55.8 | yes | | 30.8 |
| noise | 10 | easy | swap | 487 | 118 | 72,592 | 91.0 | 87.1 | yes | | 40.8 |
| noise | 10 | hard | swap | 105 | 70 | 627,369 | 74.5 | 60.0 | yes | | 33.6 |
| noise | 10 | pseudo | swap | 285 | 77 | 116,496 | 63.6 | 54.5 | yes | | 27.9 |
| noise | 10 | root | none | 217 | 27 | 34,258,578 | 119.2 | – | no | 1.354% | 42.9 |
| noise | 10 | root | ascent | 220 | 27 | 34,514,085 | 118.6 | – | no | 1.641% | 45.6 |
| four-factor | 6 | root | swap | 97 | 4 | 3,838,380 | 13.4 | 6.3 | yes | | 5.6 |
| four-factor | 6 | easy | swap | 97 | 4 | 3,838,380 | 13.2 | 5.0 | yes | | 5.0 |
| four-factor | 6 | hard | swap | 97 | 4 | 3,838,380 | 13.4 | 4.5 | yes | | 5.2 |
| four-factor | 6 | pseudo | swap | 96 | 4 | 3,838,380 | 13.7 | 8.4 | yes | | 5.3 |
| four-factor | 6 | root | none | 97 | 4 | 3,838,380 | 13.4 | 6.3 | yes | | 5.3 |
| four-factor | 6 | root | ascent | 97 | 4 | 3,838,380 | 13.4 | 6.3 | yes | | 5.2 |
| four-factor | 8 | root | swap | 234 | 16 | 45,254,925 | 118.4 | 12.1 | no | 0.228% | 40.6 |
| four-factor | 8 | easy | swap | 233 | 16 | 46,012,889 | 120.1 | 11.2 | no | 0.261% | 41.5 |
| four-factor | 8 | hard | swap | 232 | 16 | 45,793,771 | 120.0 | 15.3 | no | 0.249% | 39.7 |
| four-factor | 8 | pseudo | swap | 308 | 16 | 45,242,338 | 118.7 | 30.9 | no | 0.219% | 38.5 |
| four-factor | 8 | root | none | 233 | 16 | 45,333,849 | 118.7 | 39.9 | no | 0.257% | 39.8 |
| four-factor | 8 | root | ascent | 234 | 16 | 45,254,925 | 118.4 | 12.1 | no | 0.228% | 38.3 |
| four-factor | 10 | root | swap | 240 | 55 | 36,933,310 | 118.9 | 0.0 | no | 0.170% | 41.6 |
| four-factor | 10 | easy | swap | 43,563 | 44 | 17,551,569 | 220.7 | 0.0 | no | 0.098% | 110.1 |
| four-factor | 10 | hard | swap | 236 | 39 | 41,207,287 | 119.8 | 0.0 | no | 0.190% | 41.8 |
| four-factor | 10 | pseudo | swap | 1,139 | 119 | 24,584,799 | 121.7 | 0.0 | no | 0.110% | 47.7 |
| four-factor | 10 | root | none | 235 | 39 | 40,656,152 | 120.2 | – | no | 0.295% | 42.4 |
| four-factor | 10 | root | ascent | 236 | 38 | 40,888,917 | 119.5 | – | no | 0.295% | 43.3 |

Every run returns the optimum or, stopped, a design above it with a bound
below it; every certified run certifies the same design.

### Findings

1. Pseudo-costs do the least work where the bounds decide. Over the seven
   instances that certify, the work totals 291 modelled seconds under the
   library's rule, 200 under pseudo-costs (31% less), 230 under hard first and
   327 under easy first; on the clock 128, 89, 103 and 146 seconds. The gain
   is on the larger instances: GeoLift with ten treated takes 25.0 instead of
   59.6, noise with eight 38.0 instead of 69.5. On the two smallest, where
   most sets are priced anyway, the trials cost more than they save (noise
   with six: 17.0 against 11.2).
2. Pseudo-costs price far fewer sets: their trials send the search to the
   children whose bound rises most, so subtrees close on bounds instead of
   being priced (GeoLift ten: 123,585 sets against 11.4 million).
3. Hard first (BiqMac's difficult first) beats the fixed root order on
   GeoLift and on noise with eight treated, and loses on noise with six and
   ten.
   Easy first is the worst rule overall.
4. On the four-factor panel no rule certifies at eight or ten treated within
   the limit; pricing dominates (#844). Pseudo-costs leave the tightest
   bracket: 0.219% against 0.228% at eight, 0.110% against 0.170% at ten.
5. The swap search's seed decides whether noise certifies at all. Without it,
   noise with eight and ten treated stop at the limit 0.8 and 1.4 to 1.6%
   from their bound, and GeoLift with eight needs 90.9 instead of 54.5. The
   concave-ascent seed does no better than none on any instance.
6. The work limit does not cap the bracket. When the limit stops the loop,
   the library bounds every open node by its trust region, and certifies it
   where that helps, outside the limit, to report the bracket. Under the
   library's rule few nodes are open and the overshoot is at most a few
   modelled seconds (120.2 at most here). Easy first on the four-factor panel
   with ten treated leaves 43,563 nodes open and spends 220.7 modelled
   seconds against a limit of 120. The library's rule never leaves that many
   open, so this is a property of the limit to fix with any change of rule,
   beside the uncharged resolution recorded in #845.

## BiqCrunch's choices

SYNDES's lift is BiqCrunch's bounding procedure (Krislock, Malick and Roupin
2017, sec. 4.1), and its `X r = 0` for `r = [-s; 1]` is the cardinality
constraint with its product constraints `x_i (1'x) = s x_i` (their sec. 5.3,
after Faye and Roupin 2007). BiqCrunch's other choices, against this study:

- Branching (their sec. 4.3) reads the last column `x̂` of the bound's
  solution: least fractional, most fractional (the default), or closest to
  one. For MAREX these are `decided`, `fractional` and the library's
  `largest`, read from the persp+N indicator `z`. For SYNDES, `easy` and
  `hard` read the node's trust-region maximiser, not the lift's `x̂`; `hard`
  is the most-fractional rule on that point.
- Node order: the weakest bound first, ties to the deeper node, depth first
  near the bottom of the tree. That is SYNDES's order since #840 and MAREX's
  best first.
- The incumbent (their Algorithm 3): random-hyperplane rounding of the
  bound's factorization at every bounding iteration, then a 1-opt local
  search. This study did not measure it (#839's incumbent candidate 2); for a
  cardinality-constrained problem the rounding has to be cut to `r` positives
  before it is feasible.
- Pruning: BiqCrunch prunes when the bound is under the incumbent plus one,
  which needs an integer objective, in floating point. Both searches here take
  real data and discard a node only on a bound certified in exact arithmetic.

## MAREX

The branching market, read from the persp+N indicator `z` of the bounded node
(the treated weight where the bound returned none):

- `largest`, the library's: the free market with the largest `z`;
- `decided`: the largest `|2 z - 1|`, BiqMac's easy first and BiqCrunch's
  least fractional;
- `fractional`: the smallest `|2 z - 1|`, difficult first, BiqCrunch's default;
- `pseudo`: reliability branching as for SYNDES, a market's gains read when a
  child is bounded, an unreliable market tried on both children's cheap bound
  (the treated fit over the markets it may treat plus the control floor), each
  trial charged as two fits at 200 sets' price.

The guess: at each bounded node the library fits the `r` free markets with the
largest score as a candidate (`on`); `off` leaves the incumbent to the
subtrees the search prices. The order: `best`, the library's, or `depth`.

The study's panels with known optima: Walmart (45 markets, 104 periods) and
GeoLift (40 markets, 90 periods), the standard design. `work` is the library's
cost model, a millionth of a second per set priced and `1 + 3 (k/40)^3` per
persp+N bound; fits are not charged.

`gap` is (upper − lower) / upper of the bracket at stop, empty when the search
closes; `fitted` counts the designs fitted.

| panel | treated | rule | guess | order | bounds | sets priced | work | near | closed | gap | fitted | seconds |
|---|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| Walmart | 6 | largest | on | best | 2 | 1,086,008 | 11.6 | 5.3 | yes |  | 1 | 13.4 |
| Walmart | 6 | largest | off | best | 2 | 1,086,008 | 11.6 | 6.4 | yes |  | 4 | 13.2 |
| Walmart | 6 | decided | on | best | 3 | 8,145,060 | 24.0 | 5.3 | yes |  | 1 | 33.0 |
| Walmart | 6 | decided | off | best | 3 | 8,145,060 | 24.0 | 24.0 | yes |  | 7 | 32.4 |
| Walmart | 6 | fractional | on | best | 2 | 1,086,008 | 11.6 | 5.3 | yes |  | 1 | 14.1 |
| Walmart | 6 | fractional | off | best | 2 | 1,086,008 | 11.6 | 6.4 | yes |  | 6 | 14.0 |
| Walmart | 6 | pseudo | on | best | 2 | 1,086,008 | 11.6 | 5.3 | yes |  | 1 | 13.8 |
| Walmart | 6 | pseudo | off | best | 2 | 1,086,008 | 11.6 | 6.4 | yes |  | 4 | 13.9 |
| Walmart | 8 | largest | on | best | 22 | 33,257,562 | 145.3 | 37.2 | yes |  | 14 | 194.9 |
| Walmart | 8 | largest | off | best | 22 | 33,257,562 | 145.3 | 37.2 | yes |  | 12 | 188.7 |
| Walmart | 8 | decided | on | best | 81 | 188,669,697 | 598.1 | 157.7 | no | 3.40% | 18 | 821.8 |
| Walmart | 8 | decided | off | best | 81 | 188,669,697 | 598.1 | 157.7 | no | 3.40% | 21 | 859.9 |
| Walmart | 8 | fractional | on | best | 30 | 55,218,579 | 207.5 | 57.4 | yes |  | 12 | 282.6 |
| Walmart | 8 | fractional | off | best | 30 | 55,218,579 | 207.5 | 57.4 | yes |  | 10 | 293.5 |
| Walmart | 8 | pseudo | on | best | 22 | 33,257,562 | 145.4 | 37.2 | yes |  | 14 | 195.2 |
| Walmart | 8 | pseudo | off | best | 22 | 33,257,562 | 145.4 | 37.2 | yes |  | 12 | 198.6 |
| Walmart | 8 | largest | on | depth | 22 | 33,257,562 | 145.3 | 16.7 | yes |  | 15 | 202.3 |
| GeoLift | 8 | largest | on | best | 7 | 5,085,465 | 32.4 | 22.3 | yes |  | 10 | 35.6 |
| GeoLift | 8 | decided | on | best | 19 | 13,884,156 | 88.6 | 88.6 | yes |  | 18 | 88.9 |
| GeoLift | 8 | fractional | on | best | 13 | 9,358,041 | 59.8 | 30.1 | yes |  | 14 | 68.9 |
| GeoLift | 8 | pseudo | on | best | 7 | 5,085,465 | 32.4 | 22.3 | yes |  | 10 | 36.8 |
| GeoLift | 8 | largest | on | depth | 7 | 5,085,465 | 32.4 | 22.3 | yes |  | 10 | 37.8 |

### Findings

1. The library's rule is best or tied on every instance. Most fractional,
   BiqCrunch's default, does 43% more work on Walmart with eight treated
   (207.5 against 145.3) and 85% more on GeoLift with eight (59.8 against
   32.4). Least fractional is worse still: it doubles Walmart with six,
   nearly triples GeoLift with eight (88.6), and on Walmart with eight stops
   at the work limit after 81 bounds, 3.4% from its bound, where the library
   certifies after 22.
2. Pseudo-costs never leave the library's order. A MAREX search makes 2 to 22
   bounds, too few for any market to gather eight gains a side, and the trials
   on the cheap bound pick the market the library picks. Their work is the
   library's plus the trials' charge.
3. The guess changes no bound and no work: the same bounds, sets priced and
   modelled seconds with it off. It finds the near-optimal design sooner on
   Walmart with six (5.3 against 6.4 under the library's rule, 5.3 against
   24.0 under least fractional) and no later anywhere.
4. Depth first does the same bounds and the same work as best first on both
   instances it ran (7 and 22 bounds). On Walmart with eight treated it
   reaches the near-optimal design sooner, 16.7 modelled seconds against
   37.2. The order changed when the design was found, not the bounds or the
   work.



## Recommendations

SYNDES two-way:

- Branching: reliability branching on pseudo-costs, trials on the trust-region
  bound. It cut the work by 31% over the instances that certify, by up to 2.4
  times on GeoLift with ten treated, and left the tightest bracket where the
  search stopped. It lost on the two smallest instances, where most sets are
  priced and the trials cost more than they save; a build should measure a
  rule that skips trials on such nodes. Filed as #881.
- Node order: keep best first with dives (#840).
- Incumbent: keep the swap search. Without it noise does not certify at eight
  or ten treated; the concave ascent adds nothing. BiqCrunch's hyperplane
  rounding with 1-opt was not measured.
- The work limit: the bracket computed after the limit stops the loop is
  outside the limit (finding 6). Any change that leaves more nodes open, a
  rule or an order, needs it charged or capped first. Recorded on #843, beside
  the uncharged resolution of #845.

MAREX: keep the library's choices (largest indicator, the guess, best first).
Nothing to build.
