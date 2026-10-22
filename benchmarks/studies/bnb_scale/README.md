# The certified branch and bounds at 60, 80 and 100 units (#842)

Every measurement before this study was at 40 to 45 units. #842 asks what
SYNDES's two-way search and MAREX's search cost at 60, 80 and 100 units, where
the time goes, and whether the default work limit still certifies. This study
runs both searches as they stand on `main`, wrapped from outside so they make
the same decisions, and adds the counts they do not report.

This is a study, not a benchmark case. Nothing here is pinned, and the seconds
are machine-dependent. Every run is single-threaded. The runs reported here set
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, and the
instruments also hold BLAS to one thread themselves, so a caller's setting
cannot change what they measure. Every run records the load average before and
after, as #840 did. The modelled work, the node counts and the bounds
are deterministic. Reruns of the same panel reproduced them exactly, while CPU
seconds moved by a factor of 1.2 to 1.3 with the load.

## The panels (`panels.py`)

- `bls`: the BLS county unemployment panel of the spillover benchmarks. It has
  193 counties of Alabama, Oregon, Pennsylvania and Wyoming over `T = 36`
  months, with a common cycle, seasonality and state structure. Counties are
  taken in one fixed shuffle, so each size holds the smaller ones and all four
  states. GeoLift (40 markets) and Walmart (45 stores) do not reach 60.
- `noise`: a common shock and nothing else (`common_shock` of
  `syndes_global_bnb/quality.py`), `T = 30`.
- `flat`: that study's four-factor generator (`factor`), on which bounds prune
  least, `T = 30`.

SYNDES runs at `K = N/2`, the hardest size.

## SYNDES: the search at the default limit (`syndes_scale.py`)

Each row is one search at the default `work_limit` of 60 modelled seconds. The
gap is `(design - certified bound) / design`. The trust-region column is the
root's own gap, for comparison.

| panel | N | nodes | lifts | pruned | sets priced | modelled s | CPU s | CPU per modelled s | largest queue | gap at the limit | root trust-region gap | load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bls | 40 | 163 | 157 | 76 | 983,212 | 9.3 | 28.8 | 3.11 | 38 | certified | 1.18% | 1.0 |
| noise | 40 | 404 | 396 | 198 | 335,920 | 21.2 | 65.2 | 3.07 | 142 | certified | 3.21% | 1.1 to 5.6 |
| flat | 40 | 415 | 262 | 0 | 49,493,290 | 59.9 | 111.2 | 1.86 | 363 | 0.14% | 0.24% | 1.1 to 6.7 |
| bls | 60 | 628 | 628 | 40 | 0 | 60.7 | 195.0 | 3.21 | 549 | 0.71% | 0.97% | 1.0 |
| noise | 60 | 521 | 521 | 0 | 0 | 60.6 | 98.8 | 1.63 | 522 | 1.46% | 1.46% | 1.0 |
| flat | 60 | 1,017 | 1,017 | 4 | 0 | 69.1 | 119.8 | 1.73 | 1,010 | 0.31% | 0.31% | 1.0 |
| bls | 80 | 393 | 393 | 8 | 0 | 61.5 | 127.5 | 2.07 | 378 | 0.52% | | 1.1 to 7.3 |
| noise | 80 | 220 | 220 | 0 | 0 | 60.9 | 119.3 | 1.96 | 221 | 0.92% | | 5.1 to 7.5 |
| flat | 80 | 459 | 459 | 0 | 0 | 67.0 | 131.8 | 1.97 | 460 | 0.18% | | 4.7 to 5.7 |
| bls | 100 | 90 | 90 | 0 | 0 | 63.6 | 72.7 | 1.14 | 91 | 0.37% | 0.37% | 0.6 to 0.9 |
| noise | 100 | 107 | 107 | 0 | 0 | 62.5 | 83.3 | 1.33 | 108 | 0.59% | 0.59% | 1.0 |
| flat | 100 | 249 | 249 | 0 | 0 | 67.4 | 77.4 | 1.15 | 250 | 0.12% | 0.12% | 1.0 |

Ten times the limit (600 modelled seconds):

| panel | N | nodes | pruned | CPU s | largest queue | gap at the limit |
|---|---:|---:|---:|---:|---:|---:|
| bls | 60 | 7,375 | 556 | 1,650.7 | 6,264 | 0.62% |
| noise | 60 | 5,392 | 0 | 1,778.3 | 5,393 | 1.41% |
| bls | 100 | 1,227 | 4 | 997.2 | 1,220 | 0.37% |

- The default limit certifies the structured and structureless panels at 40
  units and nothing above. The flat panel already misses at 40.
- From 60 units up, the search prices no sets and prunes almost no nodes. The
  best design it reports was held by the swap search's seeds, or their
  complements, in every run from 60 to 100 units. At 40 units the search found
  a better design than the seeds on all three panels. Above 40 units the search
  works only on the bound.
- On noise and flat at 60 units and on all three panels at 100, the certified
  gap at the limit equals the root's trust-region gap. At 100 BLS units, 1,227
  lifted nodes left it at 0.3722%, the root's gap to the digit.
- Best first's queue holds about one node per node bounded, since almost
  nothing prunes: 6,264 after 7,375 nodes on BLS at 60 units.

## Why the bound stops at the trust region (`root_gap.py`)

`H = (G + lam I)^-1 = I/lam - U D U'`, with `U` the panel's basis of at most
`T` columns, so

    R = alpha H - p p' = (alpha/lam) I - W,     W PSD, rank(W) <= T + 1.

`R 1 = 0` gives `W 1 = (alpha/lam) 1`. With `W' = W - (alpha/(lam N)) 11'`,
which is PSD with `W' 1 = 0` and rank at most `T`, every balanced design has

    sigma'R sigma = (alpha/lam) N - sigma'W' sigma.

The two-way search maximises `sigma'R sigma`, so it minimises the residual
`sigma'W' sigma` of a balancing problem of rank `T`. On all nine panels,
`lambda_max(R) = alpha/lam` to 12 digits, with multiplicity exactly
`N - T - 1`. The root's trust-region bound equals the eigenvalue bound
`N alpha/lam` to 1e-15, so it certifies `sigma'W' sigma >= 0` and nothing more.

`root_bounds` puts four bounds at the root beside the best design known:

- the trust region;
- the lift as the search runs it;
- the lift with a finer stop (`LONG`: `alpha` down to 1e-8, gradients to 1e-5);
- Shor's relaxation with the cardinality, its products and the RLT products,
  solved by interior point (`interior`).

`triangle_closure` solves the lift's whole relaxation by interior point,
separating triangles until none is violated by more than 1e-6. That is the
value the lift's dual converges to.

The partition papers read for #837 report gaps relative to the cut value, here
the best residual `w*`. On that scale, `(w* - w_L)/w*`:

| panel | N | N − T − 1 | trust region, design scale | Shor, products, RLT | lift | triangle closure |
|---|---:|---:|---:|---:|---:|---:|
| bls | 40 | 3 | 1.18% | 61.2% | 16.9% | 12.4% |
| noise | 40 | 9 | 3.21% | 92.3% | 48.0% | 47.4% |
| flat | 40 | 9 | 0.24% | 95.2% | 76.0% | 68.5% |
| bls | 60 | 23 | 0.97% | 93.3% | 72.8% | 66.0% |
| noise, flat | 60 | 29 | 1.46%, 0.31% | 100% | 100% | 100% |
| bls, noise, flat | 100 | 63 to 69 | 0.37%, 0.59%, 0.12% | 100% | 100% | 100% |

The design-scale column is the trust region's gap as the search reports it. It
equals `w*/(N alpha/lam)`, the residual's share of the constant.

- Near `N = T` the relaxation is close to the range the partition papers
  report: Wolkowicz and Zhao's two-part gaps run from 1.7% to 7.6%.
- At `N - T - 1 = 29` and above, every relaxation tested equals the trust
  region to within 2e-9, the interior-point solver's accuracy. The closure loop
  found 0 to 169 violated triangles there, and they did not move the bound.
  BLS at 60 units (`N - T - 1 = 23`) still gains, so on these panels the
  threshold lies between 23 and 29.
- At 600 modelled seconds the noise panel at 60 units moved from 1.46% to
  1.41%. Fixed units do not break the degeneracy: their linear term misses
  the top eigenspace, and only the free count ends it (see "How deep a node
  must be" below, #858).

## Where the lift stops (`root_gap.trace_lift`)

`trace_lift` runs one root lift and records each round: its `alpha`, the
smoothed value at the round's solution, and whether it added triangles. It
applies the lift's own stopping tests to each round:

- at the floor: `alpha` at `ALPHA_MIN` and no new triangle;
- stalled: the smoothed value moved by at most 1e-6 relative, with `alpha`
  under 1e-4;
- give up: at least `MIN_ROUNDS` rounds, and `MAX_ROUNDS` reached or stalled.

The tests hold the replica to the lift. Exactly the last round meets a rule,
on fixed panels and as a `hypothesis` property across panels, sizes and
schedules. The rules themselves are also checked on rounds written by hand,
whose answer does not depend on the machine. Three schedules ran: the shipped
one, `LONG`, and `NO_STALL` (`LONG` with the stall rule off).

The rounds a lift takes depend on the BLAS kernel's last digits, so the table
is this machine's. On a CI runner the shipped 40-unit run took 824 evaluations,
not 761, and the `LONG` schedule ran 6,982 where this machine stopped it at
702.

| panel | N | schedule | evaluations | final alpha | ended by | root gap | closure |
|---|---:|---|---:|---:|---|---:|---:|
| bls | 40 | shipped | 761 | 5e-5 | floor | 0.2013% | 0.1477% |
| bls | 40 | LONG | 702 | 4.9e-5 | stall | 0.2021% | |
| bls | 40 | NO_STALL | 7,835 | 1e-8 | floor | 0.1520% | |
| bls | 60 | shipped | 601 | 5e-5 | floor | 0.7109% | 0.6451% |
| bls | 60 | LONG | 6,832 | 2.4e-8 | stall | 0.6465% | |
| bls | 60 | NO_STALL | 9,105 | 1e-8 | floor | 0.6465% | |
| noise | 40 | shipped | 291 | 5e-5 | floor | 1.5660% | 1.5469% |
| noise | 40 | LONG | 1,478 | 4.8e-8 | stall | 1.5480% | |
| flat | 40 | shipped | 180 | 5e-5 | stall | 0.1814% | 0.1635% |
| flat | 40 | LONG | 2,152 | 4.8e-8 | stall | 0.1637% | |
| bls | 100 | all three | 267 to 1,654 | | | 0.3722% | 0.3722% |

- Where the relaxation has room, a lower `alpha` floor recovers it. The finer
  stop reaches the closure to within 0.002 points on BLS at 60 units, noise at
  40 and flat at 40, for 5 to 12 times the evaluations.
- On this machine, on BLS at 40 units, the lower floor alone did not help. The
  stall rule ended the run at `alpha = 4.9e-5`, 59 evaluations sooner than the
  shipped run and on a certificate 8e-6 weaker. The stall test sat on a
  knife-edge there, a relative change of 9e-7 against its 1e-6, and on the CI
  runner above the same schedule ran on. With the stall rule off the lift
  reaches 0.1520%, 92% of the way to the closure, for ten times the
  evaluations (3.2 CPU seconds).
- So the slack at these roots is the `alpha` floor, the share Malick and
  Roupin's price `alpha (n+1)^2 / 2` predicts. The stall rule, which tests
  whether the dual has converged at the current `alpha`, can end the run before
  a lower floor is reached, and whether it does depends on the last digits.
  This is #837's first candidate.

## How deep a node must be (`depth.py`, #858)

The collapse above is at the root. #858 asks whether fixing units breaks it,
and how deep a node must be before its bound passes the trust region.
`depth.depth_table` bounds nodes on the path to each panel's design: `fixed`
units decided, half of them drawn from the design into the treated set and the
rest from its complement out of it, three nodes per count (seed 0). Every lift
here has its dual scaled by the node's free part (#864). The shipped lift took
its scale from the fixed units' constant too, and stopped at the trust region
at every deep node; that is how #864 was found, and the wrong turns below say
how it first read as a collapse.

The structure. At a node with free set `F` and `n_F` free units, the free
block of `R = (alpha/lam) I - W` keeps the top eigenvalue `alpha/lam` on the
plane `1'x = s`, on the null space of `W_FF` there, of dimension
`n_F - T - 1`, the node's excess. The fixed units add the linear term
`b = R_FD sigma_D = -W_FD sigma_D`, which lies in `W`'s row space, and that
eigenspace avoids it. Measured at the 144 nodes with positive excess: the
multiplicity equals the excess exactly at every one, and the linear term's
share in the eigenspace (`reach`) is at most 9e-14. Fixing a unit lowers the
excess by one and leaves the degeneracy otherwise as it was. Only the free
count can end it.

Where the lift gains nothing. Each panel's last excess at which all three
nodes gained nothing, and the first at which all three gained. The closure is
the triangle closure by interior point at the first node of a count:

| panel | units | T | root | zero at every node down to excess | gain at every node from excess | closure there |
|---|---:|---:|---|---:|---:|---|
| BLS | 60 | 36 | gains (2.6e-3) | none | 23 (the root) | |
| BLS | 80 | 36 | zero | 43 (the root) | 38 (5 fixed) | 2.8e-3 at 23 |
| BLS | 100 | 36 | zero | 53 (10 fixed) | 33 (30 fixed) | 1.3e-4 at 43 |
| noise | 60 | 30 | zero | 24 (5 fixed) | 9 (20 fixed) | 8.9e-3 at 9 |
| noise | 80 | 30 | zero | 29 (20 fixed) | 19 (30 fixed) | 6.8e-8 at 29 |
| noise | 100 | 30 | zero | 29 (40 fixed) | 19 (50 fixed) | 2.0e-5 at 29 |
| flat | 60 | 30 | zero | 19 (10 fixed) | 9 (20 fixed) | 8.5e-4 at 9 |
| flat | 80 | 30 | zero | 29 (20 fixed) | 9 (40 fixed) | 2.1e-9 at 29 |
| flat | 100 | 30 | zero | 29 (40 fixed) | 9 (60 fixed) | 3.0e-9 at 29 |

Between the two columns some nodes gain and some do not (noise and flat at an
excess of 19, BLS at 100 units at 43). Every row is in `results/depth.jsonl`.

- On the structureless panels, at an excess of 29 and above, the lift gains
  nothing and the closure at most 2e-5. At 9 both have room. The threshold
  lies between 19 and 29, where the nodes differ.
- On BLS the threshold sits higher and moves with the panel's size: at 80
  units the root collapses and nodes with five units fixed (excess 38) gain,
  while at 100 units nodes gain nothing down to an excess of 53.
- A collapsed lift is not free: 0.39 s on average at the 77 nodes where it
  gained nothing, against 0.50 s at the 130 where it gained.

The certificate. Where the trust region is in its hard case, `depth.certificate`
writes a point of the lift's relaxation at the trust region's value, by
fitting an ellipsoid through the rows that span the top eigenspace
(Saunderson, Chandrasekaran, Parrilo and Willsky 2012). It certified 25 of the
77 zero-gain nodes and no node that gained: the roots and the nodes with up to
10 or 20 units fixed of noise and flat at 80 and 100 units, the nodes best
first visits there. It certified no BLS node and nothing at 60 units. It costs
3 ms on average and 25 ms at most, against 0.39 s for a lift that gains
nothing.

Where the least-norm fit is not semidefinite, a semidefinite fit may still
exist. The projected fit (Dykstra's alternating projections) found one at 25
of the 29 such nodes that gained nothing, all 10 BLS nodes among them, and at
6 of the 46 that gained. Every fit it found violated a triangle, so it
certified nothing:

- the semidefinite relaxation collapses at those nodes, BLS included, and the
  triangles are what a fitted point cannot match;
- the six gaining nodes with a fit gain through their triangles;
- the projections cost a median 0.52 s (0.02 to 2.4 s), against 0.56 s for
  the lifts at those nodes, so a better fit is no shortcut past the lift.

## Inside one lift (`syndes_scale.profile_lift`)

One root lift on BLS, profiled on a quiet machine (load 0.2). The shares are
read on the profiler's clock and are disjoint: the dual's evaluation (with its
eigendecomposition), triangle separation, and SciPy's L-BFGS-B by own time.

| N | evaluations | CPU s | dual evaluation | of which eigendecomposition | separation | SciPy L-BFGS-B |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 761 | 0.33 | 64.8% | 31.4% | 2.1% | 19.2% |
| 60 | 601 | 0.42 | 64.1% | 37.9% | 9.5% | 15.2% |
| 80 | 380 | 0.55 | 59.6% | 39.9% | 20.4% | 11.2% |
| 100 | 267 | 0.48 | 59.2% | 44.1% | 28.5% | 5.4% |

- A lift costs under a second at every size, because the lift stops sooner as
  `N` grows (761 evaluations at 40 units, 267 at 100).
- Separation grows to 28.5% at 100 units. `separate` enumerates every triple of
  the `n + 1` indices each round, which is the cheaper target for #844.
- SciPy's L-BFGS-B is 5% of a lift at 100 units, so a faster solver is not the
  lever. The comparison with LBFGSpp is in #837.
- With BLAS unpinned on a loaded machine, a lift was 44 times slower at 60
  units (#857).

## MAREX (`marex_scale.py`)

MAREX runs on the BLS panel as its market panel, with `m = 6` treated.

One root bound, the persp+N relaxation solved by Clarabel:

| markets | seconds | modelled seconds | measured over modelled | load |
|---:|---:|---:|---:|---|
| 45 | 7.1 | 5.3 | 1.35 | 0.1 to 0.2 |
| 60 | 44.4 | 11.1 | 4.0 | 0.2 to 1.0 |
| 80 | 224.7 | 25.0 | 9.0 | 1.1 to 6.6 |
| 100 | 787.6 | 47.9 | 16.5 | 3.5 to 6.6 |

The search, capped at 30 to 40 bounds (#843 has since moved the cap to the
cost model's seconds; every run here closed, so the rows stand):

| markets | closed | bounds | sets priced | CPU s | in bounds | slowest bound | load |
|---:|---|---:|---:|---:|---:|---:|---|
| 45 | yes | 3 | 8,145,060 | 42.0 | 23.4 | 8.4 | 3.0 |
| 60 | yes | 6 | 21,074,185 | 308.1 | 253.3 | 50.4 | 2.6 to 2.9 |
| 80 | yes | 11 | 169,384,215 | 2,466.0 | 1,971.1 | 213.7 | 2.3 to 3.0 |

- MAREX closes at 45, 60 and 80 markets with a few bounds. The bounds take 56% to 82% of
  the search, and a single bound passes a minute between 60 and 80 markets.
  The 80- and 100-market bounds ran under load, so their seconds are upper
  values.
- The cost model under-costs the bound by 1.35 times at 45 markets and by 4
  times at 60, on a quiet machine. MAREX's lever is the bound itself, which is
  #833.

## Answers to #842

- Where the time goes at 100 units: neither the node count nor the cost per
  node. A lift costs about half a second, but on these panels no relaxation in
  the lift's family certifies past the eigenvalue bound when `N - T - 1` is 29
  or more, so every lift there is spent without moving the bound. This is
  #858.
- Near `N = T` (40 units, and BLS at 60) the bound decides, and the lift's
  stopping leaves part of what its own relaxation reaches. This is #837.
- The default work limit certifies the structured and structureless panels at
  40 units and nothing above.
- Best first's queue grows by about one node per node bounded: 549 at the
  default limit and 6,264 at ten times it, on BLS at 60 units.
- MAREX: a single bound passes a minute between 60 and 80 markets.

## Answers to #858

- How deep a node must be: until its excess `n_F - T - 1` falls below a
  threshold that depends on the panel, between 19 and 29 on the
  structureless panels and between 38 and 53 on BLS. Fixing units moves the
  bound only through the free count.
- Where the threshold sits: no free count serves all three panels. To skip
  no node that gains here, a rule on the excess must stop above 38, where BLS
  at 80 units gains with five units fixed. That rule skips the shallow nodes
  of noise and flat at 80 and 100 units and nothing at 60, and it rests on
  three panels.
- The certificate skips exactly the nodes it proves: 25 of the 77 that
  gained nothing, at 3 ms each, with no gain lost, and none on BLS.
- On #864's lift the default limit narrows BLS at 80 units from 0.5225% to
  0.4712%; at 100 units it stays at the root's 0.3722%, every node it reaches
  inside the collapse.

## Recommendation

In the order of the roadmap (#848) as the author revised it after this study:

1. #857: hold BLAS to one thread inside the lift. Unpinned, a root lift took
   16 s at 60 units against 0.8 s on one thread on a quiet machine. Under
   parallel load SciPy's L-BFGS-B core alone took 10.7 s of an 11.8 s lift at
   20 units.
2. #858, measured (above): build the certificate's skip together with #839,
   not before. Skipping a lift the certificate proves futile is exact and
   costs 3 ms, but at a work limit the search spends what it saves on more
   collapsed nodes and returns the same gap and design. The saving pays once
   it goes to the incumbent (#839), as #858 proposed. A rule on the free
   count is not recommended: its threshold moved from 19 to 53 across three
   panels.
3. #837: run the lift's dual past the stall rule with a lower floor, at nodes
   within a margin of their prune target. It is the one change measured here
   that moves a root bound: 0.711% to 0.6465% on BLS at 60 units, for 11 times
   the evaluations.
4. #844: incremental or restricted triangle separation (28.5% of a lift at 100
   units), and the eigendecomposition (44%). Not the L-BFGS-B solver.
5. #845: the cost model under-costs SYNDES's search by 1.1 to 3.2 times in CPU
   per modelled second, and MAREX's bound by 1.35 to 16.5 times.
6. #833 for MAREX, whose cost is its bound.

## Wrong turns

- The first lift profile divided the profiler's seconds by `process_time`,
  which sums CPU over BLAS threads, and reported the dual's evaluation as a few
  percent of a lift. Every share is now read on the profiler's clock, and a
  test refuses a share under 25%.
- The finer stop was expected never to lose bound. On BLS at 40 units it ended
  sooner on a weaker certificate, because the stall rule ended it. Malick and
  Roupin's Table 3 has the same reversal across `alpha`. The test now checks the
  floor every lift keeps, the trust region.
- The first reading of the 40-unit trace put the slack on the stall rule. The
  shipped run ends at its floor. The stall rule only blocks a lower floor.
- `schedule` restored only the constants of `LONG`, so turning the stall rule
  off would have left `MIN_ROUNDS` changed for the rest of the process, and a
  misspelt name would have been set beside the real one. It now restores every
  name it sets and refuses unknown ones.
- Two tests pinned the 40-unit trace as this machine ran it: a finer floor
  stalling above itself, and fewer evaluations than the shipped run. A CI
  runner's BLAS kernel took another path and both failed. The rules they
  guarded are now checked on rounds written by hand.
- In CI the profile read the dual's evaluation as 8 to 10% of a lift. The cause
  was SciPy's L-BFGS-B core under unpinned BLAS while other test workers ran
  threaded BLAS: 10.7 s of an 11.8 s lift at 20 units, about 7 ms a call
  against 0.03 ms. Two first explanations measured out wrong:
  - CPU load alone: three or four busy processes left the share at 0.63 to
    0.65;
  - garbage collection over a large heap: three million live objects left it
    at 0.61.

  Four xdist workers beside the SYNDES and MAREX tests reproduced it. The
  instruments now hold BLAS to one thread themselves.
- The 16-unit test compared the search's design with the enumerated one as
  sets. At K = N/2 a set and its complement are one design, and a CI runner
  returned the complement. The test accepts either.

- The first depth measurement found the lift gaining nothing on BLS at nodes
  with up to 40 units fixed at 80 units and up to 60 at 100, and read it as
  the collapse reaching deep into the tree. The closure by interior point at the same
  nodes gained 0.2% to 0.5%. The lift scaled its dual by the fixed units'
  constant, which grows with the units fixed; #864 fixed it, and every depth
  number here is on the fix.
- The certificate's first form held the relaxation's mean at the trust
  region's range part. At many nodes that mean left the box (`|x_i| > 1`).
  The top eigenspace lets the mean move without changing the value, and the
  homogeneous fit (`G_00 = 1`) uses that.
- The ellipsoid-fitting threshold for random points, about `m^2 / 4` points
  in dimension `m`, puts collapse near an excess of 13 to 15 at these sizes.
  The panels need 19 to 53: their rows are not random points.

## Running it

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -m benchmarks.studies.bnb_scale.run syndes --kinds bls noise flat --sizes 40 60 80 100
    python -m benchmarks.studies.bnb_scale.run root --kinds bls noise flat --sizes 40 60 100 --closure
    python -m benchmarks.studies.bnb_scale.run trace --kinds bls --sizes 40 60 100
    python -m benchmarks.studies.bnb_scale.run profile --kinds bls --sizes 40 60 80 100
    python -m benchmarks.studies.bnb_scale.run marex-bound --sizes 45 60 80 100
    python -m benchmarks.studies.bnb_scale.run marex-search --sizes 45 60 --m 6
    python -m benchmarks.studies.bnb_scale.run depth --kinds bls noise flat --sizes 60 80 100 --closure-at 20 40 --projection
    python -m pytest benchmarks/studies/bnb_scale/test_scale.py benchmarks/studies/bnb_scale/test_depth.py -q

Each run appends to `results/<what>.jsonl`. The `.log` files are the console
output of the runs reported above.
