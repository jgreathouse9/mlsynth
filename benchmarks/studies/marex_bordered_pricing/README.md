# MAREX's exact search: pricing candidates from a prefix

Issue #769 asks whether the closed-form bound that prices every candidate
treated set can be computed by updating a parent's factorisation, not from
scratch, and how much that saves. This study measures it at two to six treated
markets on GeoLift (40 markets) and on an 80-market panel, runs the GeoLift
searches end to end with it, and measures the subtree bound of Furnival and
Wilson (1974) on the same searches.

Bordering a prefix's inverse prices a candidate 2 to 7 times faster than the
shipped solve, returns the same bounds to 6e-11 relative, and leaves every
design unchanged. A form of the bound due to Neumaier and Shcherbina (2004)
makes its validity independent of how accurately the update was computed. The
subtree bound skips up to 60% of the candidates at six treated markets, but
computed node by node it costs more than it saves.

This is a study, not a benchmark case, and it changes nothing in the library.
`bordered.py` is a prototype; `search.py` swaps it in from the outside. Timings
are machine-dependent and nothing here is pinned.

```
python benchmarks/studies/marex_bordered_pricing/search.py
python benchmarks/studies/marex_bordered_pricing/pricing.py
python benchmarks/studies/marex_bordered_pricing/subtree.py
```

Run them from the repository root, `search.py` first: `subtree.py` reads the
optima it records.

## What is here

| file | what it does |
| --- | --- |
| `panel.py` | GeoLift as one MAREX cluster, and the 80-market synthetic panel |
| `bordered.py` | the prototype: bordering, the safe form, and its a priori rounding bound |
| `pricing.py` | pricing time per candidate, shipped against bordered, `m` = 2 to 6, `n` = 40 and 80 |
| `search.py` | the GeoLift searches end to end with each pricing |
| `subtree.py` | the subtree bound's skipped candidates on GeoLift's standard design |
| `results/*.json` | the run behind the tables below |

## The bound and the update

On the centred columns, with Gram matrix `G`, the search bounds a candidate's
treated fit below by `1 / s`, `s = 1'G_S^-1 1`, or by
`(2 + t)^2 / (4 s) - u / 4` with a penalty `a` (`t = 1'G_S^-1 a`,
`u = a'G_S^-1 a`). The shipped code solves an `m`-by-`m` system per candidate.

For `S = P + {j}` the Banachiewicz inversion formula (Zhang 2005, (0.7.2))
extends all three from the prefix: with `H = G_P^-1`, `g = G[P, j]`,
`z = H g` and the Schur complement `sigma = G_jj - g'z`,

```
e'G_S^-1 f = e_P'H f_P + (e_j - z'e_P)(f_j - z'f_P) / sigma,
```

`O(m^2)` per candidate. The candidates come in lexicographic order, so those
sharing a prefix are contiguous and its inverse is formed once for all of them.
A prefix tree only ever adds a market, so no downdate is needed, the operation
for which Golub and Van Loan (2013, sec. 6.5.4) need "a different set of tools
and a new set of numerical concerns".

## Validity in floating point

The issue requires the updated bound to stay a valid lower bound in floating
point. The textbooks do not analyse the accuracy of an updated factorisation:
Higham (2002) omits updating and downdating, and Golub and Van Loan (2013,
sec. 6.5) compare updating with recomputing only for cost. Miller (2002,
sec. 2.3) reports that methods built on sums-of-squares-and-products matrices,
as bordering an inverse is, lose about twice as many digits as orthogonal
ones.

Neumaier and Shcherbina (2004, sec. 3) make the question moot. Their bounds
charge an approximate dual point's residual against bounds on the variables,
and the same move applies here with the simplex as the bounds: for any vector
`v`, `||B~_S (w - v)||^2 >= 0` gives

```
g(S) >= -v'G_S v + min over i of (2 (G_S v)_i + a_i),
```

with equality at the affine minimiser, where it is the closed form. The bound
holds for every `v`, so the update's accuracy cannot make it invalid; only the
arithmetic of this last evaluation can. Higham's inner-product bounds (2002,
(3.5) and (3.13)), counting the products that form `G`, bound that arithmetic a
priori, and the bound is at most 3e-4 of the search's margin in every case
below.

## Pricing time per candidate

Microseconds per candidate, single-threaded, shipped, bordered, and bordered
followed by the safe evaluation:

| panel | `m` | candidates | shipped | bordered | bordered and safe |
| --- | ---: | ---: | ---: | ---: | ---: |
| GeoLift | 2 | 780 | 0.91 | 0.32 | 0.34 |
| GeoLift | 3 | 9,880 | 1.32 | 0.20 | 0.65 |
| GeoLift | 4 | 91,390 | 1.05 | 0.31 | 0.75 |
| GeoLift | 5 | 658,008 | 0.92 | 0.34 | 0.61 |
| GeoLift | 6 | 3,838,380 | 1.03 | 0.50 | 0.83 |
| synthetic | 2 | 3,160 | 0.68 | 0.13 | 0.17 |
| synthetic | 3 | 82,160 | 0.52 | 0.20 | 0.31 |
| synthetic | 4 | 1,581,580 | 0.67 | 0.25 | 0.43 |
| synthetic | 5 | 24,040,016 | 0.85 | 0.30 | 0.53 |
| synthetic | 6 | 300,500,200 | 0.96 | 0.37 | 0.67 |

Every case prices every candidate except the last, which prices 4 million: the
prefixes are drawn with probability proportional to their number of extensions
and all their extensions are priced, which keeps the cost per candidate
unbiased.

Bordering is 2.1 to 6.8 times faster in this run; an earlier run of the same
algorithm gave 2.0 to 4.0, and the per-candidate times move between runs by about
that much. The safe evaluation brings the gain to 1.2 to 3.9. The bordered and
shipped bounds agree to 6e-11 relative at worst, and neither sets a bound to 0
anywhere.

## The searches end to end

| search | shipped | bordered | design |
| --- | ---: | ---: | --- |
| standard, `m` = 4 | 0.11 s | 0.05 s | identical |
| standard, `m` = 5 | 0.71 s | 0.33 s | identical |
| standard, `m` = 6 | 4.60 s | 2.67 s | identical |
| weakly-targeted, `beta` = 1, `m` = 4 | 0.73 s | 0.67 s | identical |

The standard design's time is mostly pricing; the weakly-targeted design's is
mostly fits, so pricing moves it little.

## The subtree bound

Furnival and Wilson (1974) and Miller (2002, sec. 3.8) bound a whole family of
subsets by the fit with every remaining variable included. Transposed: order the
markets, take a prefix `P` and the markets `R` after its last, and every
candidate `S = P + Q` with `Q` in `R` has

```
f(S) >= lb(P + R) + h(all - P),
```

since the affine treated bound only falls as markets are added (the bordering
formula shows `s` only grows) and a control fit only falls as donors are added.
A prefix whose bound clears the cutoff skips all its candidates.

Share of the standard design's candidates skipped with the cutoff at the
optimum, by market order:

| `m` | as stored | nearest the target first | farthest first |
| ---: | ---: | ---: | ---: |
| 4 | 19.6% | 13.4% | 22.5% |
| 5 | 33.7% | 31.2% | 51.1% |
| 6 | 43.1% | 37.8% | 60.1% |

Adding the control term `h(all - P)` raises the share by at most 1.4 points at
`m = 4` and 5, for a simplex fit per prefix.

It does not pay as measured. Each prefix's bound here is a fresh solve on up to
40 markets: at `m = 6`, farthest first, 298,756 of them take 10.4 s, against
about 2 s to price all 3,838,380 candidates by bordering. Furnival and Wilson
make the node bounds cheap with a second tree that reuses its parent's
elimination; that is not measured here. Even with free node bounds, skipping
60% caps the gain at 2.5 times. The search would also need an incumbent before
pricing, and it prices every candidate first today. Miller (2002) reports
leaps and bounds feasible up to about 30 predictors, and the bound is 0 once
`P + R` holds more than `T` markets, since their affine hull is then the whole
space.

## What this means for the candidate limit

Ten treated among 40 markets is about 8.5e8 candidates, above the backend's
limit of 20 million. Bordering cuts the cost per candidate by 2 to 3 times at
equal validity, which moves the limit by about that much at equal time, not by
40 times. The measurements stop at six treated markets.

## Sources

* Zhang, F. (ed.) (2005). *The Schur Complement and Its Applications*.
  Springer. The Banachiewicz inversion formula (0.7.2); Theorem 1.12, a positive
  definite matrix has positive definite Schur complements, so a non-positive
  `sigma` in floating point marks a breakdown.
* Neumaier, A. and Shcherbina, O. (2004). Safe bounds in linear and
  mixed-integer linear programming. *Mathematical Programming* 99, 283-296.
  The safe form.
* Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd
  ed. SIAM. Inner-product and matrix-product rounding bounds, (3.5) and (3.13).
* Golub, G. H. and Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. Johns
  Hopkins. Algorithm 4.2.1, whose step `j` is one bordering step; sec. 6.5 on
  updating and downdating.
* Furnival, G. M. and Wilson, R. W. (1974). Regressions by leaps and bounds.
  *Technometrics* 16, 499-511. The subtree bound and its bound tree.
* Miller, A. (2002). *Subset Selection in Regression*, 2nd ed. Chapman &
  Hall/CRC. Generation orders (sec. 3.7), branch and bound (sec. 3.8), the
  accuracy of inverse-based updating (sec. 2.3).
