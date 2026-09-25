# Runs behind the tables in the study README

| file | what produced it |
| --- | --- |
| `cells_dgp6.txt` | DGP6 at both sample configurations, 100 replications, `r = 2` |
| `cells_dgp7_and_cv_r.txt` | DGP7 at both configurations with `r = 2`, then DGP6 with `r` cross-validated |
| `empirics_section7.txt` | Tables 9, 10 and 11-16 through `run_empirics.py` |
| `link_c_candidates.txt` | the Link C sweep: the LASSO penalty grid, the M2 random split, the aggregation form and both factor-count rules, 120 replications per cell |

All five ran in one container at one sitting. The cells are reproduced by
`experiment.py`, the empirics by `run_empirics.py`, the Link C sweep by
`link_c_candidates.py`.
