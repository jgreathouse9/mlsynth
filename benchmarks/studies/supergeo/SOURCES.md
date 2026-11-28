# Sources: the supergeo study

One entry per source, in the form `agents/agents_citations.md` sets. This file
was started with the phase 4 roadmap update and holds the sources that update
cites; the sources the roadmap's earlier phases cite are still to be entered.
Uploaded PDFs are not committed; each entry records the version and what was
read.

## chen2023supergeo
- Reference: Chen, A., Doudchenko, N., Jiang, S., Stein, C., & Ying, B. (2023).
  "Supergeo Design: Generalized Matching for Geographic Experiments."
  arXiv:2301.12044v1 (28 Jan 2023).
- Read: sections 1 and 2 (pp. 1-8), from the PDF's extracted text, page
  numbers from the printed footers.
- Used for:
  - When units are aggregated, the covariates used for matching are added up
    (section 1, p. 4; the authors call this an assumption of the current paper).
  - The treated half of each supergeo pair is chosen by a fair coin after the
    pairs are formed (section 2, p. 7).
  - The design objective and the variance of the iROAS estimator,
    Var(theta_hat) = sum_k (Z_{G_k,+} - Z_{G_k,-})^2 / B^2 (section 2, eq. (3),
    p. 8), under a homogeneous effect and a fixed budget (Assumptions 1 to 3).
- Checks: none second-hand.

## wing2024stacked
- Reference: Wing, C., Freedman, S. M., & Hollingsworth, A. (2024). "Stacked
  Difference-in-Differences." NBER Working Paper 32054, January 2024.
- Read: the full PDF's extracted text; sections 4 and 5 (pp. 12-18) in detail,
  page numbers from the printed footers.
- Used for:
  - The trimmed aggregate ATT weights each sub-experiment's ATT by its share of
    the treated units (section 4, eq. (2), p. 12).
  - The population-weighted ATT and the sample-share-weighted ATT, with the
    treatment and control weights that identify each (Table 1, p. 13).
  - The unweighted stacked regression weights treatment and control trends
    differently across sub-experiments, and the corrective weight Q_sa (1 for
    treated, (N_a^D / N^D) / (N_a^C / N^C) for control) weights them equally,
    so the weighted stacked DID equals the target aggregate (section 5.3,
    pp. 17-18). Conditions: common trends and no anticipation within each
    sub-experiment, weights fixed across event time.
- Checks: none second-hand.

## yamamoto2014ffkm
- Reference: Yamamoto, M., & Terada, Y. (2014). "Functional factorial K-means
  analysis." Computational Statistics and Data Analysis 79, 133-148.
- Read: pp. 133-139, from the PDF's extracted text, page numbers from the
  running heads.
- Used for: when the part of the data that carries the cluster structure has no
  substantial correlations and a masking part is present, a dimension reduction
  chosen without the cluster structure (the FPCK method, and tandem analysis)
  can fail to find the true clusters (section 2, p. 136).
- Checks: none second-hand.

## malinen2014balanced
- Reference: Malinen, M. I., & Franti, P. (2014). "Balanced K-Means for
  Clustering." In Structural, Syntactic, and Statistical Pattern Recognition
  (S+SSPR 2014), LNCS 8621, pp. 32-41. Springer.
- Read: pp. 32-36, from the PDF (page 35 located by extracting it alone).
- Used for: the assignment step fixes the cluster sizes by assigning points to
  n pre-allocated slots (n/k per cluster, sizes differing by at most one when k
  does not divide n) and solving the assignment problem with the Hungarian
  algorithm (section 2, p. 35).
- Checks: our use (`scipy.optimize.linear_sum_assignment` on the slot costs)
  was checked against brute force with the cluster sizes held fixed.

## shaw2025asd
- Reference: Shaw, C. (2025). "Adaptive Supergeo Design: A Scalable Framework
  for Geographic Marketing Experiments." arXiv:2506.20499v2 (30 Jun 2025).
- Read: Appendix A (p. 21), from the PDF's extracted text.
- Used for: nothing. PANGEO's docs no longer cite this paper or its predecessor
  (Optimized Supergeo Design).
- Checks: failed. Lemma A.3 ("Hierarchical clustering fidelity", p. 21) states
  that if rho_in - rho_out > 2 Delta, Ward-linkage HAC exactly recovers the
  true communities, citing "Theorem 3 of [24]", where [24] is von Luxburg
  (2010), "Clustering stability: an overview". That paper has no Theorem 3 and
  nothing on Ward linkage (see vonluxburg2010stability). The lemma's claim is
  unsupported by its citation, and the paper's results that rest on it (the
  design-cost bound of Lemma A.4) are unchecked.

## vonluxburg2010stability
- Reference: von Luxburg, U. (2010). "Clustering stability: an overview."
  arXiv:1007.1075v1 (7 Jul 2010).
- Read: the whole paper, from the PDF's extracted text, searched for every
  numbered theorem and for "Ward".
- Used for: the check of shaw2025asd's Lemma A.3 only.
- Checks: the theorems named in the text are Theorems 1, 2 and 6; no Theorem 3
  appears, and "Ward" occurs only inside other words and in one reference's
  author list. The journal version (Foundations and Trends in Machine Learning
  2(3), 235-274, as Shaw cites it) was not read, so the check is against the
  arXiv version.

## sergin2025lrtsd
- Reference: Sergin, N. D., Hu, J., Li, Z., Zhang, C., Tsung, F., & Yan, H. (2025).
  "Low-Rank Robust Subspace Tensor Clustering for Metro Passenger Flow
  Modeling." INFORMS Journal on Data Science 4(1), 33-50.
  doi:10.1287/ijds.2022.0028.
- Read: the abstract, the section headings, section 3.3 and Algorithm 1
  (pp. 38-39), from the PDF's extracted text, page numbers from the running
  heads.
- Used for: the partial Tucker decomposition on the non-clustering modes,
  L = C x_2 U_2 x_3 U_3 with a core C in R^{N x P2 x P3} that keeps the sample
  mode, orthonormal U_2 and U_3, and clustering done on the core (section 3.3,
  eq. (1), p. 39). The roadmap uses only this embedding; the paper's
  self-expressive subspace clustering and sparse anomaly term are not used.
- Checks: none second-hand.

## zhang2018lrrhtd
- Reference: Zhang, J., Li, X., Jing, P., Liu, J., & Su, Y. (2018). "Low-Rank
  Regularized Heterogeneous Tensor Decomposition for Subspace Clustering."
  IEEE Signal Processing Letters 25(3), 333-337.
- Read: the whole letter, from the PDF's extracted text.
- Used for: its clustering step is k-means on the rows of the low-rank
  sample-mode factor Z (section III, Algorithm 1, step 16), the embedding the
  Tucker POC compares against. Its experiments are two face-image datasets
  (CBCL, ORL) scored against image labels.
- Checks: none second-hand.

## baldacci2008vrp
- Reference: Baldacci, R., Christofides, N., & Mingozzi, A. (2008). "An exact
  algorithm for the vehicle routing problem based on the set partitioning
  formulation with additional cuts." Mathematical Programming, Ser. A 115,
  351-385. doi:10.1007/s10107-007-0178-5.
- Read: the abstract and section 4.1 (p. 360), from the PDF's extracted text;
  the page located by extracting it alone.
- Used for: the exact method solves the problem reduced to the routes whose
  reduced cost, at the dual solution of the bounding procedure, is smaller than
  the gap between a valid upper bound and the bound, with an integer
  programming solver (section 4.1, p. 360). The upper bound is a valid one.
- Checks: this is the origin wang2023fmrsp credits for the reduced problem;
  checked here.

## wang2023fmrsp
- Reference: Wang, S., Baldacci, R., Yu, Y., Zhang, Y., Tang, J., Luo, X., &
  Sun, W. (2023). "An Exact Method for a First-Mile Ridesharing Problem."
  Transportation Science 57(6), 1581-1604. doi:10.1287/trsc.2022.0139.
- Read: the abstract and sections 4.3 and 4.4 (pp. 1590-1592), from the PDF's
  extracted text, page numbers from the running heads.
- Used for: the primal heuristic that guesses the upper bound as
  (1 + alpha) times the lower bound (alpha = 0.03), enumerates the routes in
  that gap, solves the reduced problem, repeats with the design found as the
  new bound, and stops with a proof when the reduced problem is complete
  (section 4.3, p. 1591).
- Checks: the reduced problem is credited to baldacci2008vrp, checked at its
  origin.

## sato2019modularity
- Reference: Sato, K., & Izunaga, Y. (2019). "An enhanced MILP-based
  branch-and-price approach to modularity density maximization on graphs."
  Computers and Operations Research 106, 236-245.
- Read: the abstract, the contributions list (pp. 237-238) and sections 3.1 to
  3.3 (pp. 239-240), from the PDF's extracted text, page numbers located by
  extracting pages alone.
- Used for: the set-packing relaxation of the restricted master (the
  set-covering relaxation of Sato and Fukumura for minimisation), restoring a
  constraint when it is over-satisfied (section 3.2, p. 239); generating
  several vertex-disjoint columns per pricing round (section 3.3, p. 240); the
  same-or-different branching rule (section 3.1, p. 239).
- Checks: Sato and Fukumura (2012) is cited through this paper and not read.

## sherali1996spp
- Reference: Sherali, H. D., & Lee, Y. (1996). "Tighter representations for
  set partitioning problems." Discrete Applied Mathematics 68, 153-167.
- Read: the abstract and sections 1 to 4 (pp. 153-163), from the PDF's
  extracted text (a scan with an OCR layer), page numbers located by
  extracting pages alone.
- Used for: the level-d relaxation multiplies the cover rows by every product
  of d variables; the first level SPP_1 keeps, for each column k and each row
  i that k does not cover, x_k = sum of w_(jk) over the columns j covering i
  that share no row with k (eqs. (12)-(14), p. 158). It implies Balas's
  elementary inequalities (Proposition 2, p. 159), and fixing w_(jk) = 0 for
  the columns j outside a set Q makes x_k <= sum over Q of x_j valid
  (Proposition 4, p. 160). The roadmap's RLT-2 is this first level, with the
  pairs restricted to those that fit in the gap.
- Checks: none second-hand.

