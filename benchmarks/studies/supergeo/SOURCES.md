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
