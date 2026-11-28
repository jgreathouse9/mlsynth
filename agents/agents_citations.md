# AGENTS.md — Citing books, papers and theorems (binding)

> Status: binding for every docs page, docstring, benchmark case, study branch
> (`benchmarks/studies/**`), roadmap and PR description. A citation is a claim
> that a source says something. This file sets how that claim is checked and
> recorded so that a reader can check it again.

## Why this exists

A citation that has not been checked is a guess presented as a fact. The guess
is usually right, which is why it survives review, and it is wrong often enough
to matter. Two cases from the PANGEO study branch:

- Our docs placed the supergeo design's coin flip in "Chen et al. (2023,
  section 3)". The paper defines it in section 2 ("Supergeo design", p. 7);
  section 3 is the algorithm.
- A paper we read gave a (1 + ε) guarantee whose recovery step cited "Theorem
  3 of" a survey. The survey (von Luxburg, "Clustering stability: an overview",
  arXiv 1007.1075v1) has no Theorem 3 and never discusses the clustering
  method the step relies on. The guarantee looked supported until the cited
  source was opened.

Both checks took minutes. Neither would have been made without a rule.

## The rules

### 1. Check against the source's own pages

Cite a result only after reading it in the source: the statement, its
conditions, and the page it is on. The abstract, a later paper's summary, or
memory of the paper is not the source. If the source is not available, do not
cite it; describe the idea by its construction instead (for example, "a
grouped fixed-effects panel: a time path per group plus noise"), or ask the
author for the PDF.

### 2. Cite at the granularity of the claim

A citation points to the smallest unit that holds the claim:

| Claim | Cite |
| --- | --- |
| a theorem, lemma, proposition | its number and page: "Theorem 5.15, p. 8" |
| an equation or formulation | its number and page: "eq. (3), p. 8" |
| an algorithm | its number and page: "Algorithm 1, p. 12" |
| a definition or condition | its number or section and page: "Definition 3.1, p. 4" |
| a simulation or table result | the table or figure and page: "Table 4, p. 18" |
| an argument in prose | the section and page: "section 2, p. 7" |
| a book | edition, section, and page: "4th ed., section 5.3.2, p. <n>" |

"Smith (2020)" alone is acceptable only when the whole paper is the claim (the
paper proposed the method).

### 3. State the version that was read

Page and theorem numbers change between versions. Record what was read:

- arXiv: the identifier with its version, `arXiv 2301.12044v1`. A later
  version can renumber theorems, move pages, or change the title.
- Journal: volume, issue, pages, year, and the DOI when there is one.
- Book: edition and publisher.
- Technical report: institution and number, `MSR-TR-2000-65`.

When the published version and a preprint both exist, cite the one that was
read, and say which.

### 4. State the conditions with the result

A theorem cited without its assumptions misstates it. Give the conditions in
words next to the citation: "Var[θ̂] = Σ_k (Z_{G_k,+} − Z_{G_k,−})² / B²
under homogeneous effects and a fixed budget (Assumptions 1 to 3, eq. (3),
p. 8)". This is the citation form of invariant 8's "state the conditions under
which it holds".

### 5. Second-hand results are checked at their origin

When a paper's argument rests on a result it cites, and our work rests on that
argument, open the cited source too and check the result there (rule 1). A
paper's citation of a result is the paper's claim, not ours to repeat. If the
cited source does not contain the result, record that (rule 8) and do not rely
on the argument.

### 6. Quote exactly or paraphrase plainly

A quotation is copied character for character from the source, inside
quotation marks, with its page. Words inside quotation marks are never edited,
including for the banned-word rules in CLAUDE.md (the SYNDES authors' "rather
strong assumptions" stays as written). Anything not copied exactly is a
paraphrase and carries no quotation marks.

### 7. Separate what the source says from what we measured

A docs page or study note says which statements come from a source and which
come from our own runs. "Chen et al. (2023, eq. (3)) give the variance as
...; on the 174-geo example panel we measured ..." A measured number carries
its configuration (panel, seed, budgets, threads) or a pointer to the script
and log that hold it.

### 8. Record negative checks

A check that fails is a finding. Record it where the next reader will look:
in the study notes or roadmap, and in the PR body when the PR depends on it.
Record what was claimed, where, what the cited source actually contains, and
the pages checked: "Lemma A.3 cites Theorem 3 of von Luxburg (2010); the
survey (arXiv 1007.1075v1) states Theorems 1, 2 and 6 and Lemmas 1, 2, 6 and 7
(pp. 14 to 29) and does not discuss Ward linkage." A failed check is never
silently dropped, because the next reader will otherwise cite the same claim.

### 9. Unverified work is not support

Results from a source whose central claims have failed a check are not cited
as support for anything in mlsynth. They may be described as related work by
their method, without repeating their performance or optimality claims.

### 10. Text extraction has limits

Checks made on text extracted from a PDF (`pdftotext`) cannot see content that
exists only in figures, or text in a broken font encoding. Say when a check
rests on extracted text, and treat "not found" as "not found in the text",
checking the figures by eye when the claim could live there. A PDF whose text
does not extract is not cited until it is read some other way.

## Study branches

Study branches (`benchmarks/studies/<name>/`) accumulate sources quickly, and
their notes become the record that docs pages and PRs are written from. Each
study keeps one sources file, `benchmarks/studies/<name>/SOURCES.md`, with one
entry per source:

```
## <short key>  (e.g. chen2023supergeo)
- Reference: full citation with version (rule 3).
- Read: which pages or sections were read, and how (PDF, extracted text).
- Used for: the claims this study rests on, each with its location (rule 2)
  and conditions (rule 4).
- Checks: anything verified at a second-hand origin (rule 5), and every failed
  check (rule 8).
```

Docs pages and PRs cite from SOURCES.md, not from memory. A source not in
SOURCES.md is not cited by the study's outputs. Uploaded PDFs are not
committed (licences); SOURCES.md records the reference and what was read.

## Reference lists on docs pages

Every source cited on a page appears in the page's References section, in the
form already used on the pages:

```
Chen, A., Doudchenko, N., Jiang, S., Stein, C., & Ying, B. (2023).
"Supergeo Design: Generalized Matching for Geographic Experiments."
arXiv:2301.12044v1.
```

Authors, year, title, venue with volume and pages or the arXiv identifier with
version, and the DOI when there is one. A source listed but not cited in the
text is removed.

## Checklist before a citation goes anywhere persistent

- [ ] Read in the source itself, at the page cited (rules 1, 5).
- [ ] Theorem, equation, algorithm, table or section number given, with page (rule 2).
- [ ] Version recorded (rule 3).
- [ ] Conditions stated (rule 4).
- [ ] Quotations exact (rule 6).
- [ ] Source claims separated from our measurements (rule 7).
- [ ] Failed checks recorded (rule 8).
- [ ] Entry in the study's SOURCES.md, when the work comes from a study branch.
