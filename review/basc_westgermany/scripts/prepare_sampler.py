"""Derive the three sampler files the R scripts source, from the authors' own
BASC_realdata.R.

The authors' code is not redistributed here. This script reads their file from a
local clone and writes the pieces the diagnostics need, so every modification is
visible as code instead of being asserted in prose.

    git clone https://github.com/sll-lee/paper-BASC
    python scripts/prepare_sampler.py paper-BASC/BASC_realdata.R

Three files are produced in the working directory:

  basc_funcs.R     the authors' function definitions, safe_chol through
                   run_basc_chain, extracted verbatim with nothing altered.
  basc_funcs_g1.R  adds four arguments: force_gamma1 skips the loop that
                   resamples the donor-inclusion indicators, leaving gamma at
                   the rep(1, j) the sampler already initialises it to;
                   alpha_u_override sets the Dirichlet dispersion; init_u and
                   init_sigsq set starting values.
  basc_funcs_q.R   generalises the post-treatment term from a scalar to a
                   basis. The published code writes `Dt * alpha.v`, which is
                   scalar recycling and works only for q = 1; this becomes
                   `Dt %*% alpha.v` at the five sites where it appears. The
                   alpha update already handles a T x q design.

Run at q = 1 with force_gamma1 = FALSE, the generated code reproduces the
original sampler exactly. That is the control every diagnostic is read against.
"""
import io
import sys


def extract(src_text):
    """The authors' function definitions: safe_chol through run_basc_chain."""
    lines = src_text.splitlines(True)
    start = next(i for i, l in enumerate(lines) if l.startswith("safe_chol <- function"))
    rbc = next(i for i, l in enumerate(lines) if l.startswith("run_basc_chain <- function"))
    end = next(i for i in range(rbc, len(lines)) if lines[i].rstrip("\n") == "}")
    return "".join(lines[start:end + 1])


def add_switches(text):
    """gamma forcing, the Dirichlet dispersion, and initial values."""
    out = text.replace(
        "run_basc_chain <- function(seed, y, x, vt, Dt, N, nburn, q = 1) {",
        "run_basc_chain <- function(seed, y, x, vt, Dt, N, nburn, q = 1,\n"
        "                           force_gamma1 = FALSE, alpha_u_override = NULL,\n"
        "                           init_u = NULL, init_sigsq = NULL) {", 1)
    assert out != text, "chain signature not found"
    text, out = out, out.replace(
        "  alpha_u <- 2.5",
        "  alpha_u <- 2.5\n  if (!is.null(alpha_u_override)) alpha_u <- alpha_u_override", 1)
    assert out != text, "alpha_u assignment not found"
    text, out = out, out.replace(
        "  u <- rgamma(j, alpha_u, scale = 1 / alpha_u)",
        "  u <- rgamma(j, alpha_u, scale = 1 / alpha_u)\n  if (!is.null(init_u)) u <- init_u", 1)
    assert out != text, "u initialisation not found"
    text, out = out, out.replace(
        "  sigsq <- rinvgamma(1, alpha_sig, beta_sig)",
        "  sigsq <- rinvgamma(1, alpha_sig, beta_sig)\n"
        "  if (!is.null(init_sigsq)) sigsq <- init_sigsq", 1)
    assert out != text, "sigsq initialisation not found"
    text, out = out, out.replace(
        "    # 6) Update gamma_j\n    for (p in 1:j) {",
        "    # 6) Update gamma_j\n    for (p in if (force_gamma1) integer(0) else 1:j) {", 1)
    assert out != text, "gamma update loop not found"
    return out


def generalise_q(text):
    """The post-treatment term becomes a basis instead of a scalar."""
    n = text.count("Dt * alpha.v")
    assert n == 5, f"expected 5 scalar-recycling sites, found {n}"
    return text.replace("Dt * alpha.v", "as.numeric(Dt %*% alpha.v)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python prepare_sampler.py <path to BASC_realdata.R>")
    src = io.open(sys.argv[1], encoding="utf-8").read()
    base = extract(src)
    g1 = add_switches(base)
    q = generalise_q(g1)
    for name, body in [("basc_funcs.R", base), ("basc_funcs_g1.R", g1), ("basc_funcs_q.R", q)]:
        io.open(name, "w", encoding="utf-8").write(body)
        print(f"wrote {name} ({body.count(chr(10))} lines)")
