"""Write the runnable copies of the authors' scripts that the reference traces came from.

    python R/prepare_scripts.py <replications_emp dir> <out dir>

Germany: ``replication_germany.R`` split into one script per specification
(its preamble plus that specification's block, verbatim), so one failing fit
costs one block. California: ``replication_california.R`` with the edits
listed in the README, each needed to run it on R 4.3 here; none touches an
estimate. The copies are written next to each other in ``<out dir>``.
"""

import re
import sys
from pathlib import Path

SEP = "#" * 79 + "\n"

CALIFORNIA_EDITS = [
    # R 4 no longer reads strings as factors; the script builds state names
    # from levels(smoking$state).
    ('smoking <- read.csv(file = "smoking_dataset.csv")',
     'smoking <- read.csv(file = "smoking_dataset.csv", stringsAsFactors = TRUE)'),
    # Loaded, never used, not packaged for apt.
    ('library("statar")', '# library("statar")'),
    # Sequential foreach, so the in-process trace sees every fit.
    ("cl <- makeCluster(n.cores)", "# cl <- makeCluster(n.cores)"),
    ("registerDoParallel(cl)", "registerDoSEQ()"),
    ("stopCluster(cl)", "# stopCluster(cl)"),
    # LowRankQP is archived from CRAN and not installable here: Synth's default ipop.
    ('method = "BFGS",quadopt = "LowRankQP")', 'method = "BFGS")'),
    # A failed fit becomes a missing row instead of stopping the script.
    ('synth.out <- synth(data.prep.obj = dataprep.out, method = "BFGS")',
     'synth.out <- tryCatch(synth(data.prep.obj = dataprep.out, method = "BFGS"), '
     'error = function(e) NULL); if (is.null(synth.out)) return(rep(NA, 37))'),
]


def germany(src: Path, out: Path) -> None:
    text = src.read_text()
    head, rest = text.split(SEP + "# Specification 1a\n", 1)
    body = ("# Specification 1a\n" + rest).split(SEP + "# Treatment effect per specification", 1)[0]
    blocks = [b for b in re.split(r"(?=^# Specification \w+\n)", body, flags=re.M)
              if b.startswith("# Specification")]
    names = [re.match(r"# Specification (\w+)", b).group(1) for b in blocks]
    if names != [f"{k}{c}" for k in "1234567" for c in "ab"]:
        raise ValueError(f"unexpected specification blocks: {names}")
    for name, block in zip(names, blocks):
        (out / f"spec_{name}.R").write_text(head + SEP + block.rstrip(SEP).rstrip() + "\n")


def california(src: Path, out: Path) -> None:
    text = src.read_bytes().decode()
    for old, new in CALIFORNIA_EDITS:
        if old not in text:
            raise ValueError(f"edit target not found: {old!r}")
        text = text.replace(old, new)
    (out / "replication_california_seq.R").write_bytes(text.encode())


if __name__ == "__main__":
    pkg, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    germany(pkg / "germany_replication" / "replication_germany.R", out)
    california(pkg / "california_replication" / "replication_2017-11-30"
               / "replication_california.R", out)
    print(f"wrote 14 Germany scripts and the California copy to {out}")
