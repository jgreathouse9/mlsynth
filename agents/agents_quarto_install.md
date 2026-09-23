# Installing Quarto in this sandbox — an exact runbook

A step-by-step recipe for getting a working [Quarto](https://quarto.org) CLI into the
Claude Code remote sandbox, including the Python execution engine, both PDF paths, and
Mermaid/Graphviz rendering. Read `agents_r_environment.md` first for the general network
constraints; this file is the Quarto-specific replay, and it follows the same shape as
`agents_discos_install.md`.

Verified end to end on 2026-09-07, on Ubuntu 24.04 (x86_64) with Python 3.11.15, as root.
Installed versions: Quarto 1.10.18, TinyTeX v2026.09 (TeX Live 2026).

## TL;DR

```bash
# 1. Quarto itself (~147 MB download, ~447 MB installed)
QV=1.10.18
curl -sSL -o /tmp/quarto.tar.gz \
  "https://github.com/quarto-dev/quarto-cli/releases/download/v${QV}/quarto-${QV}-linux-amd64.tar.gz"
mkdir -p /opt/quarto
tar -xzf /tmp/quarto.tar.gz -C /opt/quarto --strip-components=1
ln -sf /opt/quarto/bin/quarto /usr/local/bin/quarto

# 2. The Python execution engine (only needed for .qmd files with {python} chunks)
python3 -m pip install jupyter nbclient

# 3. LaTeX, for format: pdf (~200 MB download). Skip if typst PDF is enough.
TV=v2026.09
curl -sSL -o /tmp/TinyTeX.tar.gz \
  "https://github.com/rstudio/tinytex-releases/releases/download/${TV}/TinyTeX-${TV}.tar.gz"
tar -xzf /tmp/TinyTeX.tar.gz -C /opt          # unpacks to /opt/.TinyTeX
ln -sf /opt/.TinyTeX/bin/x86_64-linux/* /usr/local/bin/

# 4. Mermaid and Graphviz diagrams reuse the pre-installed Playwright Chromium
export QUARTO_CHROMIUM=/opt/pw-browsers/chromium-1194/chrome-linux/chrome

quarto check
```

Budget 5–10 minutes on a cold container. Nothing compiles; all four steps are downloads
and extractions.

## What is reachable and what is not

The container's egress runs through the agent proxy described in `/root/.ccr/README.md`.
The distinction that decides this recipe is not host-by-host allowlisting of everything
GitHub, but the split between GitHub's API and its release-asset downloads.

| Host / route | Result | Consequence |
| --- | --- | --- |
| `pypi.org`, `files.pythonhosted.org` | open (in `noProxy`) | `pip install jupyter` works |
| `registry.npmjs.org` | open (in `noProxy`) | irrelevant; Quarto is not on npm |
| `github.com/<owner>/<repo>/releases/download/...` | open | the route this recipe uses |
| `api.github.com/repos/...` | 403, session-scoped | breaks anything that resolves "latest" |
| `github.com/<owner>/<repo>/releases/latest` | 403 | same |
| `quarto.org` | 403 (CONNECT tunnel refused) | the documented `.deb` route is unavailable |
| `mirror.ctan.org`, `ctan.org`, `tlnet.yihui.org` | unreachable | `tlmgr install` cannot add packages |
| `cran.r-project.org` | unreachable | no R engine; see `agents_r_environment.md` |

Release-asset downloads became available after calling `add_repo` for `quarto-dev/quarto-cli`
(and again for `rstudio/tinytex-releases`), which reports the repository as already served
by the session's anonymous git-read lane. Call `add_repo` for each of the two repositories
before the downloads; it attaches nothing and costs one tool call.

## Why the obvious routes fail

Four install methods appear first in any search, and three of them are dead here.

The official installer at `quarto.org/download` needs `quarto.org`, which the proxy
refuses at the CONNECT stage. Nothing downstream of that page is reachable either.

`pip install quarto-cli` looks promising, because PyPI is open and the package exists at
1.10.18. Its sdist is 4.7 KB: the wheel ships no binary, and `setup.py` fetches one at
build time from

```
https://github.com/quarto-dev/quarto-cli/releases/download/v{version}/quarto-{version}-linux-amd64.tar.gz
```

which is exactly the URL step 1 fetches directly. The pip route adds a build step that
can fail silently — `download_quarto` catches its own exception, prints, and returns
`None` — for no benefit. Fetch the tarball yourself and see the HTTP code.

apt does not carry Quarto in any Ubuntu 24.04 channel, and npm carries an unrelated
package named `quarto` (a UI-component tool, version 0.0.0). Neither is the right thing.

The route that works is the release tarball, unpacked into `/opt` with one symlink onto
PATH. Quarto ships Pandoc 3.10.0, Dart Sass 1.101.0, Deno 2.7.14 and Typst 0.15.1 inside
the tarball, so the core install has no system dependencies at all.

## The four traps

### `quarto install tinytex` cannot work here

Quarto's own tool installer resolves the newest release through `api.github.com`, which
this session refuses:

```
ERROR: Unable to determine latest release for rstudio/tinytex-releases
403 - Forbidden
```

The same 403 blocks `quarto install chromium` and `quarto update`. Every tool Quarto
would fetch for you has to be placed by hand. This generalises: any CLI that resolves
"latest" through the GitHub API fails in this sandbox even when its release assets
download fine, so pin an explicit version and construct the asset URL yourself.

### The TinyTeX asset name has no version-prefixed variant

`git ls-remote --tags https://github.com/rstudio/tinytex-releases` lists the tags without
touching the API, which is how step 3 learns that `v2026.09` is current. The asset naming
is the part that costs time: `TinyTeX-1-v2026.09.tar.gz`, `TinyTeX-1.tar.gz` and
`TinyTeX.tar.gz` all 404. Only `TinyTeX-v2026.09.tar.gz` exists, and it is the default
scheme, not the minimal one — which is the reason a `format: pdf` render succeeds
immediately on it. Probe candidate names with
`curl -o /dev/null -L -w "%{http_code}"` before committing to a 200 MB download.

### `tlmgr path add` exits 0 without linking anything

It reports success and creates no symlinks, so `pdflatex` stays missing. Do the linking
yourself:

```bash
ln -sf /opt/.TinyTeX/bin/x86_64-linux/* /usr/local/bin/
```

After that `pdflatex --version` reports `pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)`
and `quarto check` reports `LaTeX ... Using: Installation From Path`.

### `matplotlib.use("Agg")` suppresses figures in a rendered document

This one is not Quarto's doing, but it wasted a render cycle and will waste another. Under
the Jupyter engine, `matplotlib.use("Agg")` overrides the IPython inline backend, so
`plt.show()` emits nothing and the HTML contains the chunk's stdout and no image. The
document renders, exits 0, and is silently wrong. Leave the backend alone in `.qmd` chunks:
the inline backend is already non-interactive.

## What each piece buys you

Step 1 alone renders Markdown to HTML, and to PDF through Typst
(`quarto render doc.qmd --to typst`), because Typst is bundled in the tarball. If PDF is
all that is needed, stop after step 1 and skip the 200 MB TinyTeX download entirely.

Step 2 is needed only for `.qmd` files with executable `{python}` chunks. Without it,
`quarto check` reports `Jupyter: (None)` and any executable chunk fails. The chunks run in
whatever `python3` is on PATH, so the packages the document imports have to be installed
there — this container starts without numpy or matplotlib.

Step 3 adds `format: pdf` through LaTeX, which is the path any document using LaTeX
packages or `\include`d TeX needs. Quarto drives it with LuaLaTeX; the first run spends a
minute building the luaotfload font database and later runs do not.

Step 4 adds Mermaid and Graphviz diagrams in non-HTML formats, and anything else Quarto
delegates to headless Chrome. The environment's Playwright Chromium serves; point
`QUARTO_CHROMIUM` at it and `quarto check` reports `Using: Chrome from QUARTO_CHROMIUM`.
Confirm the build number first, since `chromium-1194` will change:

```bash
ls -d /opt/pw-browsers/chromium-*/chrome-linux/chrome
```

## What still does not work

`tlmgr install <pkg>` fails — every CTAN mirror is unreachable, so the TeX installation is
frozen at whatever the TinyTeX default scheme ships. A document needing a package outside
that scheme has two options: render it through Typst instead, or obtain the package's TeX
Live tarball from a GitHub-hosted mirror and unpack it into `/opt/.TinyTeX/texmf-local`.

The R engine is unavailable: R is not installed and CRAN is blocked. `.qmd` files with
`{r}` chunks cannot render. `agents_r_environment.md` covers the apt-plus-`cran/<pkg>`-clone
workaround for R packages if an R install is added later.

`quarto publish` targets (Quarto Pub, Netlify, GitHub Pages) are untested and most likely
blocked.

## Verification transcript

```
$ quarto --version
1.10.18

$ quarto check
[✓] Checking versions of quarto binary dependencies...
      Pandoc version 3.10.0: OK
      Dart Sass version 1.101.0: OK
      Deno version 2.7.14: OK
      Typst version 0.15.1: OK
[✓] Checking Quarto installation......OK
      Version: 1.10.18
      Path: /opt/quarto/bin
[✓] Checking tools....................OK
      TinyTeX: (external install)
[✓] Checking LaTeX....................OK
      Using: Installation From Path
      Path: /usr/local/bin
      Version: 2026
[✓] Checking Chrome Headless....................OK
      Using: Chrome from QUARTO_CHROMIUM
[✓] Checking basic markdown render....OK
[✓] Checking Python 3 installation....OK
      Version: 3.11.15
      Jupyter: 5.9.1
[✓] Checking Jupyter engine render....OK
```

Four renders were exercised, each producing output:

| Input | Command | Output |
| --- | --- | --- |
| Markdown only | `quarto render smoke.qmd` | `smoke.html` |
| Markdown only | `quarto render smoke.qmd --to typst` | `smoke.pdf` |
| Markdown only | `quarto render smoke.qmd --to pdf` | `smoke.pdf` via LuaLaTeX |
| `{python}` chunk plotting with matplotlib | `quarto render pysmoke.qmd` | `pysmoke.html` with `pysmoke_files/figure-html/` |

The Python case is the one to re-run after any change, since it is the only one that
touches all three of Quarto, the Jupyter engine and the interpreter's packages:

````bash
cat > /tmp/pysmoke.qmd <<'EOF'
---
title: "Quarto + Python smoke test"
format: html
jupyter: python3
---

```{python}
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
y = rng.normal(size=50).cumsum()
fig, ax = plt.subplots()
ax.plot(y)
ax.set_title("random walk")
plt.show()
print("rows:", y.shape[0])
```
EOF
quarto render /tmp/pysmoke.qmd
grep -o 'img src="[^"]*"' /tmp/pysmoke.html   # must print a figure-html path
````

The `grep` is the assertion. A render that emits `rows: 50` and no `<img>` means the
figure was dropped — see the `matplotlib.use("Agg")` trap above.

## Persistence

The container is ephemeral and reclaimed after inactivity, so `/opt/quarto`,
`/opt/.TinyTeX` and the `/usr/local/bin` symlinks do not survive into the next session.
Nothing in this install belongs in the repository — the ~650 MB of binaries are not
committable and would break the working tree. Re-run the TL;DR block at the start of any
session that needs Quarto, or wire it into a `SessionStart` hook.

`QUARTO_CHROMIUM` is a shell variable and does not persist across Bash tool calls. Either
export it in the same command that renders, or write it into the environment the hook sets
up.
