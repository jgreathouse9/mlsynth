"""Progress reporting for the TBR certified search (#965).

The search emits events; the caller decides where they go. Each kind of event has a
level and a declared set of fields (``KINDS``):

1. the treatment search and each treatment group, each gated round, the group
   screen, the hull, each size and each new incumbent;
2. every node of a size's branch and bound, with its outcome and the tier that
   decided it;
3. every first-order schedule, first-order triangle round, relaxation solve and
   priced subtree, with its time;
4. adds Clarabel's own iteration log (stdout, Clarabel's format), no events.

``reporting`` routes the events of the code it wraps. ``verbose`` prints them to
stderr, one line each, indented by level and depth; ``progress`` calls a function
with each ``Event``; ``log`` writes them to a parquet dataset, one file per process
and flush (``part-<pid>-<run>-<n>.parquet``), read back whole by ``read_log``.
Without any of the three nothing is reported and the search runs as before:
reporting never changes what the search computes.

The reporter is module state, so a worker forked by certify_treatment's pool reports
through it too: its printer writes to the shared stderr, its log to its own files,
and a ``progress`` function runs in the worker's process. Each group's events carry
the group as the tag ``treatment``. A forked worker leaves without running exit
handlers, so a log flushes at the end of every tagged group as well as every
``FLUSH_EVERY`` events, every ``FLUSH_SECONDS``, and when reporting ends."""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from mlsynth.exceptions import MlsynthConfigError

#: A parquet log writes a file after this many events in one process, or once this
#: many seconds have passed since its last write, so a long run is readable as it goes.
FLUSH_EVERY = 20000
FLUSH_SECONDS = 60.0

#: kind -> (level, declared fields)
KINDS: Dict[str, Tuple[int, Tuple[str, ...]]] = {
    "treatment": (1, ("k", "p", "groups", "n_jobs")),
    "treatment-done": (1, ("certified", "treatment", "rss", "passing", "survivors", "seconds")),
    "group": (1, ("treatment", "ceiling")),
    "group-done": (1, ("treatment", "certified", "passing", "rounds", "seconds")),
    "gated-round": (1, ("round", "controls", "rss", "passing", "seconds")),
    "slope-free": (1, ("p", "ratio", "test_ratio", "excluded")),
    "screen": (1, ("certified", "tier", "rung", "seconds")),
    "hull": (1, ("sizes", "seconds")),
    "size": (1, ("m", "nodes", "open", "seconds")),
    "incumbent": (1, ("ratio", "controls", "source", "m", "depth")),
    "slope-free-done": (1, ("certified", "nodes", "ratio", "controls", "seconds")),
    "node": (2, ("m", "depth", "ones", "outcome", "tier", "control", "seconds")),
    "schedule": (3, ("m", "rung", "maxfun", "closed", "seconds")),
    "triangles": (3, ("m", "round", "triangles", "closed", "seconds")),
    "solve": (3, ("relaxation", "solver", "rows_eq", "rows_in", "status", "seconds")),
    "price": (3, ("m", "depth", "designs", "exact", "seconds")),
}

_INT, _FLOAT, _STR, _BOOL, _INTS = pa.int64(), pa.float64(), pa.string(), pa.bool_(), pa.list_(pa.int64())
FIELD_TYPES = {
    "k": _INT, "p": _INT, "groups": _INT, "n_jobs": _INT, "m": _INT, "depth": _INT, "ones": _INT,
    "control": _INT, "rung": _INT, "maxfun": _INT, "round": _INT, "triangles": _INT, "rows_eq": _INT,
    "rows_in": _INT, "designs": _INT, "exact": _INT, "nodes": _INT, "open": _INT, "rounds": _INT,
    "survivors": _INT, "excluded": _INT,
    "ratio": _FLOAT, "test_ratio": _FLOAT, "rss": _FLOAT, "ceiling": _FLOAT, "seconds": _FLOAT,
    "certified": _BOOL, "passing": _BOOL, "closed": _BOOL,
    "tier": _STR, "outcome": _STR, "source": _STR, "relaxation": _STR, "solver": _STR, "status": _STR, "treatment": _STR,
    "controls": _INTS, "sizes": _INTS,
}
SCHEMA = pa.schema([("level", pa.int8()), ("kind", _STR), ("t", _FLOAT), ("pid", pa.int32())]
                   + [(k, v) for k, v in FIELD_TYPES.items()])


@dataclass(frozen=True)
class Event:
    level: int
    kind: str
    t: float                           # seconds since reporting began
    pid: int
    tags: Mapping
    fields: Mapping


@dataclass
class _State:
    level: int                         # the deepest level a sink takes
    sinks: List[Tuple[int, Callable[[Event], None]]]
    t0: float
    clarabel: bool
    tags: Dict = field(default_factory=dict)


_STATE: Optional[_State] = None


def current() -> Optional[_State]:
    return _STATE


def wants(level: int) -> bool:
    """Whether an event of ``level`` goes anywhere."""
    return _STATE is not None and level <= _STATE.level


def clarabel_log() -> bool:
    """Whether Clarabel prints its iteration log (verbose 4)."""
    return _STATE is not None and _STATE.clarabel


def emit(kind: str, **fields) -> None:
    """Report an event of ``kind`` with its declared ``fields``."""
    spec = KINDS.get(kind)
    if spec is None:
        raise ValueError(f"no event kind {kind!r}; the kinds are {', '.join(KINDS)}.")
    level, names = spec
    bad = set(fields) - set(names)
    if bad:
        raise ValueError(f"undeclared field(s) {sorted(bad)} for a {kind!r} event; it has {names}.")
    if _STATE is None or level > _STATE.level:
        return
    ev = Event(level, kind, time.perf_counter() - _STATE.t0, os.getpid(), dict(_STATE.tags), fields)
    for lv, sink in _STATE.sinks:
        if level <= lv:
            sink(ev)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, Fraction):
        v = float(v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


def _tag(tags) -> str:
    T = tags.get("treatment")
    return "" if T is None else " T=" + ",".join(str(x) for x in T)


class Printer:
    """One line an event on ``stream``, indented by level and, for a node, depth."""

    def __init__(self, stream=None):
        self.stream = stream

    def __call__(self, e: Event) -> None:
        indent = "  " * (e.level - 1 + e.fields.get("depth", 0) if e.kind == "node" else e.level - 1)
        body = " ".join(f"{k}={_fmt(v)}" for k, v in e.fields.items())
        s = self.stream if self.stream is not None else sys.stderr
        s.write(f"[{e.t:9.2f}s]{_tag(e.tags)} {indent}{e.kind} {body}\n")
        s.flush()


def _cell(name, v):
    if v is None:
        return None
    if name == "treatment":
        return ",".join(str(int(x)) for x in v)
    if name in ("controls", "sizes"):
        return [int(x) for x in v]
    t = FIELD_TYPES[name]
    if t == _INT:
        return int(v)
    if t == _FLOAT:
        return float(v)
    if t == _BOOL:
        return bool(v)
    return str(v)


class ParquetLog:
    """Events buffered per process and written as parquet files under ``path``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.run = f"{time.time_ns():x}"
        self.pid = os.getpid()
        self.rows: List[dict] = []
        self.seq = 0
        self.last = time.perf_counter()

    def __call__(self, e: Event) -> None:
        if os.getpid() != self.pid:     # pragma: no cover - a forked worker, which coverage does not trace; the parent writes its own rows
            self.pid, self.rows, self.seq = os.getpid(), [], 0
        row = {"level": e.level, "kind": e.kind, "t": float(e.t), "pid": e.pid}
        for k, v in e.fields.items():
            row[k] = _cell(k, v)
        if row.get("treatment") is None and "treatment" in e.tags:
            row["treatment"] = _cell("treatment", e.tags["treatment"])
        self.rows.append(row)
        if len(self.rows) >= FLUSH_EVERY or time.perf_counter() - self.last >= FLUSH_SECONDS:
            self.flush()

    def flush(self) -> None:
        if os.getpid() != self.pid:     # pragma: no cover - a forked worker with nothing of its own yet, untraced
            self.pid, self.rows, self.seq = os.getpid(), [], 0
            return
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=SCHEMA)
        pq.write_table(table, self.path / f"part-{self.pid}-{self.run}-{self.seq:05d}.parquet")
        self.seq += 1
        self.rows = []
        self.last = time.perf_counter()


def read_log(path):
    """Every event of a parquet log, as a pandas DataFrame in the order each process
    wrote them (one column per declared field, null where a kind has none)."""
    parts = sorted(Path(path).glob("part-*.parquet"))
    if not parts:
        return SCHEMA.empty_table().to_pandas()
    return pa.concat_tables([pq.read_table(p, schema=SCHEMA) for p in parts]).to_pandas()


def _flush_logs():
    if _STATE is not None:
        for _, sink in _STATE.sinks:
            if isinstance(sink, ParquetLog):
                sink.flush()


@contextmanager
def reporting(verbose: int = 0, progress: Optional[Callable[[Event], None]] = None, log=None,
              log_level: int = 3, stream=None):
    """Route the events of the code inside to stderr (``verbose`` 1-4), to
    ``progress`` (up to ``verbose``, or 3 when ``verbose`` is 0) and to a parquet log
    at ``log`` (up to ``log_level``). With none of the three it changes nothing, so a
    call inside reports to its caller's reporter."""
    global _STATE
    if type(verbose) is not int or not 0 <= verbose <= 4:
        raise MlsynthConfigError(f"verbose is a level from 0 to 4, an int, not {verbose!r}.")
    if progress is not None and not callable(progress):
        raise MlsynthConfigError(f"progress is a function called with each event, not {progress!r}.")
    if type(log_level) is not int or not 1 <= log_level <= 3:
        raise MlsynthConfigError(f"log_level is a level from 1 to 3, an int, not {log_level!r}.")
    if log is not None and Path(log).exists() and not Path(log).is_dir():
        raise MlsynthConfigError(f"log is a directory for the parquet files; {str(log)!r} is a file.")
    if not verbose and progress is None and log is None:
        yield
        return
    sinks: List[Tuple[int, Callable[[Event], None]]] = []
    if verbose:
        sinks.append((min(verbose, 3), Printer(stream)))
    if progress is not None:
        sinks.append((min(verbose, 3) if verbose else 3, progress))
    if log is not None:
        Path(log).mkdir(parents=True, exist_ok=True)
        sinks.append((log_level, ParquetLog(Path(log))))
    before = _STATE
    _STATE = _State(level=max(lv for lv, _ in sinks), sinks=sinks, t0=time.perf_counter(), clarabel=verbose >= 4)
    try:
        yield
    finally:
        _flush_logs()
        _STATE = before


@contextmanager
def tagged(**tags):
    """Tag every event inside with ``tags``; a log flushes as the block ends."""
    if _STATE is None:
        yield
        return
    old = _STATE.tags
    _STATE.tags = {**old, **tags}
    try:
        yield
    finally:
        _flush_logs()
        _STATE.tags = old
