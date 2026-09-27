"""Convex solvers for synthetic-control weight problems.

These are general numerics, not a method. Each solves a small constrained
least-squares program that many estimators in this library need, and none of
them knows what an estimator is:

* :mod:`~mlsynth.utils.solvers.active_set` -- exact simplex least squares by a
  primal active set, the workhorse behind most weight fits here.
* :mod:`~mlsynth.utils.solvers.minnorm` -- the same program in Gram form,
  solved for a whole batch at once (Wolfe's min-norm point).
* :mod:`~mlsynth.utils.solvers.simplex` -- self-contained simplex least
  squares and the projection primitives.
* :mod:`~mlsynth.utils.solvers.nnls` -- Lawson-Hanson non-negative least
  squares, with the scipy-version selector.
* :mod:`~mlsynth.utils.solvers.accelerate` -- first-order warm starts, speed
  only; removing it changes no answer.
* :mod:`~mlsynth.utils.solvers.ridge_augment` -- ridge augmentation and its
  simplex QP (Augmented SCM).

They lived under ``bilevel`` because the bilevel V-optimisation was the first
caller. It was never the only one, and the gap has widened: at the time of this
move 56 modules import ``active_set``, 28 ``minnorm``, 24 ``simplex`` and 22
``ridge_augment``, and none of the six reaches for the V machinery
(``engine``, ``regression_v``, ``mscmt``, ``determine_v``, ``stages``,
``structure``). Their only dependency outside this package is
:mod:`mlsynth.exceptions`.

The inversion was concrete: the typed layer in :mod:`mlsynth.utils.weights`,
meant to be this library's general interface for weight solving, imported from
``bilevel`` -- a general interface depending on one estimator family's package.

``bilevel/__init__.py`` no longer re-exports the moved symbols. That
re-export was a shim by another name, and it is what kept the misfiling
invisible at the call site.
"""
