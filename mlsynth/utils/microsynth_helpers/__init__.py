"""MicroSynth helper subpackage.

Robbins-Davenport (2021) user-level balancing synthetic control.
Solves a constrained QP for non-negative simplex weights on the
control population that exactly balance covariate moments against
the treated group's moments. Operates at individual-user scale
where the donor pool is the entire untreated population.

Module layout:

* :mod:`setup` -- long DataFrame -> (X_T, X_C, Y_T, Y_C, user names).
* :mod:`dual_solver` -- L-BFGS-B dual ascent + KKT primal recovery.
* :mod:`diagnostics` -- SMD-before/after, ESS, weight concentration,
  feasibility flag.
* :mod:`inference` -- paired stratified bootstrap CI (default).
* :mod:`plotter` -- love plot + lift trajectory.
* :mod:`structures` -- typed dataclasses for ``MicroSynth.fit()``.
"""
