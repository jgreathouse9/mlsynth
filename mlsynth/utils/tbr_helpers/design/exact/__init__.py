"""The certified (exact) TBR design search (#965).

``treatment.certify_treatment`` returns the treatment and control groups that
minimise the library's relative key over every design the rules allow, with a
certificate that no allowed design scores lower: floors on each treatment group,
an RSS descent for the incumbents, and a slope-free branch and bound over each
group's control sets, bounded by a semidefinite relaxation with triangle and
pentagonal cuts. ``fastfloor``, ``fastqp``, ``fastpent`` and ``fastprice`` are
compiled kernels, used where numba is installed (the ``fast`` extra) and replaced
by the NumPy / SciPy path where it is not.

The search was built and measured in ``benchmarks/studies/tbr_bnb/``, and its tests
live there. References in these modules to the README, to ``results/*.log`` and to
``test_*.py`` files are to that directory."""
