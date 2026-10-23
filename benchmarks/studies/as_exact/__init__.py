"""A NumPy/SciPy-only solver for the Abadie-Shang exact synthetic control, and its speed
against the authors' code.

Abadie and Shang (2026), "Synthetic Control: An Exact Geometric Formulation"
(arXiv:2610.26576), replace the bilevel choice of predictor weights with a search over
the faces of the donor hull. Their replication code solves the small programs that
search needs with cvxpy and SciPy's linear programming. This study reimplements the
exterior case (the treated unit outside the donor hull) without cvxpy and times both
on Proposition 99. See README.md.
"""
