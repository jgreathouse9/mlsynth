"""Instrument-space projection for the projected-SIV variant.

The projected estimator of Gulek and Vives-i-Bastida (2024,
Section 5.1) replaces the pre-period outcome ``Y_pre`` with its
projection onto the column space of the instrument loadings
``Z = (Z_1, ..., Z_J)'``:

    Y_z,t = Z (Z' Z)^{-1} Z' Y_t

This projection removes idiosyncratic ``epsilon_it`` noise that is
orthogonal to the instrument under the partial validity assumption,
and is used as the alternative design matrix in the projected SC
fit. Trade-off: pre-treatment fit on raw ``Y`` worsens (we lose the
direct ``Y_pre`` matching channel) but the estimator becomes
robust to high-noise regimes.

Notes
-----
The "Z" used here is the time-invariant instrument loading vector
(``Z_i``) implicit in the shift-share structure ``Z_it = Z_i' g_t``.
For panels where the user already supplies the full ``Z_it`` matrix
we recover the loading by averaging the post-period instrument
across the post-treatment columns; for the Syrian/China-shock
applications this matches the paper's "share" component of the
shift-share design.
"""

from __future__ import annotations

import numpy as np

from .structures import SIVInputs


def project_outcome_pre(
    inputs: SIVInputs,
    loading: np.ndarray | None = None,
) -> np.ndarray:
    """Return the instrument-space projection of the pre-period outcome.

    Parameters
    ----------
    inputs : SIVInputs
        Preprocessed panel.
    loading : np.ndarray, optional
        Length-``J`` instrument loading vector ``Z_i``. If ``None``, we
        recover it as the post-period average of ``Z_it``, scaled so
        the entries are comparable to the original instrument: for
        shift-share designs this is exactly the share component.

    Returns
    -------
    np.ndarray
        ``(J, T0)`` matrix giving ``Z (Z' Z)^{-1} Z' Y_pre`` --
        the projection of every column of the pre-period outcome onto
        the span of the loading vector.

    Notes
    -----
    With a single loading direction the projection matrix
    ``P = Z (Z' Z)^{-1} Z'`` is rank 1, so the projected outcome is
    ``(Z' Y_t / Z' Z) Z`` per time period. We implement that closed
    form to avoid forming the ``(J, J)`` projector explicitly.
    """

    if loading is None:
        Z_post = inputs.Z[:, inputs.T0:]
        if Z_post.shape[1] == 0:
            raise ValueError(
                "Cannot infer instrument loading: no post-treatment "
                "periods available for Z."
            )
        loading_vec = Z_post.mean(axis=1)
    else:
        loading_vec = np.asarray(loading, dtype=float).flatten()
        if loading_vec.shape[0] != inputs.J:
            raise ValueError(
                f"loading must have length J={inputs.J}; got "
                f"{loading_vec.shape[0]}."
            )

    Y_pre = inputs.Y[:, :inputs.T0]
    norm_sq = float(loading_vec @ loading_vec)
    if norm_sq <= 1e-12:
        # Degenerate loading; fall back to identity projection.
        return Y_pre.copy()
    coefs = loading_vec @ Y_pre / norm_sq          # length T0
    return np.outer(loading_vec, coefs)             # (J, T0)
