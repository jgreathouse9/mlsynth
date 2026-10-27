import numpy as np
import pandas as pd
import pytest

from gppmta import EventSpace, GraphicalPointProcess, attribute, channel_shares, paths_from_frame

SPACE = EventSpace(
    conversion="purchase",
    customer=("search_click",),
    firm=("email_sent", "display_imp"),
    channels={"email": ("email_sent",), "search": ("search_click",), "display": ("display_imp",)},
)


def test_layout():
    assert SPACE.names == ("purchase", "search_click", "email_sent", "display_imp")
    assert (SPACE.p, SPACE.q) == (4, 2)
    assert SPACE.channel_types() == {"email": [2], "search": [1], "display": [3]}


def test_from_frame_shifts_origin_and_maps_labels():
    df = pd.DataFrame({
        "path_id": ["b", "a", "a", "a", "b"],
        "time": [5.0, 13.0, 10.0, 11.5, 9.0],
        "event": ["email_sent", "purchase", "display_imp", "search_click", "purchase"],
    })
    paths, ids = paths_from_frame(df, SPACE)
    assert ids == ["a", "b"]
    np.testing.assert_allclose(paths[0].times, [0.0, 1.5, 3.0])
    np.testing.assert_array_equal(paths[0].types, [3, 1, 0])
    assert paths[0].T == 3.0 and paths[1].T == 4.0


def test_end_to_end_on_a_frame(rng):
    from gppmta import simulate

    alpha = np.zeros((4, 4))
    alpha[2, 1] = 0.3   # email -> search click
    alpha[1, 0] = 0.4   # search click -> purchase
    alpha[3, 0] = 0.05  # display -> purchase
    sim = simulate([0.005, 0.02, 0.05, 0.1], alpha, 0.5, 100.0, 2000, SPACE.channel_types(), seed=4)
    rows = [(j, t, SPACE.names[e]) for j, p in enumerate(sim.paths) for t, e in zip(p.times, p.types)]
    df = pd.DataFrame(rows, columns=["path_id", "time", "event"])
    df["end"] = 100.0
    paths, ids = paths_from_frame(df, SPACE, end_col="end", origin="zero")
    model = GraphicalPointProcess(decay=0.5, gamma="cv").fit(paths, SPACE.p, SPACE.q)
    assert {(2, 1), (1, 0), (3, 0)} <= set(model.params_.edges)
    table = attribute(model.params_, paths, SPACE.channel_types(), ids)
    shares = channel_shares(table, "tre")
    truth = pd.Series(sim.ccc_shares)
    assert (shares - truth[shares.index]).abs().max() < 0.03


def test_bad_frames_raise():
    df = pd.DataFrame({"path_id": [1], "time": [0.0], "event": ["tv_spot"]})
    with pytest.raises(ValueError, match="not in the event space"):
        paths_from_frame(df, SPACE)
    with pytest.raises(ValueError):
        EventSpace("purchase", ("x",), ("y",), {"c": ("purchase",)})  # conversion in a channel
    with pytest.raises(ValueError):
        EventSpace("purchase", ("x",), ("x",))  # duplicate names
