import numpy as np
import pytest
from mlsynth.utils.helperutils import prenorm

def test_prenorm_vector():
    x = np.array([10, 20, 40])
    result = prenorm(x)
    expected = x / 40 * 100
    np.testing.assert_allclose(result, expected)

def test_prenorm_matrix():
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 10]])
    result = prenorm(X)
    expected = X / np.array([5, 10]) * 100
    np.testing.assert_allclose(result, expected)

def test_prenorm_custom_target():
    x = np.array([5, 10])
    result = prenorm(x, target=200)
    expected = x / 10 * 200
    np.testing.assert_allclose(result, expected)

def test_prenorm_list_input():
    x = [2, 4, 8]
    result = prenorm(x)
    expected = np.array(x) / 8 * 100
    np.testing.assert_allclose(result, expected)

def test_prenorm_division_by_zero():
    x = np.array([1, 0])
    with pytest.raises(ZeroDivisionError):
        prenorm(x)
