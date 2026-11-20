"""Wall kernel tests: distance to a hand-computed pentagon, the inside test and input validation."""

import math
import unittest

import numpy as np
from matplotlib.path import Path

from fishproviz.methods import distance_to_wall_chunk, points_inside_polygon

# A(0,0) B(12,0) C(12,8) D(6,14) E(0,8); edges AB y=0, BC x=12, CD x+y=20, DE y-x=8, closing EA x=0
PENTAGON: np.ndarray = np.array([[0, 0], [12, 0], [12, 8], [6, 14], [0, 8]], dtype=np.float64)
HALF_SQRT2: float = math.sqrt(2) / 2

# (point, distance, inside)
CASES: dict[str, tuple[tuple[float, float], float, bool]] = {
    "inside near AB": ((6, 1), 1.0, True),
    "inside near BC": ((11, 4), 1.0, True),
    "inside near CD": ((10, 9), HALF_SQRT2, True),
    "inside near DE": ((2, 9), HALF_SQRT2, True),
    "inside near closing EA": ((1, 4), 1.0, True),
    "outside beyond AB": ((6, -3), 3.0, False),
    "outside beyond BC": ((15, 4), 3.0, False),
    "outside beyond CD": ((11, 11), math.sqrt(2), False),
    "outside beyond DE": ((1, 11), math.sqrt(2), False),
    "outside beyond closing EA": ((-2, 5), 2.0, False),
    "outside nearest vertex B": ((15, -4), 5.0, False),
    "outside nearest vertex D": ((6, 17), 3.0, False),
    "on edge AB": ((5, 0), 0.0, True),
    "on diagonal edge CD": ((9, 11), 0.0, True),
    "on closing edge EA": ((0, 5), 0.0, True),
    "on vertex D": ((6, 14), 0.0, True),
    "on closing vertex E": ((0, 8), 0.0, True),
}


def _points() -> np.ndarray:
    return np.array([point for point, _, _ in CASES.values()], dtype=np.float64)


class TestDistanceToWall(unittest.TestCase):
    def test_hand_computed_distances(self) -> None:
        dists = distance_to_wall_chunk(_points(), PENTAGON)
        for (name, (_, expected, _)), got in zip(CASES.items(), dists):
            with self.subTest(name):
                self.assertAlmostEqual(got, expected, delta=1e-12)

    def test_empty_data(self) -> None:
        dists = distance_to_wall_chunk(np.empty((0, 2)), PENTAGON)
        self.assertEqual(dists.shape, (0,))
        self.assertEqual(dists.dtype, np.float64)


class TestPointsInsidePolygon(unittest.TestCase):
    def test_hand_computed_cases(self) -> None:
        inside = points_inside_polygon(_points(), PENTAGON)
        self.assertEqual(inside.dtype, np.bool_)
        for (name, (_, _, expected)), got in zip(CASES.items(), inside):
            with self.subTest(name):
                self.assertIs(bool(got), expected)

    def test_non_finite_point_is_outside(self) -> None:
        data = np.array([[np.nan, 4.0], [6.0, np.nan], [np.inf, 4.0], [6.0, 4.0]])
        np.testing.assert_array_equal(points_inside_polygon(data, PENTAGON), [False, False, False, True])

    def test_agrees_with_matplotlib_off_boundary(self) -> None:
        rng = np.random.default_rng(20261120)
        data = rng.uniform([-3.0, -3.0], [15.0, 17.0], size=(10_000, 2))
        keep = distance_to_wall_chunk(data, PENTAGON) > 1e-9
        expected = Path(PENTAGON).contains_points(data[keep])
        np.testing.assert_array_equal(points_inside_polygon(data[keep], PENTAGON), expected)
        self.assertTrue(0 < expected.sum() < keep.sum())

    def test_empty_data(self) -> None:
        inside = points_inside_polygon(np.empty((0, 2)), PENTAGON)
        self.assertEqual(inside.shape, (0,))
        self.assertEqual(inside.dtype, np.bool_)


class TestValidation(unittest.TestCase):
    BAD_AREAS: dict[str, tuple[np.ndarray, str]] = {
        "three columns": (np.zeros((5, 3)), r"area must have shape \(n_vertices, 2\)"),
        "no vertices": (np.empty((0, 2)), "at least 3 vertices, got 0"),
        "one vertex": (np.array([[0.0, 0.0]]), "at least 3 vertices, got 1"),
        "two vertices": (np.array([[0.0, 0.0], [5.0, 0.0]]), "at least 3 vertices, got 2"),
        "nan vertex": (np.array([[0.0, 0.0], [5.0, 0.0], [np.nan, 5.0]]), "vertex 2 is not finite"),
        "inf vertex": (np.array([[0.0, np.inf], [5.0, 0.0], [5.0, 5.0]]), "vertex 0 is not finite"),
        "repeated vertex": (np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 5.0]]), "zero-length edge: vertex 1 equals vertex 2"),
        "closing pair repeated": (np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 0.0]]), "zero-length edge: vertex 3 equals vertex 0"),
    }

    def test_bad_area_raises_for_both_functions(self) -> None:
        data = np.array([[1.0, 1.0]])
        for func in (distance_to_wall_chunk, points_inside_polygon):
            for name, (area, message) in self.BAD_AREAS.items():
                with self.subTest(func=func.__name__, case=name), self.assertRaisesRegex(ValueError, message):
                    func(data, area)

    def test_bad_data_columns_raise_for_both_functions(self) -> None:
        for func in (distance_to_wall_chunk, points_inside_polygon):
            for data in (np.zeros((4, 3)), np.zeros((0, 3)), np.zeros((4, 1))):
                with self.subTest(func=func.__name__, shape=data.shape), self.assertRaisesRegex(ValueError, r"data must have shape \(n, 2\)"):
                    func(data, PENTAGON)


if __name__ == "__main__":
    unittest.main()
