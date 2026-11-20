"""Wall distance output: averaged rows with the outside count, raw rows per loaded frame, and their agreement."""

import unittest

import numpy as np
import pandas as pd

import fishproviz.metrics as metrics
from fishproviz.metrics.results_to_csv import DISTANCE_TO_WALL, DISTANCE_TO_WALL_COLUMNS
from fishproviz.utils.transformation import px2cm
from tests.test_step_metrics import StepMetricTestCase, _day_frames
from tests.test_wall_kernel import CASES, PENTAGON

_RAW_COLUMNS = ["frame", "xpx", "ypx", DISTANCE_TO_WALL, "excluded", "outside"]


def _raw_per_bin(raw: pd.DataFrame, bin_starts: pd.Index) -> pd.DataFrame:
    """Mean, population std, count and outside count of the non-excluded raw rows of every bin."""
    included = raw[~raw["excluded"]]
    bins = np.searchsorted(bin_starts.to_numpy(), included["frame"].to_numpy(), side="right") - 1
    grouped = included.groupby(bin_starts[bins])
    return pd.DataFrame(
        {
            "mean": grouped[DISTANCE_TO_WALL].mean(),
            "std": grouped[DISTANCE_TO_WALL].std(ddof=0),
            "n": grouped[DISTANCE_TO_WALL].size(),
            "outside": grouped["outside"].sum(),
        }
    ).reindex(bin_starts)


class TestWallDistanceOutput(StepMetricTestCase):
    def test_raw_matches_averaged(self) -> None:
        for kwargs in (dict(all_points=True), dict(time_interval=100)):
            averaged = metrics.distance_to_wall_per_interval(fish_ids=self.fish_ids, **kwargs)["results"]
            raw = metrics.distance_to_wall_per_interval(fish_ids=self.fish_ids, every_point=True, write_to_csv=True, **kwargs)["results"]
            for fish_key, days in raw.items():
                for day, table in days.items():
                    with self.subTest(fish_key=fish_key, day=day, **kwargs):
                        expected = averaged[fish_key][day].set_axis(DISTANCE_TO_WALL_COLUMNS, axis=1)
                        got = _raw_per_bin(table, expected.index)
                        np.testing.assert_allclose(got["mean"], expected["mean"], rtol=0, atol=1e-12)
                        np.testing.assert_allclose(got["std"], expected["std"], rtol=0, atol=1e-12)
                        np.testing.assert_array_equal(got["n"].fillna(0), expected["num_valid_points"])
                        np.testing.assert_array_equal(got["outside"].fillna(0), expected["num_outside"])

    def test_raw_has_one_row_per_loaded_frame(self) -> None:
        raw = metrics.distance_to_wall_per_interval(fish_ids=self.fish_ids, all_points=True, every_point=True, write_to_csv=True)["results"]
        for fish_key, days in raw.items():
            for day, table in days.items():
                with self.subTest(fish_key=fish_key, day=day):
                    self.assertEqual(list(table.columns), _RAW_COLUMNS)
                    np.testing.assert_array_equal(table["frame"].to_numpy(), _day_frames(fish_key, day))
                    not_finite = table[["xpx", "ypx"]].isna().any(axis=1)
                    self.assertTrue(table.loc[not_finite, DISTANCE_TO_WALL].isna().all())
                    self.assertFalse(table.loc[not_finite, "outside"].any())
                    self.assertTrue(table.loc[not_finite, "excluded"].all())
                    csv = self.results / DISTANCE_TO_WALL / "raw" / f"{fish_key}_{day}.csv"
                    self.assertEqual(list(pd.read_csv(csv, nrows=0).columns), _RAW_COLUMNS)


class TestWallDistanceSynthetic(StepMetricTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fish_key = self.fish_keys[0]
        self.points = np.array([point for point, _, _ in CASES.values()], dtype=np.float64)
        self.distances_cm = px2cm(np.array([distance for _, distance, _ in CASES.values()]), fish_key=self.fish_key)
        self.outside = np.array([not inside for _, _, inside in CASES.values()])

    def test_outside_count_with_boundary_inside(self) -> None:
        no_errors = np.zeros(len(self.points), dtype=bool)
        result = metrics.distance_to_wall(self.points, [], no_errors, (self.fish_key, PENTAGON))
        self.assertEqual(result.shape, (1, 4))
        mean, std, n_valid, n_outside = result[0]
        self.assertAlmostEqual(mean, self.distances_cm.mean(), places=12)
        self.assertAlmostEqual(std, self.distances_cm.std(), places=12)
        self.assertEqual((n_valid, n_outside), (len(self.points), 7))
        self.assertEqual(n_outside, self.outside.sum())

    def test_empty_bin(self) -> None:
        errors = np.zeros(len(self.points), dtype=bool)
        errors[5:10] = True
        result = metrics.distance_to_wall(self.points, [5, 10], errors, (self.fish_key, PENTAGON))
        np.testing.assert_array_equal(result[1], [np.nan, np.nan, 0, 0])
        self.assertEqual(result[0, 3] + result[2, 3], self.outside[~errors].sum())

    def test_raw_rows(self) -> None:
        points = np.vstack([self.points, [[np.nan, 3.0], [-5.0, -5.0]]])
        errors = np.zeros(len(points), dtype=bool)
        errors[-2:] = True
        result = metrics.distance_to_wall(points, [], errors, (self.fish_key, PENTAGON), unaveraged=True)
        np.testing.assert_array_equal(result[:, 0], np.arange(len(points)))
        np.testing.assert_allclose(result[:-2, 1], self.distances_cm, rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(result[-2, 1]))
        self.assertAlmostEqual(result[-1, 1], px2cm(5.0 * np.sqrt(2), fish_key=self.fish_key), places=12)
        np.testing.assert_array_equal(result[:, 2], errors)
        np.testing.assert_array_equal(result[:, 3], [*self.outside, False, True])


if __name__ == "__main__":
    unittest.main()
