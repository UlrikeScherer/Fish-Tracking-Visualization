"""Smoke tests for the entropy plotting consumers, on synthetic camera-days written to a temp root."""

import pathlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import fishproviz.config as config
from fishproviz.metrics.compute_primitives import entropy_for_chunk
from fishproviz.utils.transformation import _area_functions, clear_transformation_cache
from fishproviz.visualizations import entropy_plots
from fishproviz.visualizations.activity_plotting import plot_metric_figure_for_days

_REPO = pathlib.Path(__file__).resolve().parent.parent
_TEST_DATA = _REPO / "test_data"
_CAM = "23442333"  # has an area polygon in test_data/area_config
_FISH = f"{_CAM}_front"
_BATCH_SIZE = 20
_HEADER = "FRAME;MillisecsByFPS;objectName;valid;id;coordinateUnit;x;y;rad;deg;xpx;ypx;time;timeString"


def _day_name(index: int) -> str:
    return f"202111{index + 1:02d}_060000"


def _day_positions(index: int) -> list[tuple[int, int]]:
    """Inside positions per day, plus an error row, an out-of-range row and one more inside row."""
    rng = np.random.default_rng(index)
    inside = [(int(x), int(y)) for x, y in rng.integers(800, 2200, size=(_BATCH_SIZE - 3, 2))]
    return [*inside, (0, 0), (9000, 9000), (1000 + index, 1500)]


def _write_day(root: pathlib.Path, index: int) -> None:
    day = _day_name(index)
    day_dir = root / _CAM / f"{day}.{_CAM}"
    day_dir.mkdir(parents=True)
    rows = [f"{frame};{frame * 40.0:.6f};0;true;2;cm;0;0;1.57;90.0;{x};{y};{1636805810000 + frame * 40};Sat Nov 13 12:16:50 2021" for frame, (x, y) in enumerate(_day_positions(index))]
    header = ["# Source name: synthetic", "# Source FPS: 25.", "# Generation time: 2021-11-13T12:16:50.000", _HEADER]
    (day_dir / f"{_CAM}_{day}.{_CAM}_000000_2021-11-13T12_16_50_3484.csv").write_text("\n".join([*header, *rows]) + "\n")


class EntropyPlotsCase(unittest.TestCase):
    n_days = 0

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        front, back = root / "front", root / "back"
        back.mkdir()
        for index in range(self.n_days):
            _write_day(front, index)
        (root / "config_data").mkdir()
        self.vis = root / "vis"
        saved_paths = {name: getattr(config, name) for name in ("CONFIG_DATA", "RESULTS_PATH", "err_file", "VIS_DIR")}
        self.addCleanup(lambda: [setattr(config, name, value) for name, value in saved_paths.items()])
        patcher = patch.multiple(
            config,
            dir_front=str(front),
            dir_back=str(back),
            area_front=str(_TEST_DATA / "area_config" / "areas_front"),
            area_back=str(_TEST_DATA / "area_config" / "areas_back"),
            CONFIG_DATA=str(root / "config_data"),
            RESULTS_PATH=str(root / "results"),
            VIS_DIR=str(self.vis),
            err_file=str(root / "log_error.csv"),
            BATCH_SIZE=_BATCH_SIZE,
            MIN_BATCH_IDX=0,
            MAX_BATCH_IDX=0,
            FRAMES_PER_SECOND=5,
            AREA_FILTER=0,
            DIRT_FILTER=0,
            ENTROPY_BINS=18,
            ENTROPY_MARGIN_PX=50,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        clear_transformation_cache()
        self.addCleanup(clear_transformation_cache)

    def _pngs(self) -> list[str]:
        return sorted(path.name for path in (self.vis / entropy_plots.ENTROPY_DIR).glob("*.png"))


class TestDensityPartialWeek(EntropyPlotsCase):
    n_days = 5

    def test_partial_week_and_overall(self) -> None:
        entropy_plots.entropy_density_main()
        self.assertEqual(self._pngs(), [f"{_FISH}_overall_entropy_density.png", f"{_FISH}_week_1_5days_entropy_density.png"])


class TestDensityWeeks(EntropyPlotsCase):
    n_days = 9

    def test_full_week_short_week_and_overall(self) -> None:
        entropy_plots.entropy_density_main()
        self.assertEqual(
            self._pngs(),
            [f"{_FISH}_overall_entropy_density.png", f"{_FISH}_week_1_entropy_density.png", f"{_FISH}_week_2_2days_entropy_density.png"],
        )

    def test_overall_entropy_pools_per_day_filtered_points(self) -> None:
        area = _area_functions()(_FISH)
        pooled = np.array([point for index in range(self.n_days) for point in _day_positions(index) if point != (0, 0)], dtype=np.float64)
        expected = entropy_for_chunk(pooled, area)
        self.assertGreater(expected[2], 0)
        per_day = [entropy_plots.valid_points_of_day(_CAM, _day_name(i), _FISH, (_FISH, area), False) for i in range(self.n_days)]
        np.testing.assert_array_equal(np.concatenate(per_day), pooled)
        with patch.object(entropy_plots, "draw_density_entropy") as draw:
            entropy_plots.entropy_density_main()
        overall = draw.call_args_list[-1].kwargs
        self.assertEqual(overall["timewindow"], "overall")
        self.assertAlmostEqual(overall["entropy_value"], expected[0])
        self.assertEqual((overall["n_valid"], overall["n_out_of_range"]), expected[1:])


class TestNoValidPoints(EntropyPlotsCase):
    n_days = 1

    def test_raises_naming_the_fish(self) -> None:
        with patch.object(entropy_plots, "valid_points_of_day", return_value=np.empty((0, 2))), self.assertRaisesRegex(ValueError, _FISH):
            entropy_plots.entropy_density_main()


class TestEntropyDayFigure(EntropyPlotsCase):
    def test_entropy_figure_is_written(self) -> None:
        results = pathlib.Path(config.RESULTS_PATH) / "entropy"
        results.mkdir(parents=True)
        table = pd.DataFrame({"cam_pos": [_FISH] * 3, "day": [_day_name(i) for i in range(3)], "entropy": [1.0, 1.5, 2.0], "num_valid_points": [10, 10, 10], "num_out_of_range": [0, 1, 0]})
        table.to_csv(results / "day_entropy.csv", sep=config.sep)
        plot_metric_figure_for_days("entropy")
        self.assertTrue((self.vis / "entropy.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
