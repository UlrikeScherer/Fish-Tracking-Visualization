import dataclasses
import json
import pathlib
import shutil
import tempfile
import unittest

import fishproviz.config as config
from fishproviz.utils import tank_area_config
from fishproviz.utils.transformation import clear_transformation_cache, px2cm

_TEST_DATA = pathlib.Path(__file__).resolve().parent.parent / "test_data"
_PATH_NAMES = [f.name for f in dataclasses.fields(config.RootPaths)]
_VALID = dict(spike_threshold=10.0, frames_per_second=5, batch_size=10000, min_batch_idx=0, max_batch_idx=12, aggregate_every_n=1)


class _ConfigRestoringCase(unittest.TestCase):
    """Snapshots every root-derived config path and restores it afterwards."""

    def setUp(self) -> None:
        self._saved = {name: getattr(config, name) for name in _PATH_NAMES}
        self._tmp = pathlib.Path(tempfile.mkdtemp())
        clear_transformation_cache()

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            setattr(config, name, value)
        clear_transformation_cache()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _point_at_test_data(self, area_front: pathlib.Path, area_back: pathlib.Path) -> None:
        config.dir_front = str(_TEST_DATA / "front")
        config.dir_back = str(_TEST_DATA / "back")
        config.area_front = str(area_front)
        config.area_back = str(area_back)
        config.CONFIG_DATA = str(self._tmp / "config_data")
        pathlib.Path(config.CONFIG_DATA).mkdir()


class TestSetConfigPaths(_ConfigRestoringCase):
    def test_every_derived_path_follows_root(self) -> None:
        config.set_config_paths(str(self._tmp))
        for name in _PATH_NAMES:
            self.assertTrue(getattr(config, name).startswith(str(self._tmp)), f"{name}={getattr(config, name)}")
        self.assertEqual(getattr(config, "DIR_CSV_LOCAL"), str(self._tmp))

    def test_matches_build_root_paths(self) -> None:
        config.set_config_paths(str(self._tmp))
        expected = dataclasses.asdict(config.build_root_paths(str(self._tmp)))
        self.assertEqual({name: getattr(config, name) for name in _PATH_NAMES}, expected)


class TestValidateConfig(unittest.TestCase):
    def test_valid_passes(self) -> None:
        config.validate_config(**_VALID)

    def test_spike_threshold_must_be_positive(self) -> None:
        for bad in (0, -1):
            with self.assertRaisesRegex(ValueError, f"SPIKE_THRESHOLD.*{bad}"):
                config.validate_config(**{**_VALID, "spike_threshold": bad})

    def test_other_keys_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "BATCH_SIZE"):
            config.validate_config(**{**_VALID, "batch_size": 0})
        with self.assertRaisesRegex(ValueError, "MIN_BATCH_IDX"):
            config.validate_config(**{**_VALID, "min_batch_idx": 5, "max_batch_idx": 4})
        with self.assertRaisesRegex(ValueError, "AGGREGATE_EVERY_N"):
            config.validate_config(**{**_VALID, "aggregate_every_n": 0})
        with self.assertRaisesRegex(ValueError, "FRAMES_PER_SECOND"):
            config.validate_config(**{**_VALID, "frames_per_second": 0})


class TestCalibrationAndAreas(_ConfigRestoringCase):
    def test_px2cm_requires_fish_key(self) -> None:
        with self.assertRaises(TypeError):
            px2cm(1.0)  # type: ignore[call-arg]

    def test_calibration_matches_known_values(self) -> None:
        self._point_at_test_data(_TEST_DATA / "area_config" / "areas_front", _TEST_DATA / "area_config" / "areas_back")
        calibration = tank_area_config.get_calibration_functions()
        self.assertEqual(calibration("23442333_front"), 0.022782343535971344)
        self.assertEqual(calibration("23442333_back"), 0.022782343535971344)
        recorded = json.loads((pathlib.Path(config.CONFIG_DATA) / "calibration.json").read_text())
        self.assertEqual(recorded["23442333"], 0.022782343535971344)

    def test_unknown_camera_raises_naming_camera(self) -> None:
        self._point_at_test_data(_TEST_DATA / "area_config" / "areas_front", _TEST_DATA / "area_config" / "areas_back")
        with self.assertRaisesRegex(KeyError, "99999999"):
            tank_area_config.get_calibration_functions()("99999999_front")

    def test_calibration_recomputed_after_area_file_change(self) -> None:
        front, back = self._tmp / "front_areas", self._tmp / "back_areas"
        shutil.copytree(_TEST_DATA / "area_config" / "areas_front", front)
        shutil.copytree(_TEST_DATA / "area_config" / "areas_back", back)
        self._point_at_test_data(front, back)
        before = tank_area_config.get_calibration_functions()("23442333_front")
        target = front / "23442333_areas.csv"
        # Double the first reference point's distance by editing the Last line: scale all coordinates of the reference points by 2
        edited = []
        for line in target.read_text().splitlines():
            if line.startswith("Last"):
                parts = line.split("#")
                parts[1] = ";".join(",".join(str(2 * int(v)) for v in pt.split(",")) for pt in parts[1].split(";"))
                line = "#".join(parts)
            edited.append(line)
        target.write_text("\n".join(edited) + "\n")
        after = tank_area_config.get_calibration_functions()("23442333_front")
        self.assertAlmostEqual(after, before / 2)

    def test_malformed_area_file_raises_naming_file(self) -> None:
        front, back = self._tmp / "front_areas", self._tmp / "back_areas"
        shutil.copytree(_TEST_DATA / "area_config" / "areas_front", front)
        shutil.copytree(_TEST_DATA / "area_config" / "areas_back", back)
        self._point_at_test_data(front, back)
        (front / "23442333_areas.csv").write_text("no relevant line here\n")
        with self.assertRaisesRegex(ValueError, "23442333_areas.csv"):
            tank_area_config.compute_calibrations()
        with self.assertRaisesRegex(ValueError, "23442333_areas.csv"):
            tank_area_config.get_areas()

    def test_empty_area_folder_raises_naming_folder(self) -> None:
        empty, back = self._tmp / "empty_areas", self._tmp / "back_areas"
        empty.mkdir()
        shutil.copytree(_TEST_DATA / "area_config" / "areas_back", back)
        self._point_at_test_data(empty, back)
        with self.assertRaisesRegex(FileNotFoundError, "empty_areas"):
            tank_area_config.compute_calibrations()

    def test_camera_without_area_raises_naming_camera_and_folder(self) -> None:
        front, back = self._tmp / "front_areas", self._tmp / "back_areas"
        shutil.copytree(_TEST_DATA / "area_config" / "areas_front", front)
        shutil.copytree(_TEST_DATA / "area_config" / "areas_back", back)
        self._point_at_test_data(front, back)
        (front / "23442333_areas.csv").unlink()
        with self.assertRaisesRegex(ValueError, f"23442333_front.*{front.name}"):
            tank_area_config.get_areas()


if __name__ == "__main__":
    unittest.main()
