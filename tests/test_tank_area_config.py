import pathlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fishproviz.config as config
from fishproviz.utils import tank_area_config
from fishproviz.utils.tank_area_config import parse_area_file

# Modelled on test_data/area_config/areas_front/23442333_areas.csv: per-video lines (no 'Last' prefix) and one 'Last#' line.
_REFERENCE = "579,516;494,595;3067,3172;3155,3089"
_POLYGON = "902,75;500,457;3183,3148;3602,2802;3592,49"
_VIDEO_LINE = "D:/DevExp_pipeline/temp_videos/23442333/20210612_140031.23442333_front/23442333_20210612_140031.23442333_000000.mp4#490,473;406,557;2963,3160;3044,3064#6,865;374,481;3036,3181;2637,3563;9,3534#2#0"


def _last_line(polygon: str = _POLYGON, reference: str = _REFERENCE) -> str:
    return f"Last#{reference}#{polygon}#2#0"


class _AreaFileCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)

    def _write(self, name: str, *lines: str, folder: str = "") -> pathlib.Path:
        path = self.root / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\r\n".join(lines) + "\r\n")
        return path

    def _assert_rejected(self, path: pathlib.Path, rule: str) -> None:
        with self.assertRaisesRegex(ValueError, f"{path.name}.*{rule}"):
            parse_area_file(str(path))


class TestParseAreaFile(_AreaFileCase):
    def test_valid_file_parses_to_known_values(self) -> None:
        path = self._write("23442333_areas.csv", _VIDEO_LINE, _last_line(), _VIDEO_LINE)
        area = parse_area_file(str(path))
        self.assertEqual(area.path, str(path))
        self.assertEqual(area.camera, "23442333")
        np.testing.assert_array_equal(area.polygon, [[902, 75], [500, 457], [3183, 3148], [3602, 2802], [3592, 49]])
        np.testing.assert_array_equal(area.reference_points, [[579, 516], [494, 595], [3067, 3172], [3155, 3089]])
        self.assertEqual(area.polygon.dtype, np.float64)
        self.assertEqual(area.reference_points.dtype, np.float64)

    def test_line_containing_last_elsewhere_is_ignored(self) -> None:
        decoy = "D:/Last_recordings/23442333_000000.mp4#0,0;0,1;1,1;1,0#0,0;5,0;5,5;0,5#2#0"
        path = self._write("23442333_areas.csv", decoy, _last_line(), " " + _last_line(polygon="0,0;9,0;9,9;4,12;0,9"))
        np.testing.assert_array_equal(parse_area_file(str(path)).polygon[0], [902, 75])

    def test_line_count_must_be_one(self) -> None:
        self._assert_rejected(self._write("23442333_areas.csv", _VIDEO_LINE), "exactly one line starting with 'Last#', found 0")
        self._assert_rejected(self._write("23442333_areas.csv", _last_line(), _last_line()), "exactly one line starting with 'Last#', found 2")

    def test_polygon_rules(self) -> None:
        cases = {
            "four points": ("902,75;500,457;3183,3148;3602,2802", "expected 5 vertices"),
            "six points": (_POLYGON + ";3000,20", "expected 5 vertices"),
            "repeated vertex": ("902,75;500,457;500,457;3602,2802;3592,49", "consecutive vertices 1 and 2 are equal"),
            "repeated closing vertex": ("902,75;500,457;3183,3148;3602,2802;902,75", "consecutive vertices 4 and 0 are equal"),
            "bow-tie": ("0,0;10,10;10,0;0,10;-5,5", "not simple: edges 0-1 and 2-3 intersect"),
            "non-finite": ("902,75;500,457;nan,3148;3602,2802;3592,49", "must be finite"),
        }
        for label, (polygon, rule) in cases.items():
            with self.subTest(label):
                path = self._write("23442333_areas.csv", _VIDEO_LINE, _last_line(polygon=polygon))
                self._assert_rejected(path, f"camera 23442333.*{rule}")

    def test_non_numeric_coordinate_raises_naming_file(self) -> None:
        self._assert_rejected(self._write("23442333_areas.csv", _last_line(polygon="902,75;500,abc;3183,3148;3602,2802;3592,49")), "malformed polygon")
        self._assert_rejected(self._write("23442333_areas.csv", _last_line(reference="579,516;494,x;3067,3172;3155,3089")), "malformed reference points")

    def test_reference_point_count_must_be_four(self) -> None:
        self._assert_rejected(self._write("23442333_areas.csv", _last_line(reference="579,516;494,595;3067,3172")), "expected 4 finite reference points")


class TestAreaFolders(_AreaFileCase):
    def _point_at(self, front: str, back: str) -> None:
        (self.root / "config_data").mkdir()
        patcher = patch.multiple(config, area_front=str(self.root / front), area_back=str(self.root / back), CONFIG_DATA=str(self.root / "config_data"))
        patcher.start()
        self.addCleanup(patcher.stop)
        keys = patch.object(tank_area_config, "get_camera_pos_keys", return_value=[])
        keys.start()
        self.addCleanup(keys.stop)

    def test_two_files_for_one_camera_side_raise_listing_both(self) -> None:
        second = self._write("22222222_b.csv", _last_line(), folder="front")
        first = self._write("22222222_a.csv", _last_line(), folder="front")
        self._write("22222222_areas.csv", _last_line(), folder="back")
        self._point_at("front", "back")
        with self.assertRaisesRegex(ValueError, f"Two area files for camera 22222222 .*{first.name} and .*{second.name}"):
            tank_area_config.get_areas()
        with self.assertRaisesRegex(ValueError, f"{first.name} and .*{second.name}"):
            tank_area_config.compute_calibrations()

    def test_files_are_processed_in_sorted_order(self) -> None:
        for camera in ("33333333", "11111111", "22222222"):
            self._write(f"{camera}_areas.csv", _last_line(), folder="front")
            self._write(f"{camera}_areas.csv", _last_line(), folder="back")
        self._write("notes_areas.csv", "not an area file", folder="front")
        self._point_at("front", "back")
        expected = [f"{c}_{side}" for side in (config.FRONT, config.BACK) for c in ("11111111", "22222222", "33333333")]
        self.assertEqual(list(tank_area_config.get_areas()), expected)
        self.assertEqual(list(tank_area_config.compute_calibrations()), ["11111111", "22222222", "33333333"])

    def test_calibration_prefers_front_file(self) -> None:
        self._write("11111111_areas.csv", _last_line(reference="0,0;0,0;30,40;30,40"), folder="front")
        self._write("11111111_areas.csv", _last_line(reference="0,0;0,0;60,80;60,80"), folder="back")
        self._write("22222222_areas.csv", _last_line(reference="0,0;0,0;60,80;60,80"), folder="back")
        self._point_at("front", "back")
        calibration = tank_area_config.compute_calibrations()
        self.assertEqual(calibration["11111111"], config.CALIBRATION_DIST_CM / 50)
        self.assertEqual(calibration["22222222"], config.CALIBRATION_DIST_CM / 100)


if __name__ == "__main__":
    unittest.main()
