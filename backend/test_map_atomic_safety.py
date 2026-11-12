"""Stage 3 tests for atomic HTML map generation safety across all 3 regions."""

from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from backend import (
    create_himalayas_risk_map,
    create_northeast_risk_map,
    create_western_ghats_polygon_risk_map,
    gis_refresh_utils,
)
from backend.gis_refresh_utils import atomic_save_map_html, parse_iso8601_utc


class TestAtomicMapOutputSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_map_atomic_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_successful_atomic_replacement_uses_pid_unique_same_dir_temp(self) -> None:
        target = self.tmp_dir / "regional_map.html"
        target.write_text("<html>OLD_VALID_MAP</html>", encoding="utf-8")

        observed_calls = []
        real_replace = os.replace

        def spy_replace(src, dst):
            src_p = Path(src)
            dst_p = Path(dst)
            observed_calls.append((src_p, dst_p, src_p.exists(), src_p.stat().st_size))
            return real_replace(src, dst)

        with patch("backend.gis_refresh_utils.os.replace", side_effect=spy_replace):
            size = atomic_save_map_html(
                target,
                lambda tmp_p: tmp_p.write_text("<html>NEW_VALID_MAP</html>", encoding="utf-8"),
            )

        self.assertEqual(len(observed_calls), 1)
        src_p, dst_p, src_existed, src_size = observed_calls[0]
        self.assertEqual(dst_p, target)
        self.assertEqual(src_p.parent, target.parent)
        self.assertIn(f".{os.getpid()}.", src_p.name)
        self.assertTrue(src_existed)
        self.assertGreater(src_size, 0)
        self.assertEqual(size, target.stat().st_size)
        self.assertEqual(target.read_text(encoding="utf-8"), "<html>NEW_VALID_MAP</html>")
        self.assertEqual([p.name for p in self.tmp_dir.iterdir()], ["regional_map.html"])

    def test_02_previous_final_map_preserved_on_generation_or_replace_failure(self) -> None:
        target = self.tmp_dir / "preserved_map.html"
        sentinel = "<html>PREVIOUS_FINAL_MAP</html>"
        target.write_text(sentinel, encoding="utf-8")

        # Case A: writer raises midway after writing partial data
        def partial_then_fail(tmp_p: Path) -> None:
            tmp_p.write_text("<html>PARTIAL", encoding="utf-8")
            raise RuntimeError("Simulated HTML rendering crash")

        with self.assertRaises(RuntimeError):
            atomic_save_map_html(target, partial_then_fail)

        self.assertEqual(target.read_text(encoding="utf-8"), sentinel)
        self.assertEqual([p.name for p in self.tmp_dir.iterdir()], ["preserved_map.html"])

        # Case B: os.replace raises after successful temp write
        with patch("backend.gis_refresh_utils.os.replace", side_effect=OSError("Simulated replace failure")):
            with self.assertRaises(OSError):
                atomic_save_map_html(
                    target,
                    lambda tmp_p: tmp_p.write_text("<html>SHOULD_NOT_REPLACE</html>", encoding="utf-8"),
                )

        self.assertEqual(target.read_text(encoding="utf-8"), sentinel)
        self.assertEqual([p.name for p in self.tmp_dir.iterdir()], ["preserved_map.html"])

    def test_03_empty_or_missing_temporary_output_rejected(self) -> None:
        target = self.tmp_dir / "non_empty_only.html"
        sentinel = "<html>KEEP_ME</html>"
        target.write_text(sentinel, encoding="utf-8")

        replace_calls = []

        def spy_replace(src, dst):
            replace_calls.append((src, dst))
            return os.replace(src, dst)

        with patch("backend.gis_refresh_utils.os.replace", side_effect=spy_replace):
            # Empty 0-byte temp file
            with self.assertRaises(RuntimeError) as ctx_empty:
                atomic_save_map_html(target, lambda tmp_p: tmp_p.write_bytes(b""))
            self.assertIn("empty", str(ctx_empty.exception).lower())

            # Writer never creates temp file
            with self.assertRaises(RuntimeError) as ctx_missing:
                atomic_save_map_html(target, lambda tmp_p: None)
            self.assertIn("does not exist", str(ctx_missing.exception).lower())

        self.assertEqual(replace_calls, [])
        self.assertEqual(target.read_text(encoding="utf-8"), sentinel)
        self.assertEqual([p.name for p in self.tmp_dir.iterdir()], ["non_empty_only.html"])

    def test_04_temp_cleanup_preserves_other_process_temp_files_and_uses_unique_names(self) -> None:
        target = self.tmp_dir / "map.html"
        target.write_text("<html>EXISTING</html>", encoding="utf-8")
        other_pid_temp = self.tmp_dir / ".map.html.tmp.999999.deadbeef"
        other_pid_temp.write_text("other process temp", encoding="utf-8")

        created_names = []

        def record_and_fail(tmp_p: Path) -> None:
            created_names.append(tmp_p.name)
            tmp_p.write_text("half written", encoding="utf-8")
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            atomic_save_map_html(target, record_and_fail)
        with self.assertRaises(RuntimeError):
            atomic_save_map_html(target, record_and_fail)

        self.assertEqual(len(created_names), 2)
        self.assertNotEqual(created_names[0], created_names[1])
        self.assertTrue(all(f".{os.getpid()}." in name for name in created_names))
        self.assertTrue(other_pid_temp.exists())
        self.assertEqual(other_pid_temp.read_text(encoding="utf-8"), "other process temp")
        for name in created_names:
            self.assertFalse((self.tmp_dir / name).exists())

    @staticmethod
    def _fresh_now_from_metadata(meta_path: Path):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        obs_end = parse_iso8601_utc(meta["gpm_observation_metadata"]["observation_end_utc"])
        assert obs_end is not None
        return obs_end + timedelta(hours=1)

    def test_05_all_three_regional_generators_use_safe_atomic_write_and_preserve_on_failure(self) -> None:
        real_atomic_save = gis_refresh_utils.atomic_save_map_html
        real_replace = os.replace

        regions = [
            ("himalayas", create_himalayas_risk_map, "generate_himalayas_dynamic_risk_map"),
            ("northeast", create_northeast_risk_map, "create_northeast_dynamic_risk_map"),
            ("western_ghats", create_western_ghats_polygon_risk_map, "generate_western_ghats_polygon_risk_map"),
        ]

        for rkey, mod, fn_name in regions:
            with self.subTest(region=rkey):
                if not mod.DEFAULT_INPUT_CSV.exists() or not mod.DEFAULT_INPUT_METADATA.exists():
                    self.skipTest(f"Missing real input artifacts for {rkey}")

                out_dir = self.tmp_dir / rkey
                out_dir.mkdir(parents=True, exist_ok=True)
                out_html = out_dir / f"{rkey}_map.html"
                out_pts = out_dir / f"{rkey}_points.geojson"
                out_regs = out_dir / f"{rkey}_regions.geojson"
                out_cells = out_dir / f"{rkey}_cells.geojson"
                out_meta = out_dir / f"{rkey}_map_meta.json"
                out_val = out_dir / f"{rkey}_val_report.json"

                now_dt = self._fresh_now_from_metadata(mod.DEFAULT_INPUT_METADATA)

                common_kwargs = {
                    "input_csv_path": mod.DEFAULT_INPUT_CSV,
                    "input_metadata_path": mod.DEFAULT_INPUT_METADATA,
                    "output_html_path": out_html,
                    "output_geojson_path": out_pts,
                    "output_regions_geojson_path": out_regs,
                    "output_cells_geojson_path": out_cells,
                    "output_metadata_path": out_meta,
                    "validation_report_path": out_val,
                    "now_utc": now_dt,
                    "overwrite": True,
                }
                if rkey == "northeast":
                    common_kwargs["sync_outputs_dir"] = None

                # 1. Verify success path uses atomic_save_map_html + same-dir PID temp + os.replace
                atomic_calls = []
                replace_calls = []

                def spy_atomic_save(target, writer_fn):
                    atomic_calls.append(Path(target))
                    return real_atomic_save(target, writer_fn)

                def spy_replace(src, dst):
                    if Path(dst) == out_html:
                        replace_calls.append((Path(src), Path(dst)))
                    return real_replace(src, dst)

                with (
                    patch.object(mod, "atomic_save_map_html", side_effect=spy_atomic_save),
                    patch("backend.gis_refresh_utils.os.replace", side_effect=spy_replace),
                ):
                    res = getattr(mod, fn_name)(**common_kwargs)

                self.assertIn(res.get("status"), ("SUCCESS", "PASSED"))
                self.assertEqual(atomic_calls, [out_html])
                self.assertEqual(len(replace_calls), 1)
                tmp_src, final_dst = replace_calls[0]
                self.assertEqual(final_dst, out_html)
                self.assertEqual(tmp_src.parent, out_html.parent)
                self.assertIn(f".{os.getpid()}.", tmp_src.name)
                self.assertFalse(tmp_src.exists())
                self.assertTrue(out_html.exists())
                self.assertGreater(out_html.stat().st_size, 100_000)

                # 2. Verify previous final HTML is preserved and owned temp cleaned up on failure
                fail_dir = self.tmp_dir / f"{rkey}_fail"
                fail_dir.mkdir(parents=True, exist_ok=True)
                fail_html = fail_dir / f"{rkey}_map.html"
                sentinel = f"<html>SENTINEL_{rkey.upper()}_VALID_MAP</html>"
                fail_html.write_text(sentinel, encoding="utf-8")

                fail_kwargs = dict(common_kwargs)
                fail_kwargs.update({
                    "output_html_path": fail_html,
                    "output_geojson_path": fail_dir / f"{rkey}_points.geojson",
                    "output_regions_geojson_path": fail_dir / f"{rkey}_regions.geojson",
                    "output_cells_geojson_path": fail_dir / f"{rkey}_cells.geojson",
                    "output_metadata_path": fail_dir / f"{rkey}_map_meta.json",
                    "validation_report_path": fail_dir / f"{rkey}_val_report.json",
                })

                def fail_on_html_replace(src, dst):
                    if Path(dst) == fail_html:
                        raise OSError(f"Simulated HTML replace error for {rkey}")
                    return real_replace(src, dst)

                with patch("backend.gis_refresh_utils.os.replace", side_effect=fail_on_html_replace):
                    with self.assertRaises(OSError):
                        getattr(mod, fn_name)(**fail_kwargs)

                self.assertEqual(fail_html.read_text(encoding="utf-8"), sentinel)
                leftover_tmps = [p for p in fail_dir.iterdir() if ".tmp." in p.name]
                self.assertEqual(leftover_tmps, [])


if __name__ == "__main__":
    unittest.main()
