"""
Dynamic NASA GPM IMERG Rainfall Refresh Pipeline for Western Ghats.

Stage 6B Deliverable (Freshness-Corrected & Resilient):
Converts the fixed-window GPM IMERG integration into an automatically refreshable,
freshness-verified rainfall monitoring system for all 14 Western Ghats DEM tiles
(182 historical events + 1,400 background samples = 1,582 points).

Key Capabilities & Freshness Controls:
1. Online Granule Discovery & Ingestion: Queries NASA CMR for latest Early Run V07C granules
   using Earthdata bearer token authentication. Automatically discovers the newest contiguous
   24-hour window from NASA servers rather than defaulting to stale local archives.
2. Local Cache Reuse: Scans existing directories (gpm_imerge_series_12h, gpm_imerge_previous12h,
   gpm_archive) to prevent downloading any granule that already exists locally.
3. Contiguous Sequence Identification: Identifies the latest contiguous sequence of 48
   half-hourly granules (24 hours). If newer granules are partially incomplete or missing,
   identifies the newest genuinely complete contiguous sequence.
4. Configurable Freshness Validation:
   - Computes actual data latency: latency = (execution_time_utc - observation_end_utc).
   - Distinguishes statuses: SUCCESS, STALE_DATA, and INCOMPLETE_DATA.
   - Strict Promotion Rule: Stale or incomplete snapshots are NEVER promoted to
     latest_validated_rainfall.csv.
5. Rolling Accumulation Engine: Calculates 1h, 3h, 6h, 12h, and 24h rolling accumulations
   with verified monotonicity. Never interprets missing measurements as zero rainfall.
6. Spatial Matching: Projects 30m Western Ghats terrain locations to native 0.1° GPM cells.
7. Safe Recovery & Snapshot Preservation: Uses atomic operations (.part staging, atomic snapshot
   promotions), cleans up partial downloads, and never overwrites existing snapshots.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import h5py
import numpy as np
import pandas as pd
import requests

from backend.gis_refresh_utils import (
    atomic_write_json,
    create_timestamped_backup,
)

# Reuse core integration engine from Stage 6A
from backend.extract_western_ghats_gpm_rainfall import (
    WesternGhatsGPMIntegrator,
    load_western_ghats_locations,
    parse_granule_times,
    haversine_distance_km,
    DEFAULT_EVENTS_CSV,
    DEFAULT_BACKGROUND_CSV,
    EXPECTED_TOTAL_COUNT,
    EXPECTED_TILE_COUNT,
    EXPECTED_GRANULES_24H,
    GRANULE_INTERVAL_SECONDS,
    MAX_GRID_MATCH_DISTANCE_KM,
    GPM_PRODUCT_NAME,
    GPM_NATIVE_RESOLUTION,
)


# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================

# Load Earthdata credentials from project root .env
load_dotenv(PROJECT_ROOT / ".env")
EARTHDATA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN")

CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION_CONCEPT_ID = "C2723758340-GES_DISC"  # GPM IMERG Early V07

DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / "data" / "raw" / "satellite" / "gpm_archive"
EXISTING_RAW_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "satellite" / "gpm_imerge_previous12h",
    PROJECT_ROOT / "data" / "raw" / "satellite" / "gpm_imerge_series_12h",
    DEFAULT_ARCHIVE_DIR,
]

DYNAMIC_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "rainfall" / "western_ghats_refresh"
SNAPSHOTS_DIR = DYNAMIC_OUTPUT_DIR / "snapshots"
METADATA_DIR = DYNAMIC_OUTPUT_DIR / "metadata"
LATEST_VALIDATED_CSV = DYNAMIC_OUTPUT_DIR / "latest_validated_rainfall.csv"
LATEST_STATUS_JSON = METADATA_DIR / "latest_refresh_status.json"

# Operational freshness ceiling (IMERG Early Run latency is ~4-6 hours)
DEFAULT_MAX_LATENCY_HOURS = 12.0


# =====================================================================
# HDF5 VALIDATION & UTILITIES
# =====================================================================

def is_valid_hdf5_granule(filepath: Path) -> bool:
    """Verify that an HDF5 granule exists, is not empty, and contains GPM grid datasets."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False
    try:
        with h5py.File(filepath, "r") as hdf:
            if "Grid/precipitation" not in hdf or "Grid/time_bnds" not in hdf:
                return False
            shape = hdf["Grid/precipitation"].shape
            if len(shape) < 3 or shape[1] != 3600 or shape[2] != 1800:
                return False
        return True
    except Exception:
        return False


def clean_cmr_title(title: str) -> str:
    """Extract standard HDF5 filename from NASA CMR title removing any collection prefix."""
    filename = title.split(":", 1)[-1].strip()
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


# =====================================================================
# LOCAL GRANULE REPOSITORY INDEX
# =====================================================================

class LocalGranuleIndex:
    """
    Indexes all locally available GPM IMERG HDF5 granules across all cache/raw folders.
    Prevents duplicate downloads and enables offline refresh capability.
    """

    def __init__(self, search_directories: Sequence[Path]):
        self.search_directories = [d for d in search_directories if d.exists()]
        self.index: Dict[datetime, Dict] = {}
        self.build_index()

    def build_index(self):
        """Scan directories and index valid HDF5 granules by start timestamp."""
        self.index.clear()
        for directory in self.search_directories:
            for filepath in directory.glob("*.HDF5"):
                try:
                    start_dt, end_dt = parse_granule_times(filepath.name)
                except ValueError:
                    continue

                if start_dt in self.index:
                    continue  # Already indexed from another directory

                if is_valid_hdf5_granule(filepath):
                    self.index[start_dt] = {
                        "path": filepath,
                        "filename": filepath.name,
                        "start": start_dt,
                        "end": end_dt,
                        "size_bytes": filepath.stat().st_size,
                    }

    def has_granule(self, start_dt: datetime) -> bool:
        return start_dt in self.index

    def get_granule(self, start_dt: datetime) -> Optional[Dict]:
        return self.index.get(start_dt)

    def all_granules_sorted(self) -> List[Dict]:
        return [self.index[k] for k in sorted(self.index.keys())]

    def add_granule(self, filepath: Path) -> Optional[Dict]:
        if not is_valid_hdf5_granule(filepath):
            return None
        start_dt, end_dt = parse_granule_times(filepath.name)
        item = {
            "path": filepath,
            "filename": filepath.name,
            "start": start_dt,
            "end": end_dt,
            "size_bytes": filepath.stat().st_size,
        }
        self.index[start_dt] = item
        return item


# =====================================================================
# NASA CMR DISCOVERY & DOWNLOADER
# =====================================================================

class GPMGranuleDownloader:
    """
    Manages communication with NASA CMR and Earthdata LP DAAC / GES DISC.
    Downloads missing granules into the archive directory with atomic .part staging.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    ):
        self.token = token or EARTHDATA_TOKEN
        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.last_discovered_cmr_count = 0
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        if self.token:
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "Western-Ghats-Landslide-Refresh/1.0",
            })

    def search_cmr_granules(
        self,
        start_time: datetime,
        end_time: datetime,
        page_size: int = 100,
    ) -> List[Dict]:
        """Query NASA CMR for IMERG granules within the requested UTC window."""
        params = {
            "collection_concept_id": COLLECTION_CONCEPT_ID,
            "temporal": f"{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')},{end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "page_size": page_size,
            "sort_key": "start_date",
        }
        response = self.session.get(CMR_SEARCH_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        entries = data.get("feed", {}).get("entry", [])

        discovered = []
        for entry in entries:
            title = entry.get("title", "")
            if "3IMERG" not in title:
                continue
            filename = clean_cmr_title(title)
            try:
                start_dt, end_dt = parse_granule_times(filename)
            except ValueError:
                continue

            if not (start_time <= start_dt < end_time):
                continue

            # Locate HDF5 download URL
            download_url = None
            for link in entry.get("links", []):
                href = link.get("href", "")
                if href.endswith(".HDF5") and not link.get("inherited", False) and not link.get("rel", "").endswith("/metadata"):
                    download_url = href
                    break

            if download_url:
                discovered.append({
                    "title": title,
                    "filename": filename,
                    "start": start_dt,
                    "end": end_dt,
                    "url": download_url,
                })

        discovered.sort(key=lambda item: item["start"])
        return discovered

    def discover_latest_contiguous_window_online(
        self,
        page_size: int = 150,
    ) -> Tuple[Optional[datetime], Optional[List[Dict]], List[datetime]]:
        """
        Query NASA CMR for recently published granules and find the newest
        contiguous 48-granule (24-hour) sequence available on NASA servers.
        Returns: (candidate_end_time, granules_list, missing_timestamps)
        """
        params = {
            "collection_concept_id": COLLECTION_CONCEPT_ID,
            "page_size": page_size,
            "sort_key": "-start_date",  # Newest first
        }
        response = self.session.get(CMR_SEARCH_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        entries = data.get("feed", {}).get("entry", [])

        granule_map: Dict[datetime, Dict] = {}
        for entry in entries:
            title = entry.get("title", "")
            if "3IMERG" not in title:
                continue
            filename = clean_cmr_title(title)
            try:
                start_dt, end_dt = parse_granule_times(filename)
            except ValueError:
                continue

            download_url = None
            for link in entry.get("links", []):
                href = link.get("href", "")
                if href.endswith(".HDF5") and not link.get("inherited", False) and not link.get("rel", "").endswith("/metadata"):
                    download_url = href
                    break

            if download_url and start_dt not in granule_map:
                granule_map[start_dt] = {
                    "title": title,
                    "filename": filename,
                    "start": start_dt,
                    "end": end_dt,
                    "url": download_url,
                }

        if not granule_map:
            self.last_discovered_cmr_count = 0
            return None, None, []

        self.last_discovered_cmr_count = len(granule_map)
        available_starts = sorted(granule_map.keys())

        # Search from the newest start downwards for a contiguous 48-granule block
        for latest_start in reversed(available_starts):
            candidate_end = latest_start + timedelta(minutes=30)
            expected_starts = [
                candidate_end - timedelta(minutes=30 * (48 - i))
                for i in range(48)
            ]
            missing = [s for s in expected_starts if s not in granule_map]
            if not missing:
                ordered_granules = [granule_map[s] for s in expected_starts]
                return candidate_end, ordered_granules, []

        # If no complete 48-granule block found, return newest candidate end with its missing list
        newest_end = available_starts[-1] + timedelta(minutes=30)
        expected_starts = [newest_end - timedelta(minutes=30 * (48 - i)) for i in range(48)]
        missing = [s for s in expected_starts if s not in granule_map]
        return newest_end, None, missing

    def download_single_granule(self, item: Dict, max_retries: int = 3) -> Path:
        """
        Download a single granule safely with atomic staging (.part file),
        HDF5 validation, and automatic cleanup of corrupted downloads.
        """
        filename = item["filename"]
        destination = self.archive_dir / filename
        temporary_part = self.archive_dir / (filename + ".part")

        if destination.exists() and is_valid_hdf5_granule(destination):
            return destination

        if temporary_part.exists():
            temporary_part.unlink()

        for attempt in range(1, max_retries + 1):
            try:
                with self.session.get(item["url"], stream=True, timeout=(30, 300)) as resp:
                    resp.raise_for_status()
                    with open(temporary_part, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)

                if not is_valid_hdf5_granule(temporary_part):
                    raise RuntimeError("Downloaded file failed HDF5 integrity check.")

                # Atomic replace
                os.replace(temporary_part, destination)
                return destination

            except Exception as exc:
                if temporary_part.exists():
                    temporary_part.unlink()
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts: {exc}")
                time.sleep(1.0 * attempt)

        raise RuntimeError(f"Unexpected exit downloading {filename}")


# =====================================================================
# DYNAMIC REFRESH ENGINE
# =====================================================================

class WesternGhatsDynamicRefreshEngine:
    """
    Coordinates dynamic GPM IMERG rainfall refresh, rolling accumulation updates,
    spatial mapping, metadata recording, and snapshot preservation for Western Ghats.
    """

    def __init__(
        self,
        events_csv: Path = DEFAULT_EVENTS_CSV,
        background_csv: Path = DEFAULT_BACKGROUND_CSV,
        raw_directories: Optional[List[Path]] = None,
        archive_dir: Path = DEFAULT_ARCHIVE_DIR,
        output_dir: Path = DYNAMIC_OUTPUT_DIR,
        max_latency_hours: float = DEFAULT_MAX_LATENCY_HOURS,
        workers: int = 4,
        offline_mode: bool = False,
    ):
        self.events_csv = events_csv
        self.background_csv = background_csv
        self.raw_directories = raw_directories or EXISTING_RAW_DIRS
        self.archive_dir = archive_dir
        self.output_dir = output_dir
        self.max_latency_hours = float(max_latency_hours)
        self.workers = workers
        self.offline_mode = offline_mode

        self.snapshots_dir = output_dir / "snapshots"
        self.metadata_dir = output_dir / "metadata"
        self.latest_validated_csv = output_dir / "latest_validated_rainfall.csv"
        self.latest_status_json = self.metadata_dir / "latest_refresh_status.json"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.local_index = LocalGranuleIndex(self.raw_directories)
        self.downloader = GPMGranuleDownloader(archive_dir=self.archive_dir)

    def find_latest_contiguous_window_locally(
        self,
        target_end_time: Optional[datetime] = None,
    ) -> Tuple[Optional[List[Dict]], Optional[datetime], List[datetime]]:
        """
        Scan locally indexed granules to find a contiguous 48-granule (24-hour) sequence.
        Returns: (granules_list, end_time, missing_timestamps)
        """
        granules = self.local_index.all_granules_sorted()
        if not granules:
            return None, None, []

        granule_map = {g["start"]: g for g in granules}
        available_starts = sorted(granule_map.keys())

        # If target end time specified, check backwards from there
        if target_end_time:
            candidate_end = target_end_time
            candidate_starts = [
                candidate_end - timedelta(minutes=30 * (48 - i))
                for i in range(48)
            ]
            missing = [s for s in candidate_starts if s not in granule_map]
            if not missing:
                return [granule_map[s] for s in candidate_starts], candidate_end, []
            return None, candidate_end, missing

        # Otherwise, search backwards from the latest available granule start
        for latest_start in reversed(available_starts):
            candidate_end = latest_start + timedelta(minutes=30)
            candidate_starts = [
                candidate_end - timedelta(minutes=30 * (48 - i))
                for i in range(48)
            ]
            missing = [s for s in candidate_starts if s not in granule_map]
            if not missing:
                return [granule_map[s] for s in candidate_starts], candidate_end, []

        # If no complete 48-granule sequence exists locally, return missing intervals for latest attempt
        latest_start = available_starts[-1]
        candidate_end = latest_start + timedelta(minutes=30)
        candidate_starts = [candidate_end - timedelta(minutes=30 * (48 - i)) for i in range(48)]
        missing = [s for s in candidate_starts if s not in granule_map]
        return None, candidate_end, missing

    def resolve_target_window(
        self,
        target_end_time: Optional[datetime] = None,
    ) -> Tuple[datetime, Optional[List[Dict]], str]:
        """
        Resolve the target 24-hour observation window:
        - If manual end time specified: uses manual window.
        - If online and token present: queries NASA CMR to find the genuinely newest
          contiguous 48-granule window on NASA servers.
        - If offline: scans local archive directories.
        Returns: (target_end_time, pre_discovered_granules, source_description)
        """
        if target_end_time is not None:
            return target_end_time, None, "MANUAL_TARGET"

        # If online mode and authentication token available: discover via CMR
        if not self.offline_mode and self.downloader.token:
            print("\nQuerying NASA CMR for latest available GPM IMERG Early granules...")
            try:
                candidate_end, online_granules, missing = self.downloader.discover_latest_contiguous_window_online()
                if candidate_end is not None and not missing and online_granules:
                    print(f"Discovered latest contiguous window on CMR ending at: {candidate_end.strftime('%Y-%m-%dT%H:%M:%SZ')}")
                    return candidate_end, online_granules, "CMR_ONLINE"
                elif candidate_end is not None and missing:
                    print(f"Notice: CMR newest window ending at {candidate_end.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                          f"has {len(missing)} missing granules.")
            except Exception as exc:
                print(f"Warning: Online CMR discovery encountered an issue ({exc}). Falling back to local cache.")

        # Fallback to local archive
        print("Examining local granule index for latest contiguous window...")
        local_granules, candidate_end, missing = self.find_latest_contiguous_window_locally()
        if candidate_end is not None and not missing and local_granules:
            return candidate_end, local_granules, "LOCAL_CACHE"

        if candidate_end is not None:
            return candidate_end, None, "LOCAL_INCOMPLETE"

        raise RuntimeError("No GPM IMERG granules available locally or online to establish a 24-hour window.")

    def ensure_granules_for_window(
        self,
        end_time: datetime,
        pre_discovered: Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], List[datetime], int, int]:
        """
        Verify all 48 expected granules for the 24h window ending at `end_time`.
        Downloads missing granules in parallel using atomic staging; reuses cached files.
        Returns: (granules_list, missing_list, downloaded_count, cached_count)
        """
        expected_starts = [
            end_time - timedelta(minutes=30 * (48 - i))
            for i in range(48)
        ]

        self.local_index.build_index()

        to_download: List[Dict] = []
        cached_items: Dict[datetime, Dict] = {}
        missing_starts: List[datetime] = []

        # Create quick lookup for pre-discovered online items if available
        pre_discovered_map = {g["start"]: g for g in (pre_discovered or [])}

        for start_dt in expected_starts:
            cached = self.local_index.get_granule(start_dt)
            if cached and is_valid_hdf5_granule(cached["path"]):
                cached_items[start_dt] = cached
                continue

            # Check if we have online download info
            if self.offline_mode or not self.downloader.token:
                missing_starts.append(start_dt)
                continue

            item = pre_discovered_map.get(start_dt)
            if not item:
                # Query CMR specifically for this 30-minute interval
                window_end = start_dt + timedelta(minutes=30)
                discovered = self.downloader.search_cmr_granules(start_dt, window_end)
                if discovered:
                    item = discovered[0]

            if item:
                to_download.append(item)
            else:
                missing_starts.append(start_dt)

        downloaded_count = 0
        cached_count = len(cached_items)

        # Download missing granules in parallel
        if to_download:
            print(f"Downloading {len(to_download)} missing granules in parallel (workers={self.workers})...")
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_item = {
                    executor.submit(self.downloader.download_single_granule, item): item
                    for item in to_download
                }
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        dest_path = future.result()
                        added = self.local_index.add_granule(dest_path)
                        if added:
                            cached_items[item["start"]] = added
                            downloaded_count += 1
                        else:
                            missing_starts.append(item["start"])
                    except Exception as exc:
                        print(f"Download failed for {item['filename']}: {exc}")
                        missing_starts.append(item["start"])

        # Collect complete sorted granules list
        granules_out = [cached_items[s] for s in expected_starts if s in cached_items]
        missing_out = sorted([s for s in expected_starts if s not in cached_items])

        return granules_out, missing_out, downloaded_count, cached_count

    def execute_refresh(
        self,
        target_end_time: Optional[datetime] = None,
        force_snapshot_name: Optional[str] = None,
    ) -> Dict:
        """
        Execute full dynamic rainfall refresh with freshness verification:
        1. Identifies the newest contiguous 48-granule sequence (via CMR online or local cache).
        2. Ingests granules, downloading only what is missing.
        3. Evaluates actual data latency against max_latency_hours.
        4. Categorizes status: SUCCESS, STALE_DATA, or INCOMPLETE_DATA.
        5. Computes 1h, 3h, 6h, 12h, 24h rolling accumulations.
        6. Spatially matches to the 14 Western Ghats tiles.
        7. Saves an immutable snapshot (never overwriting existing snapshots).
        8. Promotes to latest_validated_rainfall.csv ONLY IF status == SUCCESS.
        9. Records detailed execution metadata JSON.
        """
        execution_start_utc = datetime.now(timezone.utc)
        run_id = f"refresh_{execution_start_utc.strftime('%Y%m%d_%H%M%S')}"

        print("\n" + "=" * 75)
        print(f"STAGE 6B: DYNAMIC GPM IMERG REFRESH ({run_id})")
        print("=" * 75)
        print(f"Execution Time UTC:    {execution_start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        print(f"Max Latency Allowed:   {self.max_latency_hours} hours")

        # 1. Resolve target observation window
        target_end_time, pre_discovered, window_source = self.resolve_target_window(target_end_time)
        target_start_time = target_end_time - timedelta(hours=24)

        print(f"\nResolved 24-Hour Observation Window ({window_source}):")
        print(f"  Window Start UTC:    {target_start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        print(f"  Window End UTC:      {target_end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}")

        # 2. Calculate actual data latency
        actual_latency_hours = (execution_start_utc - target_end_time).total_seconds() / 3600.0
        print(f"  Actual Data Latency: {actual_latency_hours:.2f} hours")

        # 3. Ensure all 48 granules are available
        granules, missing_starts, n_downloaded, n_cached = self.ensure_granules_for_window(
            target_end_time, pre_discovered=pre_discovered
        )
        print(f"\nGranule Ingestion Summary:")
        print(f"  Reused locally:      {n_cached}")
        print(f"  Newly downloaded:    {n_downloaded}")
        print(f"  Missing intervals:   {len(missing_starts)}")

        is_complete = len(granules) == EXPECTED_GRANULES_24H and len(missing_starts) == 0

        # 4. Freshness & status classification
        if not is_complete:
            status = "INCOMPLETE_DATA"
            promotion_status = "REJECTED_INCOMPLETE"
        elif actual_latency_hours > self.max_latency_hours:
            status = "STALE_DATA"
            promotion_status = "REJECTED_STALE"
        else:
            status = "SUCCESS"
            promotion_status = "PROMOTED"

        print(f"\nRefresh Status Determination:")
        print(f"  Status:              {status}")
        print(f"  Promotion Action:    {promotion_status}")

        metadata = {
            "run_id": run_id,
            "status": status,
            "promotion_status": promotion_status,
            "execution_timestamp_utc": execution_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observation_start_utc": target_start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observation_end_utc": target_end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_latency_hours": round(actual_latency_hours, 3),
            "max_latency_hours": self.max_latency_hours,
            "window_discovery_source": window_source,
            "expected_granules": EXPECTED_GRANULES_24H,
            "available_granules": len(granules),
            "downloaded_granules": n_downloaded,
            "cached_granules": n_cached,
            "missing_intervals_count": len(missing_starts),
            "missing_intervals": [s.strftime("%Y-%m-%dT%H:%M:%SZ") for s in missing_starts],
            "is_24h_complete": is_complete,
            "gpm_product": GPM_PRODUCT_NAME,
            "spatial_resolution": GPM_NATIVE_RESOLUTION,
            "total_points": EXPECTED_TOTAL_COUNT,
            "dem_tiles_count": EXPECTED_TILE_COUNT,
        }

        # Handle incomplete window: fail safely without generating false zeros
        if status == "INCOMPLETE_DATA":
            meta_path = self.metadata_dir / f"refresh_metadata_{target_end_time.strftime('%Y%m%dT%H%M%SZ')}.json"
            atomic_write_json(meta_path, metadata)
            create_timestamped_backup(
                [self.latest_status_json],
                self.metadata_dir / "backup",
                run_ts=execution_start_utc.strftime("%Y%m%d_%H%M%S"),
            )
            atomic_write_json(self.latest_status_json, metadata)

            print(f"\nWARNING: Incomplete 24-hour GPM window ({len(granules)}/48 granules).")
            print(f"Never interpreting missing measurements as zero rainfall. Refresh halted safely.")
            print(f"Saved diagnostic metadata to: {meta_path.name}")
            return metadata

        # 5. Extract features using Stage 6A integration engine
        print(f"\n5. Loading Western Ghats locations and extracting features...")
        locations_df = load_western_ghats_locations(self.events_csv, self.background_csv)

        integrator = WesternGhatsGPMIntegrator()
        integrator.granules = granules
        feature_df = integrator.extract_features_for_dataframe(locations_df)

        # 6. Snapshot management (Never overwrite existing snapshots)
        snapshot_filename = (
            force_snapshot_name
            or f"western_ghats_rainfall_snapshot_{target_end_time.strftime('%Y%m%dT%H%M%SZ')}.csv"
        )
        snapshot_path = self.snapshots_dir / snapshot_filename

        if snapshot_path.exists():
            print(f"\nNotice: Snapshot already exists: {snapshot_path.name}")
            print("Preserving existing snapshot without modification.")
            snapshot_preserved = True
        else:
            staging_csv = self.snapshots_dir / f"{run_id}.tmp.csv"
            feature_df.to_csv(staging_csv, index=False)
            os.replace(staging_csv, snapshot_path)
            print(f"\nSaved new immutable snapshot to:")
            print(f"  {snapshot_path}")
            snapshot_preserved = False

        # 7. Promotion to latest validated copy (ONLY if status == SUCCESS)
        if status == "SUCCESS":
            create_timestamped_backup(
                [self.latest_validated_csv],
                self.output_dir / "backup",
                run_ts=execution_start_utc.strftime("%Y%m%d_%H%M%S"),
            )
            staging_latest = self.output_dir / f"{run_id}_latest.tmp.csv"
            feature_df.to_csv(staging_latest, index=False)
            os.replace(staging_latest, self.latest_validated_csv)
            print(f"PROMOTED: Updated latest validated snapshot at:")
            print(f"  {self.latest_validated_csv}")
        else:
            print(f"\nREJECTED PROMOTION: Stale dataset (latency {actual_latency_hours:.1f}h > {self.max_latency_hours}h).")
            print("Preserved existing latest validated snapshot without updating.")

        # 8. Record metadata
        max_dist = float(feature_df["gpm_matching_distance_km"].max())
        mean_dist = float(feature_df["gpm_matching_distance_km"].mean())
        min_dist = float(feature_df["gpm_matching_distance_km"].min())
        unique_pixels = int(len(set(zip(feature_df["gpm_grid_latitude"], feature_df["gpm_grid_longitude"]))))

        metadata.update({
            "granules_discovered_cmr": getattr(self.downloader, "last_discovered_cmr_count", 0),
            "source_granules": [g["filename"] for g in granules],
            "snapshot_path": str(snapshot_path),
            "snapshot_preserved_existing": snapshot_preserved,
            "latest_validated_path": str(self.latest_validated_csv) if status == "SUCCESS" else None,
            "spatial_matching": {
                "min_distance_km": round(min_dist, 3),
                "mean_distance_km": round(mean_dist, 3),
                "max_distance_km": round(max_dist, 3),
                "unique_gpm_pixels": unique_pixels,
            },
            "rainfall_summary": {
                "latest_rate_mean_mm_hr": round(float(feature_df["gpm_latest_rainfall_rate_mm_hr"].mean()), 3),
                "latest_rate_max_mm_hr": round(float(feature_df["gpm_latest_rainfall_rate_mm_hr"].max()), 3),
                "accum_1h_mean_mm": round(float(feature_df["gpm_1h_accumulation_mm"].mean()), 3),
                "accum_1h_max_mm": round(float(feature_df["gpm_1h_accumulation_mm"].max()), 3),
                "accum_3h_mean_mm": round(float(feature_df["gpm_3h_accumulation_mm"].mean()), 3),
                "accum_3h_max_mm": round(float(feature_df["gpm_3h_accumulation_mm"].max()), 3),
                "accum_6h_mean_mm": round(float(feature_df["gpm_6h_accumulation_mm"].mean()), 3),
                "accum_6h_max_mm": round(float(feature_df["gpm_6h_accumulation_mm"].max()), 3),
                "accum_12h_mean_mm": round(float(feature_df["gpm_12h_accumulation_mm"].mean()), 3),
                "accum_12h_max_mm": round(float(feature_df["gpm_12h_accumulation_mm"].max()), 3),
                "accum_24h_mean_mm": round(float(feature_df["gpm_24h_accumulation_mm"].mean()), 3),
                "accum_24h_max_mm": round(float(feature_df["gpm_24h_accumulation_mm"].max()), 3),
            },
        })

        meta_path = self.metadata_dir / f"refresh_metadata_{target_end_time.strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic_write_json(meta_path, metadata)
        create_timestamped_backup(
            [self.latest_status_json],
            self.metadata_dir / "backup",
            run_ts=execution_start_utc.strftime("%Y%m%d_%H%M%S"),
        )
        atomic_write_json(self.latest_status_json, metadata)

        print(f"Recorded refresh metadata to:")
        print(f"  {meta_path.name}")
        print("\n" + "=" * 75)
        print(f"STAGE 6B DYNAMIC REFRESH FINISHED (STATUS: {status}).")
        print("=" * 75)

        return metadata


# =====================================================================
# CLI ENTRYPOINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dynamic NASA GPM IMERG Rainfall Refresh Pipeline for Western Ghats."
    )
    parser.add_argument(
        "--target-end-utc",
        type=str,
        default=None,
        help="Specific observation end time in UTC (ISO format: YYYY-MM-DDTHH:MM:SSZ).",
    )
    parser.add_argument(
        "--max-latency-hours",
        type=float,
        default=DEFAULT_MAX_LATENCY_HOURS,
        help="Maximum data latency in hours before marking refresh as STALE_DATA (default: 12.0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent download threads (default: 4).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode using locally available granules only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DYNAMIC_OUTPUT_DIR,
        help="Root directory for refresh outputs, snapshots, and metadata.",
    )

    args = parser.parse_args()

    target_end_dt = None
    if args.target_end_utc:
        try:
            target_end_dt = datetime.strptime(
                args.target_end_utc, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            print("Error: --target-end-utc must be in YYYY-MM-DDTHH:MM:SSZ format", file=sys.stderr)
            sys.exit(1)

    engine = WesternGhatsDynamicRefreshEngine(
        output_dir=args.output_dir,
        max_latency_hours=args.max_latency_hours,
        workers=args.workers,
        offline_mode=args.offline,
    )

    try:
        engine.execute_refresh(target_end_time=target_end_dt)
    except Exception as exc:
        print(f"\nREFRESH ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
