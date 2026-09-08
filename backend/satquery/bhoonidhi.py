"""ISRO Bhoonidhi STAC API Connector & Client

Provides programmatic discovery and acquisition of Indian Remote Sensing
satellite scenes (Cartosat-2S, Cartosat-3, EOS-04 SAR, RISAT-1, ResourceSat,
and regional Sentinel-1/2 coverage) from NRSC Bhoonidhi (bhoonidhi.nrsc.gov.in).
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output encoding on Windows so rich/unicode spinners never crash
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from bhoonidhi_downloader.sdk import BhoonidhiClient
    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False

try:
    from bhoonidhi.SmartSearch import bhoonidhiSmartSearch
    _HAS_SMART_SEARCH = True
except ImportError:
    _HAS_SMART_SEARCH = False

_CLIENT: Optional[Any] = None


def get_client() -> Optional[Any]:
    global _CLIENT
    if not _HAS_SDK:
        return None
    if _CLIENT is None:
        try:
            _CLIENT = BhoonidhiClient()
        except Exception as e:
            print(f"[bhoonidhi] Client initialization error: {e}", file=sys.stderr)
            return None
    return _CLIENT


def get_auth_status() -> Dict[str, Any]:
    """Check if Bhoonidhi session is authenticated or anonymous."""
    client = get_client()
    if client is None:
        return {"available": False, "authenticated": False, "user": None}
    try:
        is_auth = client.is_authenticated
        user = client.whoami() if is_auth else None
        return {"available": True, "authenticated": is_auth, "user": user}
    except Exception as e:
        return {"available": True, "authenticated": False, "error": str(e)}


def list_archives(filter_keyword: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available satellite missions and sensor packages on Bhoonidhi."""
    client = get_client()
    if client is None:
        return []

    try:
        raw_archives = client.archive.list()
        results = []
        for arch in raw_archives:
            sat_name = arch.get("satName", "")
            if filter_keyword and filter_keyword.lower() not in sat_name.lower():
                continue

            sensors = []
            for s in arch.get("sensors", []):
                sensors.append({
                    "name": s.get("senName"),
                    "display": s.get("dispSen"),
                    "resolution_m": s.get("res"),
                    "product": s.get("products"),
                    "start_date": s.get("stDate"),
                    "end_date": s.get("endDate") or "Active",
                })

            results.append({
                "satellite": sat_name,
                "access": arch.get("priced", "OpenData"),
                "min_res_m": arch.get("thisMinRes"),
                "max_res_m": arch.get("thisMaxRes"),
                "start_date": arch.get("totalStartDate"),
                "end_date": arch.get("totalEndDate") or "Present",
                "sensors": sensors,
            })
        return results
    except Exception as e:
        print(f"[bhoonidhi] Error listing archives: {e}", file=sys.stderr)
        return []


def search_scenes(
    start_date: str,
    end_date: str,
    satellite: str,
    sensor: Optional[str] = None,
    bbox: Optional[List[float]] = None,
    limit: int = 15,
) -> Dict[str, Any]:
    """Search Bhoonidhi catalog for satellite scenes by bounding box and dates.

    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        satellite: e.g. 'EOS-04', 'CartoSat-2S', 'Sentinel-2A', 'RISAT-1'
        sensor: e.g. 'SAR(MRS)', 'MSI', 'PAN(SPOT)'
        bbox: [min_lon, min_lat, max_lon, max_lat]
        limit: max number of scenes to return
    """
    client = get_client()
    if client is None:
        return {"error": "Bhoonidhi client not installed", "scenes": []}

    try:
        s_dt = datetime.strptime(start_date, "%Y-%m-%d")
        e_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        return {"error": f"Invalid date format (use YYYY-MM-DD): {e}", "scenes": []}

    # Default to a representative Indian region (Bengaluru / Karnataka) if no bbox
    if not bbox or len(bbox) != 4:
        bbox = [77.50, 12.90, 77.70, 13.10]

    minx, miny, maxx, maxy = bbox

    try:
        kwargs: Dict[str, Any] = {
            "satellite": satellite,
            "minx": minx,
            "maxx": maxx,
            "miny": miny,
            "maxy": maxy,
            "save": False,
        }
        if sensor:
            kwargs["sensor"] = sensor

        f_out = io.StringIO()
        with redirect_stdout(f_out), redirect_stderr(f_out):
            query = client.query.create(s_dt, e_dt, **kwargs)
        raw_scenes = getattr(query, "scenes", []) or []

        scenes = []
        for s in raw_scenes[:limit]:
            s_dict = s if isinstance(s, dict) else (s.dict() if hasattr(s, "dict") else {})
            scene_id = s_dict.get("ID") or s_dict.get("FILENAME") or "Unknown"

            scenes.append({
                "id": scene_id,
                "satellite": s_dict.get("SATELLITE", satellite),
                "sensor": s_dict.get("SENSOR", sensor or ""),
                "product_type": s_dict.get("PRODTYPE", ""),
                "dop": s_dict.get("DOP", ""),
                "orbit": s_dict.get("GROUND_ORBIT_NO") or s_dict.get("IMAGING_ORBIT_NO"),
                "tile_id": s_dict.get("TILE_ID", ""),
                "priced": s_dict.get("PRICED", ""),
                "coverage_pct": s_dict.get("OverLapPercent", ""),
                "corners": {
                    "nw": [float(s_dict.get("CrnNWLon", minx)), float(s_dict.get("CrnNWLat", maxy))],
                    "ne": [float(s_dict.get("CrnNELon", maxx)), float(s_dict.get("CrnNELat", maxy))],
                    "se": [float(s_dict.get("CrnSELon", maxx)), float(s_dict.get("CrnSELat", miny))],
                    "sw": [float(s_dict.get("CrnSWLon", minx)), float(s_dict.get("CrnSWLat", miny))],
                },
                "download_ready": "DirectDownload" in (s_dict.get("PRICED") or ""),
            })

        return {
            "satellite": satellite,
            "sensor": sensor,
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "total_found": len(raw_scenes),
            "returned": len(scenes),
            "scenes": scenes,
        }
    except Exception as e:
        return {"error": str(e), "scenes": []}


def smart_search(natural_language_query: str) -> List[Dict[str, Any]]:
    """Execute natural language search using the official Bhoonidhi SmartSearch NLP engine."""
    if not _HAS_SMART_SEARCH:
        return []
    try:
        f_out = io.StringIO()
        with redirect_stdout(f_out), redirect_stderr(f_out):
            raw_results = bhoonidhiSmartSearch(natural_language_query)
        if isinstance(raw_results, list):
            return raw_results
        return []
    except Exception as e:
        print(f"[bhoonidhi] SmartSearch error: {e}", file=sys.stderr)
        return []
