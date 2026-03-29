"""
spatialos_tool.py — SpatialOps: geometry + spatial logic tool for remote sensing agents

Features (logic-class tools):
  • buffer_geometry     — Buffer/buffered ring around geometry
  • distance            — Minimum distance between two geometries
  • overlap_ratio       — IoU / overlap over A / overlap over B
  • containment         — Whether A contains B (with tolerance)

Design goals:
  • Minimal external deps: requires shapely>=2.0, numpy. Skimage is optional (for mask polygonization).
  • Accepts common perception outputs: bbox, polygon/multipolygon, mask (binary np.ndarray), COCO RLE, OBB.
  • Works in pixel space by default; can scale to metric units with provided pixel resolution.
  • Friendly agent integration via a single `apply_operation(payload: dict) -> dict` entrypoint.

Author: (you)
License: MIT
"""
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

try:
    from shapely.geometry import Polygon, MultiPolygon, box, Point, shape, mapping
    from shapely.affinity import scale, rotate, translate
    from shapely.ops import unary_union
    from shapely.validation import make_valid
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Shapely is required. Please install shapely>=2.0 (pip install shapely)."
    ) from e

# Optional: polygonize masks using scikit-image if available
_HAS_SKIMAGE = False
try:  # pragma: no cover
    from skimage import measure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# -----------------------------
# Data Structures & Utilities
# -----------------------------
@dataclass
class GeoMeta:
    crs: Optional[str] = None              # e.g., 'pixel' (default) or 'EPSG:3857'
    resolution: Optional[Tuple[float, float]] = None  # (sx, sy) pixel size in meters (or unit for scaling)


GeometryLike = Union[Polygon, MultiPolygon]


def _ensure_valid(g: GeometryLike) -> GeometryLike:
    """Make geometry valid; fix self-intersections.
    """
    if g.is_empty:
        return g
    try:
        vg = make_valid(g)
        # Shapely may return GeometryCollection; keep only polygonal parts
        if vg.geom_type == "GeometryCollection":
            polys = [geom for geom in vg.geoms if geom.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                return g
            vg = unary_union(polys)
        return vg
    except Exception:
        # Fallback: tiny buffer trick
        try:
            return g.buffer(0)
        except Exception:
            return g


# -----------------------------
# RLE decoding & mask polygonization
# -----------------------------

def decode_coco_rle(rle: Dict[str, Any], height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
    """Decode COCO-style RLE to binary mask.

    Supports both compressed RLE ("counts" as bytes-like string) and uncompressed (list of counts).
    Parameters height/width are required for compressed counts if not embedded.
    """
    counts = rle.get("counts")
    h = rle.get("size", [height, width])[0] if rle.get("size") else height
    w = rle.get("size", [height, width])[1] if rle.get("size") else width
    if h is None or w is None:
        raise ValueError("RLE decode requires height and width if not present in rle['size'].")

    # Uncompressed case: counts is list of ints
    if isinstance(counts, (list, tuple)):
        flat = np.zeros(h * w, dtype=np.uint8)
        idx = 0
        val = 0
        for run in counts:
            if run > 0:
                flat[idx:idx + run] = val
                idx += run
                val = 1 - val
        mask = flat.reshape((h, w), order="F")  # COCO uses Fortran order
        return mask

    # Compressed RLE (string/bytes) — minimal implementation
    # We implement a light decoder compatible with pycocotools encoding.
    if isinstance(counts, str):
        counts = counts.encode("utf-8")

    if not isinstance(counts, (bytes, bytearray)):
        raise ValueError("Unsupported RLE counts type.")

    # Decode compact RLE (as in pycocotools)
    def _rle_decompress(s: bytes) -> List[int]:
        cnts: List[int] = []
        p = 0
        m = 0
        val = 0
        more = 1
        x = 0
        while p < len(s):
            x = 0
            k = 0
            more = 1
            while more:
                c = s[p] - 48
                p += 1
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                k += 1
                if not more and (c & 0x10):
                    x |= - (1 << (5 * k))
            if m and x == 0:
                break
            cnts.append(x)
            m = 1
        # COCO flips run values starting with 0s
        runs: List[int] = []
        s0 = 0
        for i, c in enumerate(cnts):
            if i % 2 == 0:
                runs.extend([0] * (c))
            else:
                runs.extend([1] * (c))
            s0 += c
        # Build mask from runs
        flat = np.array(runs, dtype=np.uint8)
        if flat.size < h * w:
            pad = np.zeros(h * w - flat.size, dtype=np.uint8)
            flat = np.concatenate([flat, pad], axis=0)
        mask = flat[: h * w].reshape((h, w), order="F")
        return mask

    return _rle_decompress(counts)


def mask_to_polygons(mask: np.ndarray, level: float = 0.5, simplify_tol: float = 1.0) -> GeometryLike:
    """Convert a binary mask to Polygon/MultiPolygon using skimage if available.
    Fallback to bounding box polygon if skimage is not installed.
    """
    mask = (mask.astype(float) > 0.5).astype(np.uint8)
    if mask.sum() == 0:
        return Polygon()

    if _HAS_SKIMAGE:
        contours = measure.find_contours(mask, level=level)
        polys: List[Polygon] = []
        for c in contours:
            # flip (row, col) -> (x, y)
            coords = [(float(x), float(y)) for y, x in c]
            if len(coords) >= 3:
                p = Polygon(coords)
                if not p.is_valid:
                    p = _ensure_valid(p)
                p = p.simplify(simplify_tol, preserve_topology=True)
                if p.area > 0:
                    polys.append(p)
        if not polys:
            return Polygon()
        mp = unary_union(polys)
        mp = _ensure_valid(mp)
        return mp
    else:
        # Fallback: approximate by bounding box
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return Polygon()
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        return box(xmin, ymin, xmax, ymax)


# -----------------------------
# Geometry Factory
# -----------------------------
class GeometryFactory:
    """Create shapely geometry from a variety of input dicts.

    Supported input formats:
      - {"type": "bbox", "value": [x1, y1, x2, y2]}
      - {"type": "polygon", "value": [[x,y], [x,y], ...]}
      - {"type": "multipolygon", "value": [ [[...]], [[...]] ] }
      - {"type": "mask", "value": <binary 2D np.ndarray>, "threshold": 0.5}
      - {"type": "rle", "value": {"counts": ..., "size": [h,w]}}
      - {"type": "geojson", "value": <dict GeoJSON>}
      - {"type": "obb", "value": {"cx":..., "cy":..., "w":..., "h":..., "angle_deg":...}}

    Optional metadata:
      - "crs": "pixel" (default) or EPSG like "EPSG:3857"
      - "resolution": [sx, sy]  (pixel size in meters to enable scaling to meters)
    """

    @staticmethod
    def from_dict(geom_spec: Dict[str, Any]) -> Tuple[GeometryLike, GeoMeta]:
        gtype = geom_spec.get("type")
        value = geom_spec.get("value")
        crs = geom_spec.get("crs", "pixel")
        res = geom_spec.get("resolution")
        res_tuple = tuple(res) if res is not None else None
        meta = GeoMeta(crs=crs, resolution=res_tuple)  # type: ignore

        if gtype == "bbox":
            x1, y1, x2, y2 = map(float, value)
            geom = box(x1, y1, x2, y2)
        elif gtype == "polygon":
            coords = [(float(x), float(y)) for x, y in value]
            geom = Polygon(coords)
        elif gtype == "multipolygon":
            polys = []
            for poly in value:
                coords = [(float(x), float(y)) for x, y in poly]
                p = Polygon(coords)
                polys.append(p)
            geom = unary_union(polys)
        elif gtype == "mask":
            if not isinstance(value, np.ndarray):
                raise TypeError("For type 'mask', value must be a numpy 2D array.")
            thresh = float(geom_spec.get("threshold", 0.5))
            geom = mask_to_polygons((value > thresh).astype(np.uint8))
        elif gtype == "rle":
            rle = dict(value)
            h, w = rle.get("size", [None, None])
            mask = decode_coco_rle(rle, height=h, width=w)
            geom = mask_to_polygons(mask)
        elif gtype == "geojson":
            geom = shape(value)
        elif gtype == "obb":
            cx = float(value["cx"]) ; cy = float(value["cy"]) ; w = float(value["w"]) ; h = float(value["h"]) ; ang = float(value.get("angle_deg", 0.0))
            rect = box(-w/2.0, -h/2.0, w/2.0, h/2.0)
            rect = rotate(rect, ang, origin=(0, 0), use_radians=False)
            geom = translate(rect, xoff=cx, yoff=cy)
        else:
            raise ValueError(f"Unsupported geometry type: {gtype}")

        geom = _ensure_valid(geom)
        # Ensure polygonal output
        if geom.geom_type == "Polygon":
            return geom, meta
        if geom.geom_type == "MultiPolygon":
            return geom, meta
        # Convert other types (e.g., LineString) by buffering a tiny epsilon
        geom = geom.buffer(1e-6)
        geom = _ensure_valid(geom)
        return geom, meta


# -----------------------------
# Unit scaling helpers (pixel <-> metric)
# -----------------------------

def _scale_for_units(geom: GeometryLike, meta: GeoMeta, unit: Literal["px", "m"]) -> GeometryLike:
    """Scale geometry based on unit selection.
    If unit=="m" and meta.resolution=(sx, sy) provided, scale x by sx and y by sy.
    If resolution missing, fall back to pixel units with a warning.
    """
    if unit == "px":
        return geom
    if unit == "m":
        if meta.resolution is None:
            warnings.warn("No 'resolution' provided; using pixel units instead of meters.")
            return geom
        sx, sy = meta.resolution
        if sx == 0 or sy == 0:
            warnings.warn("Invalid resolution; using pixel units.")
            return geom
        return scale(geom, xfact=sx, yfact=sy, origin=(0, 0))
    raise ValueError("unit must be 'px' or 'm'")


# -----------------------------
# Core Spatial Ops
# -----------------------------

def op_buffer(geom: GeometryLike, meta: GeoMeta, distance: float, unit: Literal["px", "m"] = "px", 
              cap_style: Literal["round", "flat", "square"] = "round", join_style: Literal["round", "mitre", "bevel"] = "round") -> GeometryLike:
    geom_u = _scale_for_units(geom, meta, unit)
    cap_map = {"round": 1, "flat": 2, "square": 3}
    join_map = {"round": 1, "mitre": 2, "bevel": 3}
    buffered = geom_u.buffer(distance, cap_style=cap_map[cap_style], join_style=join_map[join_style])
    return _ensure_valid(buffered)


def op_distance(a: GeometryLike, a_meta: GeoMeta, b: GeometryLike, b_meta: GeoMeta, unit: Literal["px", "m"] = "px") -> float:
    # Bring both to same scale (unit space). We assume both pixel spaces have same resolution if provided.
    a_u = _scale_for_units(a, a_meta, unit)
    b_u = _scale_for_units(b, b_meta, unit)
    return float(a_u.distance(b_u))


def _areas_for_overlap(a: GeometryLike, b: GeometryLike) -> Tuple[float, float, float]:
    inter = a.intersection(b)
    if inter.is_empty:
        return 0.0, float(a.area), float(b.area)
    return float(inter.area), float(a.area), float(b.area)


def op_overlap_ratio(a: GeometryLike, a_meta: GeoMeta, b: GeometryLike, b_meta: GeoMeta, unit: Literal["px", "m"] = "px",
                     mode: Literal["iou", "over_a", "over_b"] = "iou") -> float:
    a_u = _scale_for_units(a, a_meta, unit)
    b_u = _scale_for_units(b, b_meta, unit)
    inter_area, a_area, b_area = _areas_for_overlap(a_u, b_u)
    if mode == "iou":
        union = a_area + b_area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union
    elif mode == "over_a":
        return 0.0 if a_area <= 0 else inter_area / a_area
    elif mode == "over_b":
        return 0.0 if b_area <= 0 else inter_area / b_area
    else:
        raise ValueError("mode must be one of 'iou', 'over_a', 'over_b'")


def op_containment(a: GeometryLike, a_meta: GeoMeta, b: GeometryLike, b_meta: GeoMeta, unit: Literal["px", "m"] = "px",
                   threshold: float = 1.0) -> bool:
    """Return True if A contains B under a coverage threshold (default 1.0 => strict contain).
    threshold in [0,1]: fraction of B's area that must be within A.
    """
    a_u = _scale_for_units(a, a_meta, unit)
    b_u = _scale_for_units(b, b_meta, unit)
    if b_u.is_empty:
        return False
    inter_area = float(a_u.intersection(b_u).area)
    b_area = float(b_u.area)
    if b_area <= 0:
        return False
    return (inter_area / b_area) >= max(0.0, min(1.0, threshold))


# -----------------------------
# Agent Entrypoint & Schema
# -----------------------------

def to_geojson_dict(geom: GeometryLike) -> Dict[str, Any]:
    return mapping(geom)


def tool_spec() -> Dict[str, Any]:
    """Function/tool schema for LLM planning systems (ToolBench/Function Calling-like)."""
    return {
        "name": "spatialos",
        "description": "Spatial logic ops over geometries (buffer, distance, overlap_ratio, containment). Accepts bbox/polygon/mask/RLE/geojson/obb; supports px or metric using resolution.",
        "parameters": {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["buffer", "distance", "overlap_ratio", "containment"],
                    "description": "Operation to perform."
                },
                "a": {"type": "object", "description": "Primary geometry spec (see docs)."},
                "b": {"type": ["object", "null"], "description": "Secondary geometry spec for pairwise ops (distance/overlap/containment)."},
                "params": {
                    "type": "object",
                    "properties": {
                        "unit": {"type": "string", "enum": ["px", "m"], "default": "px"},
                        "distance": {"type": "number", "description": "Buffer radius (for op=buffer)."},
                        "cap_style": {"type": "string", "enum": ["round", "flat", "square"], "default": "round"},
                        "join_style": {"type": "string", "enum": ["round", "mitre", "bevel"], "default": "round"},
                        "mode": {"type": "string", "enum": ["iou", "over_a", "over_b"], "default": "iou"},
                        "threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 1.0}
                    },
                    "additionalProperties": True
                }
            },
            "required": ["op", "a"]
        }
    }


def _build_geom(geom_spec: Dict[str, Any]) -> Tuple[GeometryLike, GeoMeta]:
    return GeometryFactory.from_dict(geom_spec)


def select_largest_geometry_per_class(detections: List[Dict[str, Any]],
                                     class_field: str = "class") -> Dict[str, Dict[str, Any]]:
    """
    Select the largest geometry object per class from detection/segmentation results.

    This function implements the geometry preprocessing strategy for spatial relationship analysis,
    ensuring that only the most representative (largest) geometry per class is used for calculations.

    Args:
        detections: List of detection/segmentation results with geometry information
        class_field: Field name containing the class information (default: "class")

    Returns:
        Dictionary mapping class names to their largest geometry objects
    """
    class_geometries = {}

    for detection in detections:
        class_name = detection.get(class_field, "unknown")

        # Calculate geometry area for comparison
        area = 0
        if "bbox" in detection:
            bbox = detection["bbox"]
            area = bbox.get("width", 0) * bbox.get("height", 0)
        elif "area_pixels" in detection:
            area = detection["area_pixels"]
        elif "area" in detection:
            area = detection["area"]

        # Keep the largest geometry for each class
        if class_name not in class_geometries or area > class_geometries[class_name].get("_area", 0):
            detection_copy = detection.copy()
            detection_copy["_area"] = area  # Store area for reference
            class_geometries[class_name] = detection_copy

    # Remove the temporary _area field
    for class_name, geometry in class_geometries.items():
        geometry.pop("_area", None)

    return class_geometries


def preprocess_all_geometries_for_spatial_relations(perception_results: Dict[str, Any],
                                                   classes_used: List[str] = None) -> Dict[str, Any]:
    """
    Preprocess perception tool results to extract ALL geometries for spatial relations.

    Unlike preprocess_geometries_for_spatial_relations(), this function includes
    all detected geometries rather than just the largest per class, providing
    comprehensive spatial relationship analysis.

    Args:
        perception_results: Results from perception tools (detection, segmentation, classification)
        classes_used: List of classes to extract (if None, extracts all detected classes)

    Returns:
        Dictionary containing:
        - success: Boolean indicating preprocessing success
        - all_geometries: Dictionary mapping class names to lists of all geometry objects
        - classes_processed: List of processed class names
        - total_original_detections: Total number of input detections
        - total_processed_geometries: Total number of output geometries
        - preprocessing_applied: Strategy identifier
    """
    try:
        # Extract detections from various perception tool formats
        detections = []

        # Handle direct segments field (from segmentation tool)
        if "segments" in perception_results:
            detections.extend(perception_results["segments"])
        # Handle direct detections field (from detection tool)
        elif "detections" in perception_results:
            detections.extend(perception_results["detections"])

        # Handle SAR detection results with "objects" field
        if "objects" in perception_results:
            detections.extend(perception_results["objects"])

        # Handle wrapped tool results (from orchestrator)
        for tool_name, tool_result in perception_results.items():
            if isinstance(tool_result, dict):
                # Handle wrapped format: {"success": True, "result": {...}}
                if "result" in tool_result and isinstance(tool_result["result"], dict):
                    result_data = tool_result["result"]

                    # Extract detections from result data
                    if "detections" in result_data:
                        detections.extend(result_data["detections"])

                    # Handle SAR detection results with "objects" field
                    if "objects" in result_data:
                        detections.extend(result_data["objects"])

                    # Handle segmentation_results from classification tool
                    if "segmentation_results" in result_data:
                        for class_name, class_data in result_data["segmentation_results"].items():
                            if "regions" in class_data:
                                for region in class_data["regions"]:
                                    region_copy = region.copy()
                                    region_copy["class"] = class_name
                                    detections.append(region_copy)

                # Handle direct format (backward compatibility)
                elif "segments" in tool_result:
                    # CRITICAL FIX: Handle segments field from segmentation tool
                    detections.extend(tool_result["segments"])
                elif "detections" in tool_result:
                    detections.extend(tool_result["detections"])
                elif "objects" in tool_result:
                    # Handle SAR detection results with "objects" field
                    detections.extend(tool_result["objects"])
                elif "segmentation_results" in tool_result:
                    for class_name, class_data in tool_result["segmentation_results"].items():
                        if "regions" in class_data:
                            for region in class_data["regions"]:
                                region_copy = region.copy()
                                region_copy["class"] = class_name
                                detections.append(region_copy)

                # CRITICAL FIX: Handle planner's coordinates_by_class storage format
                elif "coordinates_by_class" in tool_result:
                    coordinates_by_class = tool_result["coordinates_by_class"]
                    for class_name, coordinates_list in coordinates_by_class.items():
                        for coordinate_data in coordinates_list:
                            # Convert planner's coordinate format back to detection format
                            detection = coordinate_data.copy()
                            detection["class"] = class_name
                            detections.append(detection)

        # Handle legacy segmentation_results field (from classification tool)
        if "segmentation_results" in perception_results:
            for class_name, class_data in perception_results["segmentation_results"].items():
                if "regions" in class_data:
                    for region in class_data["regions"]:
                        region_copy = region.copy()
                        region_copy["class"] = class_name
                        detections.append(region_copy)

        if not detections:
            return {
                "success": False,
                "error": "No detections found in perception results",
                "all_geometries": {},
                "classes_processed": []
            }

        # Group ALL geometries by class (not just largest)
        all_geometries = {}
        for detection in detections:
            class_name = detection.get("class", "unknown").lower()

            if class_name not in all_geometries:
                all_geometries[class_name] = []

            all_geometries[class_name].append(detection)

        # Filter by classes_used if specified with singular/plural normalization
        if classes_used:
            # ENHANCED: Create a mapping of normalized class names with fuzzy matching
            # This handles: singular/plural, compound names, semantic variations
            class_mapping = {}

            # Common class name aliases for semantic variations
            CLASS_ALIASES = {
                'tree_canopy': 'trees',
                'tree_canopies': 'trees',
                'water_bodies': 'water',
                'water_body': 'water',
                'forest_area': 'forest',
                'forest_areas': 'forest',
                'building_footprint': 'building',
                'building_footprints': 'buildings',
                'road_network': 'road',
                'road_networks': 'roads',
                'car_parking': 'car',
                'vehicle': 'car',
                'vehicles': 'car',
            }

            for found_class in all_geometries.keys():
                # Map both singular and plural forms to the found class
                class_mapping[found_class] = found_class

                # Handle common singular/plural patterns
                if found_class.endswith('s') and found_class not in ['grass', 'class']:
                    # If found class is plural, also map singular
                    singular = found_class[:-1]
                    class_mapping[singular] = found_class
                else:
                    # If found class is singular, also map plural
                    plural = found_class + 's'
                    class_mapping[plural] = found_class

                # Special cases for common class name variations
                if found_class == 'building':
                    class_mapping['buildings'] = found_class
                elif found_class == 'buildings':
                    class_mapping['building'] = found_class
                elif found_class == 'road':
                    class_mapping['roads'] = found_class
                elif found_class == 'roads':
                    class_mapping['road'] = found_class

            # Filter geometries using normalized class names with fuzzy matching
            filtered_geometries = {}
            for expected_class in classes_used:
                expected_class_lower = expected_class.lower()
                matched_class = None

                # Try exact match first
                if expected_class_lower in class_mapping:
                    matched_class = class_mapping[expected_class_lower]

                # Try alias mapping
                elif expected_class_lower in CLASS_ALIASES:
                    alias_target = CLASS_ALIASES[expected_class_lower]
                    if alias_target in class_mapping:
                        matched_class = class_mapping[alias_target]
                        print(f"🔍 [Fuzzy Match] Mapped '{expected_class_lower}' → '{alias_target}' → '{matched_class}'")

                # Try substring/keyword matching as fallback
                if not matched_class:
                    # Extract keywords from expected class (split by underscore/space)
                    keywords = expected_class_lower.replace('_', ' ').split()
                    for keyword in keywords:
                        # Check if any found class contains this keyword or vice versa
                        for found_class in all_geometries.keys():
                            found_class_lower = found_class.lower()
                            # Bidirectional substring matching
                            if (keyword in found_class_lower or found_class_lower in keyword or
                                keyword in class_mapping.get(found_class_lower, '') or
                                found_class_lower in keyword):
                                matched_class = found_class
                                print(f"🔍 [Fuzzy Match] Keyword '{keyword}' matched '{expected_class_lower}' → '{matched_class}'")
                                break
                        if matched_class:
                            break

                # Add matched geometries to filtered results
                if matched_class and matched_class in all_geometries:
                    # Use the expected class name (from classes_used) as the key
                    filtered_geometries[expected_class_lower] = []
                    for geometry in all_geometries[matched_class]:
                        geometry_copy = geometry.copy()
                        # Update the class name in the geometry data to match expected
                        geometry_copy["class"] = expected_class_lower
                        filtered_geometries[expected_class_lower].append(geometry_copy)
                else:
                    print(f"⚠️ [Fuzzy Match] Could not match '{expected_class_lower}' with any available classes: {list(all_geometries.keys())}")

            all_geometries = filtered_geometries

        # Calculate total processed geometries
        total_processed = sum(len(geometries) for geometries in all_geometries.values())

        return {
            "success": True,
            "all_geometries": all_geometries,
            "classes_processed": list(all_geometries.keys()),
            "total_original_detections": len(detections),
            "total_processed_geometries": total_processed,
            "preprocessing_applied": "all_geometries_per_class"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "all_geometries": {},
            "classes_processed": []
        }


def preprocess_geometries_for_spatial_relations(perception_results: Dict[str, Any],
                                               classes_used: List[str] = None) -> Dict[str, Any]:
    """
    Preprocess perception tool results to extract largest geometry per class for spatial relations.

    This function implements the geometry preprocessing strategy that ensures spatial relationship
    calculations are performed on the most representative geometry objects rather than all fragments.

    Args:
        perception_results: Results from perception tools (detection, segmentation, classification)
        classes_used: List of classes to extract (if None, extracts all detected classes)

    Returns:
        Dictionary with preprocessed geometries ready for spatial relation tools
    """
    try:
        # Extract detections from various perception tool formats
        detections = []

        # Handle direct segments field (from segmentation tool)
        if "segments" in perception_results:
            detections.extend(perception_results["segments"])
        # Handle direct detections field (from detection tool)
        elif "detections" in perception_results:
            detections.extend(perception_results["detections"])

        # Handle wrapped tool results (from orchestrator)
        for tool_name, tool_result in perception_results.items():
            if isinstance(tool_result, dict):
                # Handle wrapped format: {"success": True, "result": {...}}
                if "result" in tool_result and isinstance(tool_result["result"], dict):
                    result_data = tool_result["result"]

                    # Extract segments from segmentation tool
                    if "segments" in result_data:
                        detections.extend(result_data["segments"])
                    # Extract detections from detection tool
                    elif "detections" in result_data:
                        detections.extend(result_data["detections"])

                    # Handle segmentation_results from classification tool
                    if "segmentation_results" in result_data:
                        for class_name, class_data in result_data["segmentation_results"].items():
                            if "regions" in class_data:
                                for region in class_data["regions"]:
                                    region_copy = region.copy()
                                    region_copy["class"] = class_name
                                    detections.append(region_copy)

                # Handle direct format (backward compatibility)
                elif "segments" in tool_result:
                    detections.extend(tool_result["segments"])
                elif "detections" in tool_result:
                    detections.extend(tool_result["detections"])
                elif "segmentation_results" in tool_result:
                    for class_name, class_data in tool_result["segmentation_results"].items():
                        if "regions" in class_data:
                            for region in class_data["regions"]:
                                region_copy = region.copy()
                                region_copy["class"] = class_name
                                detections.append(region_copy)

                # CRITICAL FIX: Handle planner's coordinates_by_class storage format
                elif "coordinates_by_class" in tool_result:
                    coordinates_by_class = tool_result["coordinates_by_class"]
                    for class_name, coordinates_list in coordinates_by_class.items():
                        for coordinate_data in coordinates_list:
                            # Convert planner's coordinate format back to detection format
                            detection = coordinate_data.copy()
                            detection["class"] = class_name
                            detections.append(detection)

        # Handle legacy segmentation_results field (from classification tool)
        if "segmentation_results" in perception_results:
            for class_name, class_data in perception_results["segmentation_results"].items():
                if "regions" in class_data:
                    for region in class_data["regions"]:
                        region_copy = region.copy()
                        region_copy["class"] = class_name
                        detections.append(region_copy)

        if not detections:
            return {
                "success": False,
                "error": "No detections found in perception results",
                "largest_geometries": {},
                "classes_processed": []
            }

        # Select largest geometry per class
        largest_geometries = select_largest_geometry_per_class(detections)

        # Filter by classes_used if specified with singular/plural normalization
        if classes_used:
            # ENHANCED: Create a mapping of normalized class names with fuzzy matching
            # This handles: singular/plural, compound names, semantic variations
            class_mapping = {}

            # Common class name aliases for semantic variations
            CLASS_ALIASES = {
                'tree_canopy': 'trees',
                'tree_canopies': 'trees',
                'water_bodies': 'water',
                'water_body': 'water',
                'forest_area': 'forest',
                'forest_areas': 'forest',
                'building_footprint': 'building',
                'building_footprints': 'buildings',
                'road_network': 'road',
                'road_networks': 'roads',
                'car_parking': 'car',
                'vehicle': 'car',
                'vehicles': 'car',
            }

            for found_class in largest_geometries.keys():
                # Map both singular and plural forms to the found class
                class_mapping[found_class] = found_class
                if found_class.endswith('s'):
                    # If found class is plural, also map singular
                    singular = found_class[:-1]
                    class_mapping[singular] = found_class
                else:
                    # If found class is singular, also map plural
                    plural = found_class + 's'
                    class_mapping[plural] = found_class

            # Filter geometries using normalized class names with fuzzy matching
            filtered_geometries = {}
            for expected_class in classes_used:
                expected_class_lower = expected_class.lower()
                matched_class = None

                # Try exact match first
                if expected_class_lower in class_mapping:
                    matched_class = class_mapping[expected_class_lower]

                # Try alias mapping
                elif expected_class_lower in CLASS_ALIASES:
                    alias_target = CLASS_ALIASES[expected_class_lower]
                    if alias_target in class_mapping:
                        matched_class = class_mapping[alias_target]
                        print(f"🔍 [Fuzzy Match] Mapped '{expected_class_lower}' → '{alias_target}' → '{matched_class}'")

                # Try substring/keyword matching as fallback
                if not matched_class:
                    # Extract keywords from expected class (split by underscore/space)
                    keywords = expected_class_lower.replace('_', ' ').split()
                    for keyword in keywords:
                        # Check if any found class contains this keyword or vice versa
                        for found_class in largest_geometries.keys():
                            found_class_lower = found_class.lower()
                            # Bidirectional substring matching
                            if (keyword in found_class_lower or found_class_lower in keyword or
                                keyword in class_mapping.get(found_class_lower, '') or
                                found_class_lower in keyword):
                                matched_class = found_class
                                print(f"🔍 [Fuzzy Match] Keyword '{keyword}' matched '{expected_class_lower}' → '{matched_class}'")
                                break
                        if matched_class:
                            break

                # Add matched geometry to filtered results
                if matched_class and matched_class in largest_geometries:
                    # Use the expected class name (from classes_used) as the key
                    filtered_geometries[expected_class_lower] = largest_geometries[matched_class].copy()
                    # Update the class name in the geometry data to match expected
                    filtered_geometries[expected_class_lower]["class"] = expected_class_lower
                else:
                    print(f"⚠️ [Fuzzy Match] Could not match '{expected_class_lower}' with any available classes: {list(largest_geometries.keys())}")

            largest_geometries = filtered_geometries

        return {
            "success": True,
            "largest_geometries": largest_geometries,
            "classes_processed": list(largest_geometries.keys()),
            "total_original_detections": len(detections),
            "total_largest_geometries": len(largest_geometries),
            "preprocessing_applied": "largest_geometry_per_class"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "largest_geometries": {},
            "classes_processed": []
        }


def extract_coordinates_from_perception_output(perception_result: Dict[str, Any], tool_type: str = "detection") -> Dict[str, Any]:
    """
    Extract coordinate data from perception tool outputs for spatial analysis.
    Bridges the data format gap between perception and spatial relations tools.

    Args:
        perception_result: JSON output from perception tools (detection, segmentation, classification)
        tool_type: Type of perception tool ("detection", "segmentation", "classification")

    Returns:
        Dictionary with extracted coordinates and metadata
    """
    try:
        coordinates = []
        geometry_type = "points"

        if tool_type == "detection" and "detections" in perception_result:
            # Extract bounding boxes from detection results
            for detection in perception_result["detections"]:
                if "bbox" in detection:
                    bbox = detection["bbox"]
                    # Convert to [x_min, y_min, x_max, y_max] format for spatial tools
                    coord = [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]
                    coordinates.append(coord)
            geometry_type = "bboxes"

        elif tool_type == "segmentation" and "detections" in perception_result:
            # Extract polygon coordinates from segmentation detections
            for detection in perception_result["detections"]:
                if "polygon" in detection and detection["polygon"]:
                    # Use actual polygon coordinates from segmentation
                    coordinates.append(detection["polygon"])
            geometry_type = "polygons"

        elif tool_type == "classification":
            # Classification typically provides region-based results
            # Use image center or classification regions as points
            coordinates = [[512, 512]]  # Default center point
            geometry_type = "points"

        return {
            "success": True,
            "coordinates": coordinates,
            "geometry_type": geometry_type,
            "coordinate_count": len(coordinates),
            "source_tool": tool_type,
            "extraction_method": "automated_perception_output_parsing"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "coordinates": [],
            "geometry_type": "points",
            "coordinate_count": 0,
            "source_tool": tool_type
        }


def convert_all_geometries_to_spatial_format(all_geometries: Dict[str, List[Dict[str, Any]]],
                                            classes_for_roles: List[str] = None) -> Dict[str, Any]:
    """
    Convert all geometries to the format expected by spatial relation tools.

    This function takes the output from preprocess_all_geometries_for_spatial_relations()
    and converts it to the coordinate format expected by containment, overlap, and buffer tools.
    Unlike convert_largest_geometries_to_spatial_format(), this processes ALL geometries.

    Args:
        all_geometries: Dictionary mapping class names to lists of all geometry objects
        classes_for_roles: List of classes in order for role assignment (container, contained, etc.)

    Returns:
        Dictionary containing:
        - success: Boolean indicating conversion success
        - coordinates_by_class: Dictionary mapping class names to lists of coordinate arrays
        - geometry_types: Dictionary mapping class names to geometry types
        - role_assignments: Dictionary with role-specific geometry assignments
        - geometry_counts: Dictionary with per-class geometry counts
    """
    try:
        result = {
            "success": True,
            "coordinates_by_class": {},
            "geometry_types": {},
            "role_assignments": {},
            "geometry_counts": {},
            "error": None
        }

        # Convert each class's geometries to coordinate format
        for class_name, geometries in all_geometries.items():
            coordinates_list = []
            geometry_type = "points"

            for geometry in geometries:
                coordinates = []

                # Prioritize polygon coordinates from segmentation
                if "polygon" in geometry and geometry["polygon"]:
                    # Parse polygon coordinates if they are JSON strings
                    polygon_data = geometry["polygon"]
                    if isinstance(polygon_data, str):
                        try:
                            import json
                            polygon_data = json.loads(polygon_data)
                        except (json.JSONDecodeError, TypeError):
                            # Keep original if parsing fails
                            pass

                    # Use actual polygon coordinates from segmentation
                    coordinates = polygon_data  # Don't wrap in list here
                    geometry_type = "polygons"  # Use plural form expected by spatial tools

                elif "bbox" in geometry:
                    # Convert bounding box to coordinate format
                    bbox = geometry["bbox"]
                    coordinates = [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]
                    geometry_type = "bboxes"

                elif "centroid" in geometry:
                    # Use centroid as point coordinate
                    centroid = geometry["centroid"]
                    if isinstance(centroid, dict) and "x" in centroid and "y" in centroid:
                        coordinates = [centroid["x"], centroid["y"]]
                        geometry_type = "points"

                elif "mask" in geometry:
                    # For mask data, we would need to extract contours
                    # This is a placeholder - in practice, you'd use cv2.findContours
                    coordinates = []
                    geometry_type = "masks"

                if coordinates:
                    coordinates_list.append(coordinates)

            result["coordinates_by_class"][class_name] = coordinates_list
            result["geometry_types"][class_name] = geometry_type
            result["geometry_counts"][class_name] = len(coordinates_list)

        # Assign roles based on classes_for_roles order if provided
        if classes_for_roles and len(classes_for_roles) >= 1:
            class_names = [cls for cls in classes_for_roles if cls in all_geometries]

            if len(class_names) >= 1:
                # For buffer: use first class as the geometry to buffer (all geometries of that class)
                result["role_assignments"]["buffer_geometries"] = {
                    "class": class_names[0],
                    "coordinates": result["coordinates_by_class"][class_names[0]],
                    "geometry_type": result["geometry_types"][class_names[0]],
                    "count": result["geometry_counts"][class_names[0]]
                }

            if len(class_names) >= 2:
                # For containment: first class is container, second is contained (all geometries)
                result["role_assignments"]["container_geometries"] = {
                    "class": class_names[0],
                    "coordinates": result["coordinates_by_class"][class_names[0]],
                    "geometry_type": result["geometry_types"][class_names[0]],
                    "count": result["geometry_counts"][class_names[0]]
                }
                result["role_assignments"]["contained_geometries"] = {
                    "class": class_names[1],
                    "coordinates": result["coordinates_by_class"][class_names[1]],
                    "geometry_type": result["geometry_types"][class_names[1]],
                    "count": result["geometry_counts"][class_names[1]]
                }

                # For overlap: assign as geometry_a and geometry_b (all geometries)
                result["role_assignments"]["geometry_a_all"] = result["role_assignments"]["container_geometries"]
                result["role_assignments"]["geometry_b_all"] = result["role_assignments"]["contained_geometries"]

        return result

    except Exception as e:
        return {
            "success": False,
            "coordinates_by_class": {},
            "geometry_types": {},
            "role_assignments": {},
            "geometry_counts": {},
            "error": str(e)
        }


def convert_largest_geometries_to_spatial_format(largest_geometries: Dict[str, Dict[str, Any]],
                                                classes_for_roles: List[str] = None) -> Dict[str, Any]:
    """
    Convert largest geometries to the format expected by spatial relation tools.

    This function takes the output from preprocess_geometries_for_spatial_relations()
    and converts it to the coordinate format expected by containment, overlap, and buffer tools.

    Args:
        largest_geometries: Dictionary mapping class names to their largest geometry objects
        classes_for_roles: List of classes in order for role assignment (container, contained, etc.)

    Returns:
        Dictionary with coordinates formatted for spatial relation tools
    """
    try:
        result = {
            "success": True,
            "coordinates_by_class": {},
            "geometry_types": {},
            "role_assignments": {},
            "error": None
        }

        # Convert each class's largest geometry to coordinate format
        for class_name, geometry in largest_geometries.items():
            coordinates = []
            geometry_type = "points"

            # Prioritize polygon coordinates from segmentation
            if "polygon" in geometry and geometry["polygon"]:
                # Parse polygon coordinates if they are JSON strings
                polygon_data = geometry["polygon"]
                if isinstance(polygon_data, str):
                    try:
                        import json
                        polygon_data = json.loads(polygon_data)
                    except (json.JSONDecodeError, TypeError):
                        # Keep original if parsing fails
                        pass

                # Use actual polygon coordinates from segmentation
                coordinates = [polygon_data]  # Wrap in list for polygons format
                geometry_type = "polygons"  # Use plural form expected by spatial tools

            elif "bbox" in geometry:
                # Convert bounding box to coordinate format
                bbox = geometry["bbox"]
                coordinates = [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]
                geometry_type = "bbox"

            elif "centroid" in geometry:
                # Use centroid as point coordinate
                centroid = geometry["centroid"]
                if isinstance(centroid, dict) and "x" in centroid and "y" in centroid:
                    coordinates = [centroid["x"], centroid["y"]]
                    geometry_type = "point"

            elif "mask" in geometry:
                # For mask data, we would need to extract contours
                # This is a placeholder - in practice, you'd use cv2.findContours
                coordinates = []
                geometry_type = "mask"

            result["coordinates_by_class"][class_name] = coordinates
            result["geometry_types"][class_name] = geometry_type

        # Assign roles based on classes_for_roles order if provided
        if classes_for_roles and len(classes_for_roles) >= 1:
            class_names = [cls for cls in classes_for_roles if cls in largest_geometries]

            if len(class_names) >= 1:
                # For buffer: use first class as the geometry to buffer (single class operation)
                result["role_assignments"]["buffer_geometry"] = {
                    "class": class_names[0],
                    "coordinates": result["coordinates_by_class"][class_names[0]],
                    "geometry_type": result["geometry_types"][class_names[0]]
                }

            if len(class_names) >= 2:
                # For containment: first class is container, second is contained
                result["role_assignments"]["container"] = {
                    "class": class_names[0],
                    "coordinates": result["coordinates_by_class"][class_names[0]],
                    "geometry_type": result["geometry_types"][class_names[0]]
                }
                result["role_assignments"]["contained"] = {
                    "class": class_names[1],
                    "coordinates": result["coordinates_by_class"][class_names[1]],
                    "geometry_type": result["geometry_types"][class_names[1]]
                }

                # For overlap: assign as geometry_a and geometry_b
                result["role_assignments"]["geometry_a"] = result["role_assignments"]["container"]
                result["role_assignments"]["geometry_b"] = result["role_assignments"]["contained"]

        return result

    except Exception as e:
        return {
            "success": False,
            "coordinates_by_class": {},
            "geometry_types": {},
            "role_assignments": {},
            "error": str(e)
        }


def apply_operation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Unified agent entry point.

    payload: {
      "op": "buffer" | "distance" | "overlap_ratio" | "containment",
      "a": { ... geometry spec ... },
      "b": { ... geometry spec ... } | null,
      "params": { ... op-specific ... }
    }

    Returns a JSON-serializable dict with results.
    """
    op = payload.get("op")
    params = payload.get("params", {}) or {}

    # Build geometry A
    a_geom, a_meta = _build_geom(payload["a"])  # type: ignore

    if op == "buffer":
        distance_val = float(params.get("distance", 0.0))
        unit = params.get("unit", "px")
        cap_style = params.get("cap_style", "round")
        join_style = params.get("join_style", "round")
        out_geom = op_buffer(a_geom, a_meta, distance_val, unit=unit, cap_style=cap_style, join_style=join_style)
        return {
            "ok": True,
            "op": op,
            "geometry": to_geojson_dict(out_geom),
            "meta": {"unit": unit}
        }

    # Pairwise ops require geometry B
    if payload.get("b") is None:
        raise ValueError(f"Operation '{op}' requires a secondary geometry 'b'.")
    b_geom, b_meta = _build_geom(payload["b"])  # type: ignore

    if op == "distance":
        unit = params.get("unit", "px")
        d = op_distance(a_geom, a_meta, b_geom, b_meta, unit=unit)
        # minor rounding for consistent outputs
        return {"ok": True, "op": op, "value": float(round(d, 6)), "meta": {"unit": unit}}

    if op == "overlap_ratio":
        unit = params.get("unit", "px")
        mode = params.get("mode", "iou")
        v = op_overlap_ratio(a_geom, a_meta, b_geom, b_meta, unit=unit, mode=mode)
        return {"ok": True, "op": op, "value": float(round(v, 6)), "meta": {"unit": unit, "mode": mode}}

    if op == "containment":
        unit = params.get("unit", "px")
        thr = float(params.get("threshold", 1.0))
        v = op_containment(a_geom, a_meta, b_geom, b_meta, unit=unit, threshold=thr)
        return {"ok": True, "op": op, "value": bool(v), "meta": {"unit": unit, "threshold": thr}}

    raise ValueError(f"Unsupported op: {op}")


# -----------------------------
# CLI for quick testing
# -----------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="SpatialOps (spatialos) quick runner")
    parser.add_argument("op", choices=["buffer", "distance", "overlap_ratio", "containment"], help="Operation")
    parser.add_argument("--payload", type=str, help="Path to JSON payload or JSON string.")
    args = parser.parse_args()

    if args.payload and args.payload.strip().endswith(".json"):
        with open(args.payload, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        # If string provided directly
        try:
            payload = json.loads(args.payload)
        except Exception:
            raise SystemExit("Provide --payload as JSON string or path to .json")

    res = apply_operation(payload)
    print(json.dumps(res, ensure_ascii=False))
