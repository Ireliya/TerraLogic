"""
DistanceCalculationTool (pure geometric)
Computes shortest distances between two sets of polygonal geometries (Polygon / MultiPolygon only).

- I/O strictly in polygon-like coordinates
- Optional AOI clipping
- Robust geometry cleaning & tiny-area filtering
- Outputs per-pair distances (pixels & meters), global shortest pair, per-source nearest, and stats
"""

from typing import List, Dict, Any, Union, Optional, Type
from pydantic import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import nearest_points, unary_union
import numpy as np
import json


PolygonLike = Union[
    List[List[float]],           # simple polygon: [[x,y], [x,y], ...]
    List[List[List[float]]]      # polygon with holes: [outer, hole1, ...]  OR multipolygon: [[poly1], [poly2], ...]
]

class DistanceGeometricInput(BaseModel):
    """
    Pure geometric input for distance analysis.
    Each item in set_a / set_b can be:
      - Simple polygon: [[x,y], ...]
      - Polygon with holes: [outer, hole1, ...]
      - MultiPolygon: [[poly1], [poly2], ...] (each poly can be simple or with holes)
    """
    set_a: List[PolygonLike] = Field(..., description="Source polygon-like list")
    set_b: List[PolygonLike] = Field(..., description="Target polygon-like list")
    aoi: Optional[PolygonLike] = Field(default=None, description="Optional Area-Of-Interest polygon-like for clipping")

    meters_per_pixel: float = Field(default=0.3, gt=0, description="Ground resolution in meters per pixel")
    tolerance: float = Field(default=1e-6, gt=0, description="Area tolerance (in pixel^2) for tiny parts filtering")
    min_area_sqm: Optional[float] = Field(default=None, ge=0, description="Drop objects smaller than this area (square meters) after cleaning/clipping")

    ids_a: Optional[List[str]] = Field(default=None, description="Optional IDs for set_a (same length as set_a)")
    ids_b: Optional[List[str]] = Field(default=None, description="Optional IDs for set_b (same length as set_b)")

    include_cleaned_geometries: bool = Field(default=False, description="Include cleaned geometry coordinates preview in output")
    cleaned_geometries_limit: Optional[int] = Field(default=None, description="Max number of objects to include geometry for (total across A+B)")
    include_holes_in_geometry: bool = Field(default=True, description="If returning geometries, include holes")

    distance_unit: str = Field(default="meters", description="Unit emphasized in summary: 'meters' or 'pixels'")

    # Optional parameters for unified arguments field (from upstream tools)
    image_path: Optional[str] = Field(default=None, description="Path to input satellite image (for traceability)")
    query_text: Optional[str] = Field(default="", description="Original natural-language query")
    classes_used: Optional[List[str]] = Field(default=None, description="The two classes involved in distance computation")

    @root_validator(skip_on_failure=True)
    def _validate_ids(cls, values):
        a = values.get("set_a", [])
        b = values.get("set_b", [])
        ida = values.get("ids_a", None)
        idb = values.get("ids_b", None)
        if ida is not None and len(ida) != len(a):
            raise ValueError("ids_a length must match set_a length")
        if idb is not None and len(idb) != len(b):
            raise ValueError("ids_b length must match set_b length")
        du = values.get("distance_unit", "meters")
        if du not in ("meters", "pixels"):
            raise ValueError("distance_unit must be 'meters' or 'pixels'")
        return values


# -----------------------------------
#        Distance Calculation Tool
# -----------------------------------

class DistanceCalculationTool(BaseTool):
    """
    Pure geometric distance calculation:
      - Pairwise shortest distances between sets A and B
      - AOI clipping, geometry cleaning, tiny-area filtering
      - I/O strictly uses polygon coordinates; no image IO
    """
    name: str = "distance"
    description: str = (
        "Compute shortest distances between two sets of polygonal geometries (Polygon/MultiPolygon only). "
        "Supports AOI clipping, robust cleaning, tiny-area filtering, and returns per-pair distances, "
        "global shortest pair, per-source nearest targets, and statistics."
    )
    args_schema: Type[BaseModel] = DistanceGeometricInput

    # -------- public JSON API --------

    def _run(self, tool_input: Union[Dict[str, Any], str]) -> str:
        try:
            params = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            inp = DistanceGeometricInput(**params)

            warnings: List[str] = []
            # Build & clean inputs
            min_area_px = self._sqm_to_pixels(inp.min_area_sqm, inp.meters_per_pixel) if inp.min_area_sqm is not None else None

            geoms_a = self._build_and_clean_batch(
                inp.set_a, warnings, label="A", tol=inp.tolerance, min_area_pixels=min_area_px
            )
            geoms_b = self._build_and_clean_batch(
                inp.set_b, warnings, label="B", tol=inp.tolerance, min_area_pixels=min_area_px
            )

            if inp.aoi is not None:
                aoi = self._create_polygon_like(inp.aoi, warnings, label="AOI")
                aoi = self._clean_geometry(aoi, inp.tolerance)
                if aoi is None:
                    warnings.append("AOI is invalid or near-zero after cleaning; ignoring AOI.")
                else:
                    geoms_a = self._clip_and_clean_batch(geoms_a, aoi, warnings, label="A", tol=inp.tolerance, min_area_pixels=min_area_px)
                    geoms_b = self._clip_and_clean_batch(geoms_b, aoi, warnings, label="B", tol=inp.tolerance, min_area_pixels=min_area_px)

            if not geoms_a or not geoms_b:
                return json.dumps(self._fail("All geometries in set_a or set_b are invalid/empty after cleaning/clipping.", warnings), indent=2)

            # Compute pairwise distances
            measurements, dist_px, global_min = self._pairwise_distances(
                geoms_a, geoms_b, ids_a=inp.ids_a, ids_b=inp.ids_b, meters_per_pixel=inp.meters_per_pixel
            )

            # Per-source nearest
            per_source_nearest = self._per_source_nearest(measurements)

            # Stats
            stats_pixels = self._basic_stats(dist_px)
            stats_meters = {k: (v * inp.meters_per_pixel if k in ("min", "max", "mean", "median", "std") else v)
                            for k, v in stats_pixels.items()}

            # Geometry preview (optional, limited)
            previews = {}
            if inp.include_cleaned_geometries:
                previews = self._geometry_previews(
                    geoms_a, geoms_b, ids_a=inp.ids_a, ids_b=inp.ids_b,
                    limit=inp.cleaned_geometries_limit, include_holes=inp.include_holes_in_geometry
                )

            # Summary emphasized in chosen unit
            if inp.distance_unit == "meters":
                shortest_value = round(global_min["distance_pixels"] * inp.meters_per_pixel, 3)
                unit = "meters"
                stats_for_summary = {k: round(v, 3) for k, v in stats_meters.items()}
            else:
                shortest_value = round(global_min["distance_pixels"], 3)
                unit = "pixels"
                stats_for_summary = {k: round(v, 3) for k, v in stats_pixels.items()}

            result = {
                "success": True,
                "tool": "distance",
                "input": {
                    "len_set_a": len(geoms_a),
                    "len_set_b": len(geoms_b),
                    "meters_per_pixel": inp.meters_per_pixel,
                    "tolerance": inp.tolerance,
                    "min_area_sqm": inp.min_area_sqm,
                    "aoi_provided": inp.aoi is not None,
                    "distance_unit": inp.distance_unit,
                    "include_cleaned_geometries": inp.include_cleaned_geometries,
                    "cleaned_geometries_limit": inp.cleaned_geometries_limit,
                    "include_holes_in_geometry": inp.include_holes_in_geometry
                },
                "output": {
                    "measurements": measurements,                 # per-pair distances (pixels & meters, closest points)
                    "per_source_nearest": per_source_nearest,     # nearest target for each source
                    "global_shortest_pair": {
                        "source_id": global_min["source_id"],
                        "target_id": global_min["target_id"],
                        "distance_pixels": round(global_min["distance_pixels"], 3),
                        "distance_meters": round(global_min["distance_pixels"] * inp.meters_per_pixel, 3),
                        "closest_points": global_min["closest_points"]
                    },
                    "stats": {
                        "pixels": {k: round(v, 3) for k, v in stats_pixels.items()},
                        "meters": {k: round(v, 3) for k, v in stats_meters.items()}
                    },
                    "summary": {
                        "total_pairs": len(measurements),
                        "shortest_distance": shortest_value,
                        "shortest_distance_unit": unit
                    },
                    **({"geometry_previews": previews} if previews else {})
                },
                "metadata": {
                    "validation_warnings": warnings
                }
            }

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics
            # This matches the format used in perception tools, spatial relation tools, and statistical tools for consistency

            # Build classes_used list
            if inp.classes_used is not None:
                classes_used_list = inp.classes_used
            else:
                classes_used_list = []

            # Sort classes alphabetically for consistency
            classes_used_sorted = sorted(classes_used_list)

            # Extract min and max distances in meters
            min_dist_m = None
            max_dist_m = None
            if dist_px:
                min_dist_m = round(min(dist_px) * inp.meters_per_pixel, 3)
                max_dist_m = round(max(dist_px) * inp.meters_per_pixel, 3)

            # Build the unified arguments field
            arguments = {
                "image_path": inp.image_path,
                "classes_used": classes_used_sorted,
                "meters_per_pixel": inp.meters_per_pixel if inp.meters_per_pixel else None,
                "count_set_a": len(geoms_a),
                "count_set_b": len(geoms_b),
                "total_pairs": len(geoms_a) * len(geoms_b),
                "min_nearest_distance_m": min_dist_m,
                "max_nearest_distance_m": max_dist_m
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps(self._fail(f"Distance calculation failed: {e}"), indent=2)

    # -----------------------------------
    #      Build & clean geometries
    # -----------------------------------

    def _build_and_clean_batch(
        self,
        items: List[PolygonLike],
        warnings: List[str],
        label: str,
        tol: float,
        min_area_pixels: Optional[float]
    ) -> List[Union[Polygon, MultiPolygon]]:
        out: List[Union[Polygon, MultiPolygon]] = []
        for i, coords in enumerate(items):
            g = self._create_polygon_like(coords, warnings, label=f"{label}[{i}]")
            g = self._clean_geometry(g, tol)
            if g is None:
                warnings.append(f"{label}[{i}] dropped (invalid/near-zero).")
                continue
            if min_area_pixels is not None and g.area < min_area_pixels:
                warnings.append(f"{label}[{i}] dropped (< min_area_sqm).")
                continue
            out.append(g)
        return out

    def _clip_and_clean_batch(
        self,
        geoms: List[Union[Polygon, MultiPolygon]],
        aoi: Union[Polygon, MultiPolygon],
        warnings: List[str],
        label: str,
        tol: float,
        min_area_pixels: Optional[float]
    ) -> List[Union[Polygon, MultiPolygon]]:
        out: List[Union[Polygon, MultiPolygon]] = []
        for i, g in enumerate(geoms):
            try:
                clipped = g.intersection(aoi)
                cleaned = self._clean_geometry(clipped, tol)
                if cleaned is None:
                    warnings.append(f"{label}[{i}] dropped by AOI clipping (empty/near-zero).")
                    continue
                if min_area_pixels is not None and cleaned.area < min_area_pixels:
                    warnings.append(f"{label}[{i}] dropped by AOI (< min_area_sqm).")
                    continue
                out.append(cleaned)
            except Exception as e:
                warnings.append(f"{label}[{i}] AOI clipping error: {e}")
        return out

    # -----------------------------------
    #   Polygon-like → Shapely geometry
    # -----------------------------------

    def _create_polygon_like(
        self,
        coordinates: PolygonLike,
        warnings: List[str],
        label: str = "poly"
    ) -> Optional[Union[Polygon, MultiPolygon]]:
        try:
            if self._looks_like_simple_polygon(coordinates):
                ring = self._ensure_ring_closed(self._validate_ring(coordinates, warnings, f"{label}.outer"))
                if ring is None:
                    return None
                return Polygon(ring)

            if self._looks_like_polygon_with_holes(coordinates):
                outer = self._ensure_ring_closed(self._validate_ring(coordinates[0], warnings, f"{label}.outer"))
                if outer is None:
                    return None
                holes = []
                for hi, hole in enumerate(coordinates[1:]):
                    h = self._ensure_ring_closed(self._validate_ring(hole, warnings, f"{label}.hole[{hi}]"))
                    if h is not None:
                        holes.append(h)
                    else:
                        warnings.append(f"{label}.hole[{hi}] dropped (invalid).")
                return Polygon(shell=outer, holes=holes if holes else None)

            if self._looks_like_multipolygon(coordinates):
                polys = []
                for pi, poly_like in enumerate(coordinates):
                    if self._looks_like_simple_polygon(poly_like):
                        ring = self._ensure_ring_closed(self._validate_ring(poly_like, warnings, f"{label}.mp[{pi}].outer"))
                        if ring is not None:
                            polys.append(Polygon(ring))
                        else:
                            warnings.append(f"{label}.mp[{pi}] dropped (invalid outer).")
                    elif self._looks_like_polygon_with_holes(poly_like):
                        outer = self._ensure_ring_closed(self._validate_ring(poly_like[0], warnings, f"{label}.mp[{pi}].outer"))
                        if outer is None:
                            warnings.append(f"{label}.mp[{pi}] dropped (invalid outer).")
                            continue
                        holes = []
                        for hi, hole in enumerate(poly_like[1:]):
                            h = self._ensure_ring_closed(self._validate_ring(hole, warnings, f"{label}.mp[{pi}].hole[{hi}]"))
                            if h is not None:
                                holes.append(h)
                            else:
                                warnings.append(f"{label}.mp[{pi}].hole[{hi}] dropped (invalid).")
                        polys.append(Polygon(shell=outer, holes=holes if holes else None))
                    else:
                        warnings.append(f"{label}.mp[{pi}] invalid polygon-like structure.")
                if not polys:
                    return None
                if len(polys) == 1:
                    return polys[0]
                return MultiPolygon(polys)

            warnings.append(f"{label} has invalid coordinate structure.")
            return None

        except Exception as e:
            warnings.append(f"{label} creation failed: {e}")
            return None

    # -----------------------------------
    #        Coordinate validators
    # -----------------------------------

    def _looks_like_simple_polygon(self, coords: Any) -> bool:
        return (
            isinstance(coords, list)
            and len(coords) >= 3
            and all(isinstance(p, list) and len(p) >= 2 for p in coords)
            and not any(isinstance(p[0], list) for p in coords)
        )

    def _looks_like_polygon_with_holes(self, coords: Any) -> bool:
        return (
            isinstance(coords, list)
            and len(coords) >= 1
            and all(isinstance(r, list) and len(r) >= 3 for r in coords)
            and all(isinstance(p, list) and len(p) >= 2 for r in coords for p in r)
            and not any(isinstance(r[0][0], list) for r in coords)
        )

    def _looks_like_multipolygon(self, coords: Any) -> bool:
        if not (isinstance(coords, list) and len(coords) >= 1):
            return False
        first = coords[0]
        if not isinstance(first, list):
            return False
        return (
            isinstance(first[0], list)
            and (
                (len(first[0]) >= 2 and not isinstance(first[0][0], list))
                or (len(first[0]) >= 1 and isinstance(first[0][0], list))
            )
        )

    def _validate_ring(self, ring: List[List[float]], warnings: List[str], name: str) -> Optional[List[List[float]]]:
        try:
            cleaned: List[List[float]] = []
            for i, p in enumerate(ring):
                if not isinstance(p, list) or len(p) < 2:
                    warnings.append(f"{name}: point[{i}] invalid (need [x,y]).")
                    continue
                x, y = float(p[0]), float(p[1])
                if not (np.isfinite(x) and np.isfinite(y)):
                    warnings.append(f"{name}: point[{i}] not finite ({x},{y}).")
                    continue
                cleaned.append([x, y])

            if len(cleaned) < 3:
                warnings.append(f"{name}: insufficient valid points (<3).")
                return None

            # remove duplicate consecutive points
            dedup = [cleaned[0]]
            for q in cleaned[1:]:
                if q != dedup[-1]:
                    dedup.append(q)
            if len(dedup) < 3:
                warnings.append(f"{name}: degenerate after dedup (<3).")
                return None

            return dedup

        except Exception as e:
            warnings.append(f"{name}: ring validation failed: {e}")
            return None

    def _ensure_ring_closed(self, ring: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
        if ring is None:
            return None
        if ring[0] != ring[-1]:
            return ring + [ring[0]]
        return ring

    # -----------------------------------
    #            Cleaning
    # -----------------------------------

    def _clean_geometry(self, geom: Optional[Union[Polygon, MultiPolygon]], tol: float) -> Optional[Union[Polygon, MultiPolygon]]:
        if geom is None:
            return None
        try:
            g = geom
            if not g.is_valid:
                g = g.buffer(0)

            def parts(x):
                if isinstance(x, Polygon):
                    return [x]
                elif isinstance(x, MultiPolygon):
                    return list(x.geoms)
                else:
                    return []

            valid_parts = []
            for p in parts(g):
                if p.is_empty:
                    continue
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_empty:
                    continue
                if p.area <= tol:
                    continue
                valid_parts.append(p)

            if not valid_parts:
                return None
            if len(valid_parts) == 1:
                return valid_parts[0]
            return unary_union(valid_parts)

        except Exception:
            return None

    # -----------------------------------
    #         Distances & Summaries
    # -----------------------------------

    def _pairwise_distances(
        self,
        A: List[Union[Polygon, MultiPolygon]],
        B: List[Union[Polygon, MultiPolygon]],
        ids_a: Optional[List[str]],
        ids_b: Optional[List[str]],
        meters_per_pixel: float
    ):
        measurements: List[Dict[str, Any]] = []
        all_dist_pixels: List[float] = []

        global_min = {
            "distance_pixels": float("inf"),
            "source_id": None,
            "target_id": None,
            "closest_points": None
        }

        for i, ga in enumerate(A):
            sid = ids_a[i] if (ids_a is not None and i < len(ids_a)) else f"A_{i+1}"
            for j, gb in enumerate(B):
                tid = ids_b[j] if (ids_b is not None and j < len(ids_b)) else f"B_{j+1}"

                d_px = float(ga.distance(gb))
                p1, p2 = nearest_points(ga, gb)  # points on the boundaries achieving the shortest distance

                rec = {
                    "source_id": sid,
                    "target_id": tid,
                    "distance_pixels": round(d_px, 3),
                    "distance_meters": round(d_px * meters_per_pixel, 3),
                    "closest_points": [[p1.x, p1.y], [p2.x, p2.y]],
                    "intersects": d_px == 0.0
                }
                measurements.append(rec)
                all_dist_pixels.append(d_px)

                if d_px < global_min["distance_pixels"]:
                    global_min.update({
                        "distance_pixels": d_px,
                        "source_id": sid,
                        "target_id": tid,
                        "closest_points": rec["closest_points"]
                    })

        return measurements, all_dist_pixels, global_min

    def _per_source_nearest(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_source: Dict[str, Dict[str, Any]] = {}
        for m in measurements:
            sid = m["source_id"]
            if sid not in by_source or m["distance_pixels"] < by_source[sid]["distance_pixels"]:
                by_source[sid] = m
        # normalize to brief records
        out = []
        for sid, m in by_source.items():
            out.append({
                "source_id": sid,
                "target_id": m["target_id"],
                "distance_pixels": m["distance_pixels"],
                "distance_meters": m["distance_meters"],
                "closest_points": m["closest_points"],
                "intersects": m["intersects"]
            })
        return out

    # -----------------------------------
    #          Geometry previews
    # -----------------------------------

    def _geometry_previews(
        self,
        geoms_a: List[Union[Polygon, MultiPolygon]],
        geoms_b: List[Union[Polygon, MultiPolygon]],
        ids_a: Optional[List[str]],
        ids_b: Optional[List[str]],
        limit: Optional[int],
        include_holes: bool
    ) -> Dict[str, Any]:
        # control output size: total (A+B) no more than limit (if provided)
        cap = limit if (limit is not None and limit >= 0) else None
        out_a, out_b = [], []
        count = 0

        for i, g in enumerate(geoms_a):
            if cap is not None and count >= cap:
                break
            rec = {
                "id": (ids_a[i] if (ids_a is not None and i < len(ids_a)) else f"A_{i+1}"),
                "geometry": self._geom_to_coords(g, include_holes=include_holes)
            }
            out_a.append(rec)
            count += 1

        for j, g in enumerate(geoms_b):
            if cap is not None and count >= cap:
                break
            rec = {
                "id": (ids_b[j] if (ids_b is not None and j < len(ids_b)) else f"B_{j+1}"),
                "geometry": self._geom_to_coords(g, include_holes=include_holes)
            }
            out_b.append(rec)
            count += 1

        previews = {}
        if out_a:
            previews["set_a"] = out_a
        if out_b:
            previews["set_b"] = out_b
        return previews

    # -----------------------------------
    #            Utilities
    # -----------------------------------

    def _sqm_to_pixels(self, area_sqm: Optional[float], meters_per_pixel: float) -> Optional[float]:
        if area_sqm is None:
            return None
        return area_sqm / (meters_per_pixel ** 2)

    def _basic_stats(self, arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
        a = np.array(arr, dtype=float)
        return {
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a))
        }

    def _geom_to_coords(self, geom: Union[Polygon, MultiPolygon], include_holes: bool = True) -> PolygonLike:
        """
        Convert Polygon/MultiPolygon to coordinate arrays.
        - Polygon: returns [outer, hole1, ...] (if include_holes) or just outer ring
        - MultiPolygon: returns [[poly1], [poly2], ...], each poly follows above rule
        """
        def ring_to_list(r) -> List[List[float]]:
            return [[float(x), float(y)] for (x, y) in list(r.coords)]

        if isinstance(geom, Polygon):
            outer = ring_to_list(geom.exterior)
            if not include_holes or len(geom.interiors) == 0:
                return outer
            holes = [ring_to_list(h) for h in geom.interiors]
            return [outer] + holes

        elif isinstance(geom, MultiPolygon):
            polys = []
            for p in geom.geoms:
                if include_holes and len(p.interiors) > 0:
                    outer = ring_to_list(p.exterior)
                    holes = [ring_to_list(h) for h in p.interiors]
                    polys.append([outer] + holes)
                else:
                    polys.append(ring_to_list(p.exterior))
            return polys

        return []

    def _fail(self, msg: str, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "success": False,
            "tool": "distance",
            "error": msg,
            "metadata": {
                "validation_warnings": warnings or []
            }
        }
