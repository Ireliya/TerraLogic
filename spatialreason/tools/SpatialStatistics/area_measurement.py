"""
AreaMeasurementTool (pure geometric)
Measures areas for polygonal geometries (Polygon / MultiPolygon only).
- I/O strictly in polygon coordinates
- Optional AOI clipping
- Robust geometry cleaning & tiny-area filtering
- Reports per-object areas and union-based area to avoid double counting
"""

from typing import List, Dict, Any, Union, Optional, Type
from pydantic import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np
import json


# -----------------------------------
#            Schemas
# -----------------------------------

PolygonLike = Union[
    List[List[float]],           # simple polygon: [[x,y], [x,y], ...]
    List[List[List[float]]]      # polygon with holes: [outer, hole1, ...]  OR multipolygon: [[poly1], [poly2], ...]
]

class AreaGeometricInput(BaseModel):
    """
    Pure geometric input for area measurement.
    Each item in `polygons` can be:
      - Simple polygon: [[x,y], ...]
      - Polygon with holes: [outer, hole1, ...]
      - MultiPolygon: [[poly1], [poly2], ...] (each poly can be simple or with holes)
    """
    polygons: List[PolygonLike] = Field(..., description="List of polygon-like coordinate arrays")
    aoi: Optional[PolygonLike] = Field(default=None, description="Optional Area-Of-Interest polygon-like for clipping")

    meters_per_pixel: float = Field(default=0.3, gt=0, description="Ground resolution in meters per pixel")
    tolerance: float = Field(default=1e-6, gt=0, description="Area tolerance (in pixel^2) for tiny parts filtering")
    min_area_sqm: Optional[float] = Field(default=None, ge=0, description="Drop parts/objects smaller than this area (square meters)")

    polygon_ids: Optional[List[str]] = Field(default=None, description="Optional IDs for polygons (same length as polygons)")

    include_cleaned_geometries: bool = Field(default=False, description="Whether to include cleaned geometry coordinates in output objects")
    cleaned_geometries_limit: Optional[int] = Field(default=None, description="Max number of objects to include geometry for (controls payload size)")
    include_holes_in_geometry: bool = Field(default=True, description="If returning geometries, include holes")

    # Optional parameters for unified arguments field (from upstream tools)
    image_path: Optional[str] = Field(default=None, description="Path to input satellite image (for traceability)")
    query_text: Optional[str] = Field(default="", description="Original natural-language query")
    classes_used: Optional[List[str]] = Field(default=None, description="Semantic classes involved in area computation")

    @root_validator(skip_on_failure=True)
    def _validate_ids(cls, values):
        polys = values.get("polygons", [])
        ids_ = values.get("polygon_ids", None)
        if ids_ is not None and len(ids_) != len(polys):
            raise ValueError("polygon_ids length must match polygons length")
        return values


# -----------------------------------
#        Area Measurement Tool
# -----------------------------------

class AreaMeasurementTool(BaseTool):
    """
    Pure geometric area measurement:
      - Per-object areas (pixels & m^2)
      - Union-based area (no double counting)
      - AOI clipping, geometry cleaning, tiny-area filtering
    I/O strictly uses polygon coordinates; no image IO or perception parsing.
    """
    name: str = "area_measurement"
    description: str = (
        "Measure areas of polygonal geometries (Polygon/MultiPolygon only). "
        "Supports AOI clipping, robust cleaning, tiny-area filtering, and returns both per-object areas and union area."
    )
    args_schema: Type[BaseModel] = AreaGeometricInput

    # -------- public JSON API --------

    def _run(self, tool_input: Union[Dict[str, Any], str]) -> str:
        try:
            params = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            inp = AreaGeometricInput(**params)

            warnings: List[str] = []
            # Build & clean inputs
            geoms = self._build_and_clean_batch(
                inp.polygons, warnings, label="P",
                tol=inp.tolerance,
                min_area_pixels=self._sqm_to_pixels(inp.min_area_sqm, inp.meters_per_pixel) if inp.min_area_sqm is not None else None
            )

            # AOI clipping
            if inp.aoi is not None:
                aoi = self._create_polygon_like(inp.aoi, warnings, label="AOI")
                aoi = self._clean_geometry(aoi, inp.tolerance)
                if aoi is None:
                    warnings.append("AOI is invalid or near-zero after cleaning; ignoring AOI.")
                else:
                    geoms = self._clip_and_clean_batch(
                        geoms, aoi, warnings, label="P", tol=inp.tolerance,
                        min_area_pixels=self._sqm_to_pixels(inp.min_area_sqm, inp.meters_per_pixel) if inp.min_area_sqm is not None else None
                    )

            # CRITICAL FIX: When all geometries are clipped away by AOI (e.g., no overlap between buffer and target class),
            # return success with 0 area instead of failing. This is a valid result indicating no intersection.
            if not geoms:
                warnings.append("All geometries clipped away by AOI - no intersection between target class and buffer zone.")
                # Return success with 0 area
                result = {
                    "success": True,
                    "tool": "area_measurement",
                    "input": {
                        "len_polygons": len(inp.polygons),
                        "meters_per_pixel": inp.meters_per_pixel,
                        "tolerance": inp.tolerance,
                        "min_area_sqm": inp.min_area_sqm,
                        "include_cleaned_geometries": inp.include_cleaned_geometries,
                        "cleaned_geometries_limit": inp.cleaned_geometries_limit,
                        "include_holes_in_geometry": inp.include_holes_in_geometry,
                        "aoi_provided": inp.aoi is not None
                    },
                    "output": {
                        "objects": [],
                        "union_summary": {
                            "total_area_pixels": 0.0,
                            "total_area_sqm": 0.0,
                            "object_count": 0,
                            "note": "No geometries after AOI clipping"
                        },
                        "stats": {
                            "per_object_area_pixels": {"count": 0, "min": None, "max": None, "mean": None, "sum": 0.0},
                            "per_object_area_sqm": {"count": 0, "min": None, "max": None, "mean": None, "sum": 0.0}
                        }
                    },
                    "metadata": {
                        "validation_warnings": warnings
                    }
                }
                return json.dumps(result, indent=2)

            # Compute per-object areas
            objects, areas_px, areas_m2 = self._per_object_areas(
                geoms,
                ids=inp.polygon_ids,
                meters_per_pixel=inp.meters_per_pixel,
                include_geom=inp.include_cleaned_geometries,
                include_holes=inp.include_holes_in_geometry,
                geom_limit=inp.cleaned_geometries_limit
            )

            # Compute union-based area (no double counting)
            union_summary = self._union_area_summary(geoms, meters_per_pixel=inp.meters_per_pixel)

            # Stats on per-object areas
            stats = {
                "per_object_area_pixels": self._basic_stats(areas_px),
                "per_object_area_sqm": self._basic_stats(areas_m2)
            }

            result = {
                "success": True,
                "tool": "area_measurement",
                "input": {
                    "len_polygons": len(inp.polygons),
                    "meters_per_pixel": inp.meters_per_pixel,
                    "tolerance": inp.tolerance,
                    "min_area_sqm": inp.min_area_sqm,
                    "include_cleaned_geometries": inp.include_cleaned_geometries,
                    "cleaned_geometries_limit": inp.cleaned_geometries_limit,
                    "include_holes_in_geometry": inp.include_holes_in_geometry,
                    "aoi_provided": inp.aoi is not None
                },
                "output": {
                    "objects": objects,
                    "union_summary": union_summary,
                    "stats": stats
                },
                "metadata": {
                    "validation_warnings": warnings
                }
            }

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics
            # This matches the format used in perception tools, BufferTool, ContainmentTool, and OverlapTool for consistency

            # Build classes_used list
            if inp.classes_used is not None:
                classes_used_list = inp.classes_used
            else:
                classes_used_list = []

            # Sort classes alphabetically for consistency
            classes_used_sorted = sorted(classes_used_list)

            # Get union area from summary
            union_area_sqm = union_summary.get("union_area_sqm", 0.0)

            # Build the unified arguments field
            arguments = {
                "image_path": inp.image_path,
                "classes_used": classes_used_sorted,
                "meters_per_pixel": inp.meters_per_pixel if inp.meters_per_pixel else None,
                "total_polygons": len(geoms),
                "union_area_sqm": union_area_sqm
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps(self._fail(f"Area measurement failed: {e}"), indent=2)

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
        # Enhanced logging for diagnostics
        input_count = len(geoms)
        dropped_count = 0

        for i, g in enumerate(geoms):
            try:
                clipped = g.intersection(aoi)
                cleaned = self._clean_geometry(clipped, tol)
                if cleaned is None:
                    warnings.append(f"{label}[{i}] dropped by AOI clipping (empty/near-zero).")
                    dropped_count += 1
                    continue
                if min_area_pixels is not None and cleaned.area < min_area_pixels:
                    warnings.append(f"{label}[{i}] dropped by AOI (< min_area_sqm).")
                    dropped_count += 1
                    continue
                out.append(cleaned)
            except Exception as e:
                warnings.append(f"{label}[{i}] AOI clipping error: {e}")
                dropped_count += 1

        # Add summary diagnostic info
        if input_count > 0 and len(out) == 0:
            warnings.append(f"AOI clipping result: {input_count} input geometries → {len(out)} output geometries (all dropped). This typically indicates no spatial overlap between target class and buffer zone.")

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
    #         Areas & Summaries
    # -----------------------------------

    def _per_object_areas(
        self,
        geoms: List[Union[Polygon, MultiPolygon]],
        ids: Optional[List[str]],
        meters_per_pixel: float,
        include_geom: bool,
        include_holes: bool,
        geom_limit: Optional[int]
    ) -> (List[Dict[str, Any]], List[float], List[float]):
        px2m = meters_per_pixel ** 2
        objs: List[Dict[str, Any]] = []
        areas_px: List[float] = []
        areas_m2: List[float] = []

        emit_count = 0
        emit_max = geom_limit if (geom_limit is not None and geom_limit >= 0) else None

        for i, g in enumerate(geoms):
            area_px = float(g.area)
            area_m2 = float(area_px * px2m)

            rec: Dict[str, Any] = {
                "id": (ids[i] if (ids is not None and i < len(ids)) else f"P_{i+1}"),
                "area_pixels": round(area_px, 2),
                "area_sqm": round(area_m2, 2),
                "geometry_type": "MultiPolygon" if isinstance(g, MultiPolygon) else "Polygon",
            }

            if include_geom and (emit_max is None or emit_count < emit_max):
                rec["geometry"] = self._geom_to_coords(g, include_holes=include_holes)
                emit_count += 1

            objs.append(rec)
            areas_px.append(area_px)
            areas_m2.append(area_m2)

        return objs, areas_px, areas_m2

    def _union_area_summary(
        self,
        geoms: List[Union[Polygon, MultiPolygon]],
        meters_per_pixel: float
    ) -> Dict[str, Any]:
        px2m = meters_per_pixel ** 2
        union_g = unary_union(geoms) if len(geoms) > 1 else geoms[0]
        if not union_g.is_valid:
            union_g = union_g.buffer(0)

        area_union_px = float(union_g.area)
        area_union_m2 = float(area_union_px * px2m)

        # sum of parts (可能重复计数)
        sum_parts_px = float(sum(g.area for g in geoms))
        sum_parts_m2 = float(sum_parts_px * px2m)

        return {
            "union_area_pixels": round(area_union_px, 2),
            "union_area_sqm": round(area_union_m2, 2),
            "sum_of_parts_pixels": round(sum_parts_px, 2),
            "sum_of_parts_sqm": round(sum_parts_m2, 2),
            "overlap_loss_pixels": round(sum_parts_px - area_union_px, 2),
            "overlap_loss_sqm": round(sum_parts_m2 - area_union_m2, 2)
        }

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
            "tool": "area_measurement",
            "error": msg,
            "metadata": {
                "validation_warnings": warnings or []
            }
        }
