"""
OverlapRatioTool (pure geometric)
Computes pairwise and union-based overlap metrics (IoU, coverage on A/B) between polygonal geometries.
I/O strictly in polygon coordinates (Polygon / MultiPolygon), no bbox/point/perception parsing, no filesystem I/O.
"""

from typing import List, Dict, Any, Tuple, Union, Optional, Type
from pydantic import BaseModel, Field, root_validator, validator
from langchain_core.tools import BaseTool
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np
import json


# -----------------------------
#           Schemas
# -----------------------------

PolygonLike = Union[
    List[List[float]],           # simple polygon: [[x,y], [x,y], ...]
    List[List[List[float]]]      # polygon with holes: [outer, hole1, ...]  OR multipolygon: [[poly1], [poly2], ...]
]

class OverlapGeometricInput(BaseModel):
    """
    Pure geometric input for overlap/IoU measurement.
    - class_a_polygons / class_b_polygons: list of polygon-like coordinate arrays.
      Each item can be:
        * Simple polygon: [[x,y], ...]
        * Polygon with holes: [outer, hole1, hole2, ...]
        * MultiPolygon: [[poly1], [poly2], ...]  (each poly is either simple or with holes)
    - aoi: optional polygon-like for clipping.
    - thresholds are defined in percentage (0-100).
    """
    class_a_polygons: List[PolygonLike] = Field(..., description="List of polygons (A)")
    class_b_polygons: List[PolygonLike] = Field(..., description="List of polygons (B)")
    aoi: Optional[PolygonLike] = Field(default=None, description="Optional Area-Of-Interest to clip inputs")

    meters_per_pixel: float = Field(default=0.3, gt=0, description="Ground resolution in meters per pixel")
    tolerance: float = Field(default=1e-6, gt=0, description="Area tolerance for filtering near-zero geometries")

    threshold_pct_on_A: Optional[float] = Field(default=None, ge=0, le=100, description="Optional coverage threshold on A (percentage)")
    threshold_pct_on_B: Optional[float] = Field(default=None, ge=0, le=100, description="Optional coverage threshold on B (percentage)")

    include_pairwise_intersections: bool = Field(default=True, description="Whether to include intersection geometries in pairwise output")
    pairwise_limit: Optional[int] = Field(default=None, description="Max number of pairwise records to include intersection geometry for (to cap payload size)")

    class_a_ids: Optional[List[str]] = Field(default=None, description="Optional IDs for A polygons (same length as class_a_polygons)")
    class_b_ids: Optional[List[str]] = Field(default=None, description="Optional IDs for B polygons (same length as class_b_polygons)")

    # Optional parameters for unified arguments field (from upstream tools)
    image_path: Optional[str] = Field(default=None, description="Path to input satellite image (for traceability)")
    query_text: Optional[str] = Field(default="", description="Original natural-language query")
    classes_used: Optional[List[str]] = Field(default=None, description="Classes involved in overlap analysis")
    class_a_name: Optional[str] = Field(default=None, description="Name of class A")
    class_b_name: Optional[str] = Field(default=None, description="Name of class B")

    @root_validator(skip_on_failure=True)
    def _validate_ids_length(cls, values):
        a_polys = values.get("class_a_polygons", [])
        b_polys = values.get("class_b_polygons", [])
        a_ids = values.get("class_a_ids", None)
        b_ids = values.get("class_b_ids", None)
        if a_ids is not None and len(a_ids) != len(a_polys):
            raise ValueError("class_a_ids length must match class_a_polygons length")
        if b_ids is not None and len(b_ids) != len(b_polys):
            raise ValueError("class_b_ids length must match class_b_polygons length")
        return values


# -----------------------------
#       Overlap Tool
# -----------------------------

class OverlapRatioTool(BaseTool):
    """
    Calculates overlap metrics between two sets of polygons:
      - Pairwise (A_i vs B_j): IoU, coverage_on_A, coverage_on_B, areas, optional intersection geometry
      - Union-based: coverage_on_A_union, coverage_on_B_union, iou_union (no double counting)
    Only polygon inputs are supported. This tool does no perception parsing or filesystem I/O.
    """
    name: str = "overlap"
    description: str = (
        "Pure geometric overlap tool. Accepts only polygon/multipolygon coordinates. "
        "Computes pairwise IoU & coverage (on A/B) and union-based coverage without double-counting. "
        "Optional AOI clipping and thresholds."
    )
    args_schema: Type[BaseModel] = OverlapGeometricInput

    # --------------- public API ---------------

    def _run(self, tool_input: Union[Dict[str, Any], str]) -> str:
        """
        JSON-friendly entry point: accepts coordinate arrays, returns JSON string.
        """
        try:
            params = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            inp = OverlapGeometricInput(**params)

            # Build shapely geometries
            warnings: List[str] = []
            A_geoms = self._build_and_clean_geoms(inp.class_a_polygons, inp.tolerance, warnings, label="A")
            B_geoms = self._build_and_clean_geoms(inp.class_b_polygons, inp.tolerance, warnings, label="B")

            if inp.aoi is not None:
                aoi_geom = self._create_polygon_like(inp.aoi, warnings, label="AOI")
                aoi_geom = self._clean_geometry(aoi_geom, inp.tolerance)
                if aoi_geom is None:
                    warnings.append("AOI geometry is invalid or near-zero after cleaning; ignoring AOI.")
                else:
                    A_geoms = self._clip_and_clean_batch(A_geoms, aoi_geom, inp.tolerance, warnings, label="A")
                    B_geoms = self._clip_and_clean_batch(B_geoms, aoi_geom, inp.tolerance, warnings, label="B")

            if not A_geoms:
                return json.dumps(self._fail("All A geometries were invalid/near-zero after cleaning/clipping", warnings), indent=2)
            if not B_geoms:
                return json.dumps(self._fail("All B geometries were invalid/near-zero after cleaning/clipping", warnings), indent=2)

            pairwise, stats = self._compute_pairwise(
                A_geoms, B_geoms,
                a_ids=inp.class_a_ids, b_ids=inp.class_b_ids,
                meters_per_pixel=inp.meters_per_pixel,
                include_intersections=inp.include_pairwise_intersections,
                pairwise_limit=inp.pairwise_limit,
                threshold_pct_on_A=inp.threshold_pct_on_A,
                threshold_pct_on_B=inp.threshold_pct_on_B
            )

            union_summary = self._compute_union_summary(
                A_geoms, B_geoms, inp.meters_per_pixel
            )

            # ========== ACTIONABLE SUMMARY FOR REACT TERMINATION ==========
            # Compute percentages that the LLM can use directly to answer queries
            coverage_a_pct = round(union_summary.get("coverage_on_A_union", 0) * 100, 1)
            coverage_b_pct = round(union_summary.get("coverage_on_B_union", 0) * 100, 1)
            iou_pct = round(union_summary.get("iou_union", 0) * 100, 1)

            # Count pairs with actual overlap
            pairs_with_actual_overlap = sum(1 for p in pairwise if p.get("has_overlap", False))

            # Build human-readable summary
            class_a_label = inp.class_a_name or "class_a"
            class_b_label = inp.class_b_name or "class_b"
            summary = (
                f"Overlap analysis: {coverage_a_pct}% of {class_a_label} overlaps with {class_b_label}, "
                f"{coverage_b_pct}% of {class_b_label} overlaps with {class_a_label}. "
                f"{pairs_with_actual_overlap} of {len(pairwise)} geometry pairs have overlap."
            )
            # ========== END ACTIONABLE SUMMARY ==========

            result = {
                "success": True,
                "tool": "overlap",
                "summary": summary,  # NEW: Actionable summary for LLM
                "coverage_a_percent": coverage_a_pct,  # NEW: Direct percentage values
                "coverage_b_percent": coverage_b_pct,
                "iou_percent": iou_pct,
                "pairs_with_overlap": pairs_with_actual_overlap,
                "input": {
                    "len_A": len(A_geoms),
                    "len_B": len(B_geoms),
                    "meters_per_pixel": inp.meters_per_pixel,
                    "tolerance": inp.tolerance,
                    "threshold_pct_on_A": inp.threshold_pct_on_A,
                    "threshold_pct_on_B": inp.threshold_pct_on_B,
                    "include_pairwise_intersections": inp.include_pairwise_intersections,
                    "pairwise_limit": inp.pairwise_limit
                },
                "output": {
                    "pairwise": pairwise,
                    "union_summary": union_summary,
                    "stats": stats
                },
                "metadata": {
                    "validation_warnings": warnings
                }
            }

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics
            # This matches the format used in perception tools, BufferTool, and ContainmentTool for consistency

            # Determine class A and B names
            class_a_name = inp.class_a_name or "class_a"
            class_b_name = inp.class_b_name or "class_b"

            # Build classes_used list from class A and B names
            if inp.classes_used is not None:
                classes_used_list = inp.classes_used
            else:
                classes_used_list = [class_a_name, class_b_name]

            # Sort classes alphabetically for consistency
            classes_used_sorted = sorted(classes_used_list)

            # Count pairs with overlap (intersection area > 0)
            pairs_with_overlap = 0
            for pair in pairwise:
                if pair.get("overlap_pct", 0) > 0 or pair.get("iou", 0) > 0:
                    pairs_with_overlap += 1

            # Build the unified arguments field
            arguments = {
                "image_path": inp.image_path,
                "classes_used": classes_used_sorted,
                "meters_per_pixel": inp.meters_per_pixel if inp.meters_per_pixel else None,
                "class_a": class_a_name,
                "class_b": class_b_name,
                "total_a_polygons": len(A_geoms),
                "total_b_polygons": len(B_geoms),
                "pairs_with_overlap": pairs_with_overlap
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps(self._fail(f"Overlap analysis failed: {e}"), indent=2)

    # --------------- helpers: I/O & cleaning ---------------

    def _build_and_clean_geoms(
        self,
        items: List[PolygonLike],
        tol: float,
        warnings: List[str],
        label: str
    ) -> List[Union[Polygon, MultiPolygon]]:
        geoms: List[Union[Polygon, MultiPolygon]] = []
        for idx, coords in enumerate(items):
            g = self._create_polygon_like(coords, warnings, label=f"{label}[{idx}]")
            g = self._clean_geometry(g, tol)
            if g is None:
                warnings.append(f"{label}[{idx}] dropped (invalid or near-zero after cleaning).")
            else:
                geoms.append(g)
        return geoms

    def _clip_and_clean_batch(
        self,
        geoms: List[Union[Polygon, MultiPolygon]],
        aoi: Union[Polygon, MultiPolygon],
        tol: float,
        warnings: List[str],
        label: str
    ) -> List[Union[Polygon, MultiPolygon]]:
        out: List[Union[Polygon, MultiPolygon]] = []
        for i, g in enumerate(geoms):
            try:
                clipped = g.intersection(aoi)
                cleaned = self._clean_geometry(clipped, tol)
                if cleaned is not None:
                    out.append(cleaned)
                else:
                    warnings.append(f"{label}[{i}] dropped by AOI clipping (empty/near-zero).")
            except Exception as e:
                warnings.append(f"{label}[{i}] AOI clipping error: {e}")
        return out

    # --------------- helpers: geometry creation ---------------

    def _create_polygon_like(
        self,
        coordinates: PolygonLike,
        warnings: List[str],
        label: str = "poly"
    ) -> Optional[Union[Polygon, MultiPolygon]]:
        """
        Build Polygon/MultiPolygon from coordinates.
        Accepted formats:
          - Simple polygon: [[x,y], ...]
          - Polygon with holes: [outer, hole1, ...]
          - MultiPolygon: [[poly1], [poly2], ...]  (each poly can be simple or with holes)
        """
        try:
            if self._looks_like_simple_polygon(coordinates):
                # simple polygon -> Polygon
                ring = self._ensure_ring_closed(self._validate_ring(coordinates, warnings, f"{label}.outer"))
                if ring is None:
                    return None
                return Polygon(ring)

            if self._looks_like_polygon_with_holes(coordinates):
                # [outer, hole1, ...]
                outer = self._ensure_ring_closed(self._validate_ring(coordinates[0], warnings, f"{label}.outer"))
                if outer is None:
                    return None
                holes = []
                for hi, hole in enumerate(coordinates[1:]):
                    h_ring = self._ensure_ring_closed(self._validate_ring(hole, warnings, f"{label}.hole[{hi}]"))
                    if h_ring is not None:
                        holes.append(h_ring)
                    else:
                        warnings.append(f"{label}.hole[{hi}] dropped (invalid).")
                return Polygon(shell=outer, holes=holes if holes else None)

            if self._looks_like_multipolygon(coordinates):
                # [[poly1], [poly2], ...]  -> MultiPolygon
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
                            h_ring = self._ensure_ring_closed(self._validate_ring(hole, warnings, f"{label}.mp[{pi}].hole[{hi}]"))
                            if h_ring is not None:
                                holes.append(h_ring)
                            else:
                                warnings.append(f"{label}.mp[{pi}].hole[{hi}] dropped (invalid).")
                        polys.append(Polygon(shell=outer, holes=holes if holes else None))
                    else:
                        warnings.append(f"{label}.mp[{pi}] is not a valid polygon-like structure.")
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

    # --------------- helpers: coordinate validators ---------------

    def _looks_like_simple_polygon(self, coords: Any) -> bool:
        return (
            isinstance(coords, list)
            and len(coords) >= 3
            and all(isinstance(p, list) and len(p) >= 2 for p in coords)
            and not any(isinstance(p[0], list) for p in coords)  # not nested rings
        )

    def _looks_like_polygon_with_holes(self, coords: Any) -> bool:
        # [outer, hole1, ...] where each is a ring (list of points)
        return (
            isinstance(coords, list)
            and len(coords) >= 1
            and all(isinstance(r, list) and len(r) >= 3 for r in coords)
            and all(isinstance(p, list) and len(p) >= 2 for r in coords for p in r)
            and not any(isinstance(r[0][0], list) for r in coords)  # not multipolygon
        )

    def _looks_like_multipolygon(self, coords: Any) -> bool:
        # [[poly1], [poly2], ...] where each poly is either a ring-list or list-of-rings
        if not (isinstance(coords, list) and len(coords) >= 1):
            return False
        first = coords[0]
        if not isinstance(first, list):
            return False
        # multipolygon: first item itself is a ring or list-of-rings
        # i.e., coords[0][0] is a point (len>=2) OR coords[0][0] is a ring (list of points)
        return (
            isinstance(first[0], list)
            and (
                (len(first[0]) >= 2 and not isinstance(first[0][0], list))  # [[ [x,y], [x,y], ... ], ...]
                or (len(first[0]) >= 1 and isinstance(first[0][0], list))   # [[ [ [x,y],... ], [ [x,y],... ] ], ...]
            )
        )

    def _validate_ring(self, ring: List[List[float]], warnings: List[str], name: str) -> Optional[List[List[float]]]:
        """Validate a ring: finite numbers, >=3 points, non-zero area after cleanup."""
        try:
            cleaned: List[List[float]] = []
            for i, p in enumerate(ring):
                if not isinstance(p, list) or len(p) < 2:
                    warnings.append(f"{name}: point[{i}] is invalid (need [x,y]).")
                    continue
                x, y = float(p[0]), float(p[1])
                if not (np.isfinite(x) and np.isfinite(y)):
                    warnings.append(f"{name}: point[{i}] not finite ({x},{y}).")
                    continue
                cleaned.append([x, y])

            if len(cleaned) < 3:
                warnings.append(f"{name}: insufficient valid points (<3).")
                return None

            # Remove duplicated consecutive points (excluding closure)
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

    # --------------- helpers: cleaning ---------------

    def _clean_geometry(self, geom: Optional[Union[Polygon, MultiPolygon]], tol: float) -> Optional[Union[Polygon, MultiPolygon]]:
        """Fix invalid geometry via buffer(0), explode MultiPolygon, filter near-zero area parts, then re-assemble."""
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
                # filter small or empty
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

    # --------------- helpers: metrics & stats ---------------

    def _compute_pairwise(
        self,
        A: List[Union[Polygon, MultiPolygon]],
        B: List[Union[Polygon, MultiPolygon]],
        a_ids: Optional[List[str]],
        b_ids: Optional[List[str]],
        meters_per_pixel: float,
        include_intersections: bool,
        pairwise_limit: Optional[int],
        threshold_pct_on_A: Optional[float],
        threshold_pct_on_B: Optional[float]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        pairwise: List[Dict[str, Any]] = []
        iou_vals, covA_vals, covB_vals = [], [], []
        px2m = meters_per_pixel ** 2

        # limit control
        include_geom_counter = 0
        include_geom_max = pairwise_limit if (pairwise_limit is not None and pairwise_limit >= 0) else None

        for i, a in enumerate(A):
            for j, b in enumerate(B):
                # fix invalid before ops
                aa = a if a.is_valid else a.buffer(0)
                bb = b if b.is_valid else b.buffer(0)
                try:
                    inter = aa.intersection(bb)
                    if not inter.is_empty and not inter.is_valid:
                        inter = inter.buffer(0)
                    inter_area = inter.area if not inter.is_empty else 0.0
                    area_a = aa.area
                    area_b = bb.area
                    uni = aa.union(bb)
                    if not uni.is_valid:
                        uni = uni.buffer(0)
                    union_area = uni.area
                except Exception:
                    # on topology failure, skip this pair
                    continue

                iou = (inter_area / union_area) if union_area > 0 else 0.0
                covA = (inter_area / area_a) if area_a > 0 else 0.0
                covB = (inter_area / area_b) if area_b > 0 else 0.0

                # thresholds (percent)
                meets_A = (threshold_pct_on_A is None) or ((covA * 100.0) >= threshold_pct_on_A)
                meets_B = (threshold_pct_on_B is None) or ((covB * 100.0) >= threshold_pct_on_B)

                # optional intersection coordinates (to reduce payload)
                inter_coords = None
                if include_intersections and inter_area > 0:
                    if (include_geom_max is None) or (include_geom_counter < include_geom_max):
                        inter_coords = self._geom_to_coords(inter, include_holes=False)
                        include_geom_counter += 1

                a_id = a_ids[i] if (a_ids is not None and i < len(a_ids)) else f"A_{i+1}"
                b_id = b_ids[j] if (b_ids is not None and j < len(b_ids)) else f"B_{j+1}"

                record = {
                    "a_id": a_id,
                    "b_id": b_id,
                    "has_overlap": inter_area > 0,
                    "meets_threshold_on_A": bool(meets_A),
                    "meets_threshold_on_B": bool(meets_B),
                    "metrics": {
                        "iou": round(iou, 6),
                        "coverage_on_A": round(covA, 6),
                        "coverage_on_B": round(covB, 6)
                    },
                    "areas": {
                        "intersection_pixels": round(inter_area, 2),
                        "intersection_sqm": round(inter_area * px2m, 2),
                        "area_A_pixels": round(area_a, 2),
                        "area_A_sqm": round(area_a * px2m, 2),
                        "area_B_pixels": round(area_b, 2),
                        "area_B_sqm": round(area_b * px2m, 2),
                        "union_pixels": round(union_area, 2),
                        "union_sqm": round(union_area * px2m, 2)
                    }
                }
                if inter_coords is not None:
                    record["intersection_geometry"] = inter_coords

                pairwise.append(record)
                iou_vals.append(iou)
                covA_vals.append(covA)
                covB_vals.append(covB)

        stats = {
            "pairwise_iou": self._basic_stats(iou_vals),
            "pairwise_coverage_on_A": self._basic_stats(covA_vals),
            "pairwise_coverage_on_B": self._basic_stats(covB_vals),
            "pairwise_count": len(pairwise),
            "pairwise_intersection_geometries_included": include_intersections and (
                (include_geom_max is None) or (include_geom_counter <= include_geom_max)
            )
        }
        return pairwise, stats

    def _compute_union_summary(
        self,
        A: List[Union[Polygon, MultiPolygon]],
        B: List[Union[Polygon, MultiPolygon]],
        meters_per_pixel: float
    ) -> Dict[str, Any]:
        px2m = meters_per_pixel ** 2

        union_A = unary_union(A) if len(A) > 1 else A[0]
        union_B = unary_union(B) if len(B) > 1 else B[0]
        if not union_A.is_valid:
            union_A = union_A.buffer(0)
        if not union_B.is_valid:
            union_B = union_B.buffer(0)

        inter = union_A.intersection(union_B)
        if not inter.is_empty and not inter.is_valid:
            inter = inter.buffer(0)

        area_A = union_A.area
        area_B = union_B.area
        inter_area = inter.area if not inter.is_empty else 0.0
        union_AB = union_A.union(union_B)
        if not union_AB.is_valid:
            union_AB = union_AB.buffer(0)
        area_union = union_AB.area

        cov_on_A = (inter_area / area_A) if area_A > 0 else 0.0
        cov_on_B = (inter_area / area_B) if area_B > 0 else 0.0
        iou_union = (inter_area / area_union) if area_union > 0 else 0.0

        return {
            "coverage_on_A_union": round(cov_on_A, 6),
            "coverage_on_B_union": round(cov_on_B, 6),
            "iou_union": round(iou_union, 6),
            "areas": {
                "union_A_pixels": round(area_A, 2),
                "union_A_sqm": round(area_A * px2m, 2),
                "union_B_pixels": round(area_B, 2),
                "union_B_sqm": round(area_B * px2m, 2),
                "intersection_pixels": round(inter_area, 2),
                "intersection_sqm": round(inter_area * px2m, 2),
                "union_AB_pixels": round(area_union, 2),
                "union_AB_sqm": round(area_union * px2m, 2)
            }
        }

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

    # --------------- helpers: geometry → coordinates ---------------

    def _geom_to_coords(self, geom: Union[Polygon, MultiPolygon], include_holes: bool = False) -> PolygonLike:
        """
        Convert Polygon/MultiPolygon to coordinate arrays.
        - Polygon: returns [outer, hole1, ...] (if include_holes) or just outer ring if include_holes=False
        - MultiPolygon: returns [[poly1], [poly2], ...] each poly as above
        """
        def ring_to_list(r) -> List[List[float]]:
            coords = list(r.coords)
            return [[float(x), float(y)] for (x, y) in coords]

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

        # Fallback (shouldn't happen)
        return []

    # --------------- helpers: failures ---------------

    def _fail(self, msg: str, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "success": False,
            "tool": "overlap",
            "error": msg,
            "metadata": {
                "validation_warnings": warnings or []
            }
        }
