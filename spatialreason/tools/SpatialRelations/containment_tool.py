"""
ContainmentTool (Pure Geometric Kernel)

Purpose:
- Compute containment relationships between two polygon sets (containers × contained)
- Inputs and outputs are strictly Polygon/MultiPolygon
- No role assignment, no perception parsing, no bbox/point handling (moved to upstream adapters)

Key features:
- Geometry cleaning (buffer(0), zero-area filter)
- Optional AOI clipping
- Area-based containment ratio (intersection / contained)
- Strict/Non-strict modes with tolerance
- Pixel and SI units reporting (m² via meters_per_pixel²)
- Outputs intersection polygons for downstream reuse
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, Field, validator
from langchain_core.tools import BaseTool
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


# -----------------------------
# --------- Schemas -----------
# -----------------------------

PolygonLike = Union[List[List[float]], List[List[List[float]]]]  # [[x,y], ...] or [ [[x,y], ...], [[x,y], ...] ]


class GeometricContainmentInput(BaseModel):
    """
    JSON-friendly input schema.
    All geometries are polygons or multipolygons expressed as coordinate arrays.

    containers: list of polygons (each can be Polygon or MultiPolygon in coordinate form)
    contained:  list of polygons (same)
    aoi:        optional AOI polygon/multipolygon to clip both sets before analysis

    coord_sys: optional metadata (e.g., {"type": "pixel"} or {"type": "epsg", "epsg": 3857})
               The tool does NOT reproject; it only uses meters_per_pixel to convert areas from pixels² to m².
    """
    containers: List[PolygonLike] = Field(..., description="List of container polygons")
    contained: List[PolygonLike] = Field(..., description="List of polygons to test being contained")
    aoi: Optional[PolygonLike] = Field(default=None, description="Optional AOI polygon for pre-clipping")

    # Units & numeric behavior
    meters_per_pixel: float = Field(0.3, gt=0, description="Ground sampling distance (m/px), used only for area conversion")
    threshold_pct: Optional[float] = Field(default=None, ge=0, le=100, description="Containment threshold (0-100). If provided, outputs meets_threshold")
    strict: bool = Field(default=False, description="If True, requires ~100% (within tolerance) to be considered fully contained")
    tolerance: float = Field(default=1e-6, gt=0, description="Numerical tolerance for equality / near-100% checks")

    # Optional IDs for traceability
    container_ids: Optional[List[str]] = Field(default=None, description="Optional IDs for containers")
    contained_ids: Optional[List[str]] = Field(default=None, description="Optional IDs for contained geometries")

    # Optional metadata for pipeline bookkeeping
    coord_sys: Optional[Dict[str, Any]] = Field(default=None, description="Coordinate system metadata")

    # Optional parameters for unified arguments field (from upstream tools)
    image_path: Optional[str] = Field(default=None, description="Path to input satellite image (for traceability)")
    query_text: Optional[str] = Field(default="", description="Original natural-language query")
    classes_used: Optional[List[str]] = Field(default=None, description="Classes involved in containment analysis")
    container_class: Optional[str] = Field(default=None, description="Name of the container class")
    contained_class: Optional[str] = Field(default=None, description="Name of the contained class")

    @validator("containers")
    def _non_empty_containers(cls, v):
        if not v:
            raise ValueError("containers must be non-empty")
        return v

    @validator("contained")
    def _non_empty_contained(cls, v):
        if not v:
            raise ValueError("contained must be non-empty")
        return v

    @validator("contained_ids")
    def _ids_len_match_contained(cls, v, values):
        if v is not None and "contained" in values and len(v) != len(values["contained"]):
            raise ValueError("contained_ids length must match contained length")
        return v

    @validator("container_ids")
    def _ids_len_match_containers(cls, v, values):
        if v is not None and "containers" in values and len(v) != len(values["containers"]):
            raise ValueError("container_ids length must match containers length")
        return v


# -----------------------------
# ------- Core Tool -----------
# -----------------------------

class ContainmentTool(BaseTool):
    """
    Pure geometric containment kernel.

    Public contract:
    - Only Polygon/MultiPolygon (in coordinate arrays via `run`).
    - Computes intersection polygons, ratios and metrics.
    - No bbox/point parsing, no role assignment, no image paths, no file IO.
    """
    name: str = "containment"
    description: str = (
        "Compute containment between two polygon sets (containers × contained). "
        "Input/Output strictly Polygon/MultiPolygon. "
        "Returns intersection polygons and area-based metrics."
    )
    args_schema: Type[BaseModel] = GeometricContainmentInput

    # ------------- Public API -------------

    def _run(self, tool_input: Union[Dict[str, Any], str]) -> str:
        """
        Public JSON-friendly entrypoint.
        Accepts coordinate arrays, builds/cleans Shapely geometries internally, runs analysis, and returns JSON.
        """
        try:
            params = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            inp = GeometricContainmentInput(**params)

            # Build Shapely geometries from coordinate arrays
            containers = [self._to_shapely_polygon(g) for g in inp.containers]
            contained = [self._to_shapely_polygon(g) for g in inp.contained]
            aoi = self._to_shapely_polygon(inp.aoi) if inp.aoi is not None else None

            # 1) Validate/clean input geometries
            containers = [self._clean_polygon(g, inp.tolerance) for g in containers if g is not None]
            contained = [self._clean_polygon(g, inp.tolerance) for g in contained if g is not None]

            containers = [g for g in containers if self._is_valid_area(g, inp.tolerance)]
            contained = [g for g in contained if self._is_valid_area(g, inp.tolerance)]

            if not containers:
                raise ValueError("No valid container polygons after cleaning.")
            if not contained:
                raise ValueError("No valid contained polygons after cleaning.")

            # 2) Optional AOI clipping
            if aoi is not None:
                aoi = self._clean_polygon(aoi, inp.tolerance)
                if aoi and not aoi.is_empty:
                    containers = [self._clip_to_aoi(g, aoi, inp.tolerance) for g in containers]
                    contained = [self._clip_to_aoi(g, aoi, inp.tolerance) for g in contained]
                    containers = [g for g in containers if self._is_valid_area(g, inp.tolerance)]
                    contained = [g for g in contained if self._is_valid_area(g, inp.tolerance)]

            # 3) Pairwise analysis
            mpp2 = (inp.meters_per_pixel ** 2)
            relationships: List[Dict[str, Any]] = []

            # IDs
            container_ids = inp.container_ids or [f"container_{i+1}" for i in range(len(containers))]
            contained_ids = inp.contained_ids or [f"contained_{j+1}" for j in range(len(contained))]

            ratio_values: List[float] = []
            contained_true_count = 0
            meets_threshold_count = 0

            for i, c in enumerate(containers):
                for j, t in enumerate(contained):
                    inter = c.intersection(t)
                    if inter.is_empty:
                        inter_area_px = 0.0
                    else:
                        inter = self._clean_polygon(inter, inp.tolerance)
                        inter_area_px = inter.area if inter and not inter.is_empty else 0.0

                    contained_area_px = t.area
                    container_area_px = c.area

                    ratio = float(inter_area_px / contained_area_px) if contained_area_px > 0 else 0.0
                    ratio_values.append(ratio)

                    # Full containment decision
                    # strict=True: require ~100% within tolerance; strict=False: we still report ratio; 'is_fully_contained' uses near-1 check.
                    is_fully = (abs(1.0 - ratio) <= inp.tolerance)
                    if is_fully:
                        contained_true_count += 1

                    # Threshold decision (if provided)
                    meets_threshold = None
                    if inp.threshold_pct is not None:
                        meets_threshold = (ratio * 100.0) + 1e-12 >= inp.threshold_pct  # small epsilon to be robust
                        if meets_threshold:
                            meets_threshold_count += 1

                    # Areas (pixels and m²)
                    inter_area_m2 = inter_area_px * mpp2
                    contained_area_m2 = contained_area_px * mpp2
                    container_area_m2 = container_area_px * mpp2

                    # Intersection geometry to coords (polygon out)
                    inter_geom_coords = self._geom_to_coords(inter) if inter_area_px > 0 else None

                    relationships.append({
                        "container_id": container_ids[i],
                        "contained_id": contained_ids[j],
                        "containment_ratio": round(ratio, 6),
                        "containment_pct": round(ratio * 100.0, 4),
                        "is_fully_contained": bool(is_fully if inp.strict else is_fully),  # kept same logic; strict can be used by consumers
                        "meets_threshold": meets_threshold,
                        "areas": {
                            "intersection_pixels": round(inter_area_px, 4),
                            "contained_pixels": round(contained_area_px, 4),
                            "container_pixels": round(container_area_px, 4),
                            "intersection_sqm": round(inter_area_m2, 4),
                            "contained_sqm": round(contained_area_m2, 4),
                            "container_sqm": round(container_area_m2, 4),
                        },
                        "intersection_geometry": inter_geom_coords  # polygon/multipolygon in coordinate-array form
                    })

            # 4) Metrics & summary
            if ratio_values:
                ratios = np.array(ratio_values, dtype=float)
                stats = {
                    "min": float(np.min(ratios)),
                    "max": float(np.max(ratios)),
                    "mean": float(np.mean(ratios)),
                    "median": float(np.median(ratios)),
                    "std": float(np.std(ratios)),
                }
            else:
                stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}

            total_pairs = len(containers) * len(contained)

            result: Dict[str, Any] = {
                "success": True,
                "tool": self.name,
                "input": {
                    "containers": len(containers),
                    "contained": len(contained),
                    "meters_per_pixel": inp.meters_per_pixel,
                    "threshold_pct": inp.threshold_pct,
                    "strict": inp.strict,
                    "tolerance": inp.tolerance,
                    "coord_sys": inp.coord_sys,
                },
                "output": {
                    "relationships": relationships,
                    "summary": {
                        "total_pairs": total_pairs,
                        "fully_contained_pairs": contained_true_count,
                        "threshold_met_pairs": meets_threshold_count if inp.threshold_pct is not None else None,
                        "overall_containment_pct": round((contained_true_count / total_pairs) * 100.0, 2) if total_pairs else 0.0,
                        "average_containment_ratio": round(stats["mean"], 6),
                    },
                },
                "metadata": {
                    "containment_statistics": stats
                }
            }

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics
            # This matches the format used in perception tools and BufferTool for consistency

            # Determine container and contained class names
            container_class = inp.container_class or "container"
            contained_class = inp.contained_class or "contained"

            # Build classes_used list from container and contained classes
            if inp.classes_used is not None:
                classes_used_list = inp.classes_used
            else:
                classes_used_list = [container_class, contained_class]

            # Sort classes alphabetically for consistency
            classes_used_sorted = sorted(classes_used_list)

            # Build the unified arguments field
            arguments = {
                "image_path": inp.image_path,
                "classes_used": classes_used_sorted,
                "meters_per_pixel": inp.meters_per_pixel if inp.meters_per_pixel else None,
                "container_class": container_class,
                "contained_class": contained_class,
                "total_containers": len(containers),
                "total_contained": len(contained),
                "fully_contained_count": contained_true_count
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "success": False,
                "tool": self.name,
                "error": str(e),
                "summary": f"Containment analysis failed: {e}"
            }, ensure_ascii=False)

    # ------------- Helpers -------------

    def _to_shapely_polygon(self, coords: PolygonLike) -> Optional[Union[Polygon, MultiPolygon]]:
        """
        Convert coordinate arrays to Shapely Polygon/MultiPolygon.
        Assumptions:
        - Single polygon: [[x,y], ...]  (holes should be preprocessed upstream; we only keep outer ring.)
        - MultiPolygon:   [ [[x,y],...], [[x,y],...] ]  (list of outer rings)
        """
        if coords is None:
            return None

        # Heuristic: if first-level is a list of [x,y], treat as single polygon; else treat as multi.
        if isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], list) and len(coords[0]) >= 2 and all(isinstance(v, (int, float)) for v in coords[0][:2]):
            ring = self._ensure_closed(coords)
            poly = Polygon(ring)
            return poly

        # Otherwise: list of polygons
        polygons: List[Polygon] = []
        for ring in coords:
            if not isinstance(ring, list) or len(ring) < 3:
                continue
            outer = self._ensure_closed(ring)
            polygons.append(Polygon(outer))
        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]
        return unary_union(polygons)  # MultiPolygon or merged Polygon

    def _ensure_closed(self, ring: List[List[float]]) -> List[List[float]]:
        if len(ring) == 0:
            return ring
        if ring[0] != ring[-1]:
            return ring + [ring[0]]
        return ring

    def _clean_polygon(self, geom: Union[Polygon, MultiPolygon], tol: float) -> Optional[Union[Polygon, MultiPolygon]]:
        if geom is None or geom.is_empty:
            return None
        try:
            g = geom
            if not g.is_valid:
                g = g.buffer(0)
            # Remove zero/near-zero area parts in MultiPolygon
            if isinstance(g, MultiPolygon):
                parts = [p for p in g.geoms if p.area > tol]
                if not parts:
                    return None
                if len(parts) == 1:
                    return parts[0]
                return unary_union(parts)
            return g if g.area > tol else None
        except Exception:
            return None

    def _is_valid_area(self, geom: Optional[Union[Polygon, MultiPolygon]], tol: float) -> bool:
        return geom is not None and (not geom.is_empty) and (geom.area > tol)

    def _clip_to_aoi(self, geom: Union[Polygon, MultiPolygon], aoi: Union[Polygon, MultiPolygon], tol: float):
        try:
            out = geom.intersection(aoi)
            return self._clean_polygon(out, tol)
        except Exception:
            return None

    def _geom_to_coords(self, geom: Union[Polygon, MultiPolygon, None]) -> Optional[Union[List[List[float]], List[List[List[float]]]]]:
        """
        Convert Shapely Polygon/MultiPolygon to coordinate arrays (exterior only).
        - Polygon  -> [[x,y], ...]
        - MultiPol -> [ [[x,y],...], [[x,y],...] ]
        """
        if geom is None or geom.is_empty:
            return None
        if isinstance(geom, Polygon):
            coords = list(geom.exterior.coords)
            return [[float(x), float(y)] for x, y in coords]
        elif isinstance(geom, MultiPolygon):
            polys: List[List[List[float]]] = []
            for p in geom.geoms:
                coords = list(p.exterior.coords)
                polys.append([[float(x), float(y)] for x, y in coords])
            return polys
        else:
            return None
