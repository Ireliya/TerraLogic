"""
ObjectCountInAOITool (pure geometric)
Counts polygonal objects that fall within polygonal AOIs.

- I/O strictly polygon-like coordinates (simple, with holes, or MultiPolygon)
- Robust cleaning & tiny-area filtering
- Counting rules: intersects / contains_centroid / covered_by
- Per-AOI results + global summary + optional geometry previews
"""

from typing import List, Dict, Any, Union, Optional, Type
from pydantic import BaseModel, Field, validator
from langchain_core.tools import BaseTool
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np
import json
import cv2

PolygonLike = Union[
    List[List[float]],           # simple polygon: [[x,y], [x,y], ...]
    List[List[List[float]]]      # polygon-with-holes: [outer, hole1, ...]  OR
                                 # multipolygon: [[poly1], [poly2], ...]
]

# ------------------------------
#            Schema
# ------------------------------

class ObjectCountAOIInput(BaseModel):
    """
    Pure geometric input:
      - objects: list of polygon-like (simple / holes / multipolygon)
      - aois:    list of polygon-like

    Note: objects and aois are now optional to support benchmark format where only
    object_class and aoi_class are provided. In such cases, the tool will return
    an error indicating that geometries are required.
    """
    objects: Optional[List[PolygonLike]] = Field(default=None, description="List of object polygon-like geometries (optional for benchmark format)")
    aois: Optional[List[PolygonLike]] = Field(default=None, description="List of AOI polygon-like geometries (optional for benchmark format)")

    meters_per_pixel: Optional[float] = Field(default=0.3, gt=0, description="Ground resolution in meters per pixel (None for IR images without GSD)")
    tolerance: float = Field(default=1e-6, gt=0, description="Area tolerance (pixel^2) for tiny parts filtering")

    # Optional filters (post-cleaning, pre-count)
    min_object_area_sqm: Optional[float] = Field(default=None, ge=0, description="Drop objects smaller than this (m^2)")
    min_aoi_area_sqm: Optional[float] = Field(default=None, ge=0, description="Drop AOIs smaller than this (m^2)")

    # Counting rule
    counting_rule: str = Field(
        default="intersects",
        description="Counting rule: 'intersects' | 'contains_centroid' | 'covered_by'"
    )

    # Optional IDs (for stable references)
    object_ids: Optional[List[str]] = Field(default=None, description="Optional IDs for objects (same length as objects)")
    aoi_ids: Optional[List[str]] = Field(default=None, description="Optional IDs for AOIs (same length as aois)")

    # Optional previews to limit payload size
    include_cleaned_geometries: bool = Field(default=False, description="Include cleaned geometry previews in output")
    cleaned_geometries_limit: Optional[int] = Field(default=None, description="Max number of previews across objects+aois")
    include_holes_in_geometry: bool = Field(default=True, description="If returning geometry previews, include holes")

    # Optional parameters for unified arguments field (from upstream tools)
    image_path: Optional[str] = Field(default=None, description="Path to input satellite image (for traceability)")
    query_text: Optional[str] = Field(default="", description="Original natural-language query")
    classes_used: Optional[List[str]] = Field(default=None, description="The classes involved in the object counting operation (e.g., object class and AOI class)")

    # Benchmark format parameters (used when geometries are not directly provided)
    object_class: Optional[str] = Field(default=None, description="Class name of objects to count (benchmark format)")
    aoi_class: Optional[str] = Field(default=None, description="Class name of AOI region (benchmark format)")

    @validator("meters_per_pixel", pre=True, always=True)
    def _check_meters_per_pixel(cls, v):
        # Allow None for IR images without GSD
        if v is None:
            return None
        # For numeric values, ensure they are positive
        if isinstance(v, (int, float)) and v > 0:
            return v
        raise ValueError("meters_per_pixel must be None (for IR images) or a positive number")

    @validator("counting_rule")
    def _check_rule(cls, v: str):
        if v not in ("intersects", "contains_centroid", "covered_by"):
            raise ValueError("counting_rule must be 'intersects', 'contains_centroid', or 'covered_by'")
        return v

    @validator("object_ids", always=True)
    def _check_object_ids(cls, v, values):
        objs = values.get("objects") or []
        if v is not None and len(v) != len(objs):
            raise ValueError("object_ids length must match objects length")
        return v

    @validator("aoi_ids", always=True)
    def _check_aoi_ids(cls, v, values):
        aois = values.get("aois") or []
        if v is not None and len(v) != len(aois):
            raise ValueError("aoi_ids length must match aois length")
        return v


# ------------------------------
#           The Tool
# ------------------------------

class ObjectCountInAOITool(BaseTool):
    """
    Count polygonal objects inside polygonal AOIs under a chosen rule.
    Output: per-AOI counts, areas, densities (per ha), global summary, optional previews.
    """
    name: str = "object_count_aoi"
    description: str = (
        "Count polygonal objects within polygonal AOIs. "
        "Inputs are polygon coordinates (simple/holes/multipolygon). "
        "Supports counting rules: intersects / contains_centroid / covered_by. "
        "Returns per-AOI results, global summary, and optional geometry previews."
    )
    args_schema: Type[BaseModel] = ObjectCountAOIInput

    # ---------- Public JSON API ----------

    def _run(self, tool_input: Union[Dict[str, Any], str]) -> str:
        try:
            params = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            inp = ObjectCountAOIInput(**params)

            warnings: List[str] = []

            # DEBUG: Log input data
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"🔍 ObjectCountAOITool._run() DEBUG:")
            logger.info(f"   Input objects count: {len(inp.objects) if inp.objects else 0}")
            logger.info(f"   Input aois count: {len(inp.aois) if inp.aois else 0}")
            logger.info(f"   object_class: {inp.object_class}, aoi_class: {inp.aoi_class}")
            logger.info(f"   First object: {str(inp.objects[0])[:100] if inp.objects else 'None'}")
            logger.info(f"   First AOI: {str(inp.aois[0])[:100] if inp.aois else 'None'}")

            # CRITICAL FIX: Check if geometries are provided
            # If objects or aois are None/empty, return an error indicating that geometries are required
            if not inp.objects or not inp.aois:
                error_msg = (
                    f"Missing required geometries for object_count_aoi tool. "
                    f"Received: objects={len(inp.objects) if inp.objects else 0}, "
                    f"aois={len(inp.aois) if inp.aois else 0}. "
                    f"This tool requires actual polygon geometries to be provided. "
                    f"If you provided object_class='{inp.object_class}' and aoi_class='{inp.aoi_class}', "
                    f"note that these are metadata parameters only - the tool needs actual geometries from perception tools. "
                    f"Ensure that detection/segmentation results are properly passed to this tool."
                )
                logger.warning(f"🔍 {error_msg}")
                return json.dumps(self._fail(error_msg, warnings), indent=2)

            # Build + clean
            min_obj_area_px = self._sqm_to_pixels(inp.min_object_area_sqm, inp.meters_per_pixel) if inp.min_object_area_sqm is not None else None
            min_aoi_area_px = self._sqm_to_pixels(inp.min_aoi_area_sqm, inp.meters_per_pixel) if inp.min_aoi_area_sqm is not None else None

            objects = self._build_and_clean_batch(
                inp.objects, warnings, label="OBJ", tol=inp.tolerance, min_area_pixels=min_obj_area_px
            )
            aois = self._build_and_clean_batch(
                inp.aois, warnings, label="AOI", tol=inp.tolerance, min_area_pixels=min_aoi_area_px
            )

            # DEBUG: Log after cleaning
            logger.info(f"   After cleaning - objects: {len(objects)}, aois: {len(aois)}")
            logger.info(f"   Warnings: {warnings}")

            if not objects or not aois:
                return json.dumps(self._fail("All objects or AOIs are invalid/empty after cleaning.", warnings), indent=2)

            # Assign IDs
            object_ids = inp.object_ids or [f"OBJ_{i+1}" for i in range(len(objects))]
            aoi_ids = inp.aoi_ids or [f"AOI_{i+1}" for i in range(len(aois))]

            # Per-AOI counting
            aoi_results = []
            all_contained_ids: List[str] = []

            for ai, aoi in enumerate(aois):
                aoi_id = aoi_ids[ai]
                contained: List[str] = []

                # DEBUG: Log AOI details
                logger.info(f"   Processing AOI {ai}: area={aoi.area:.2f} px, bounds={aoi.bounds}")

                for oi, obj in enumerate(objects):
                    oid = object_ids[oi]
                    matches = self._matches_rule(obj, aoi, rule=inp.counting_rule)
                    if matches:
                        contained.append(oid)
                        logger.info(f"     Object {oi} ({oid}): MATCHES (area={obj.area:.2f} px, bounds={obj.bounds})")
                    else:
                        logger.debug(f"     Object {oi} ({oid}): no match (area={obj.area:.2f} px, bounds={obj.bounds})")

                logger.info(f"   AOI {ai} result: {len(contained)} objects matched")

                area_px = float(aoi.area)

                # For IR images (meters_per_pixel=None), we can't calculate area in m²
                # Only calculate area_m2 if meters_per_pixel is provided
                if inp.meters_per_pixel is not None:
                    area_m2 = area_px * (inp.meters_per_pixel ** 2)
                else:
                    area_m2 = None

                # Density (objects / ha), reliability threshold: 100 m²
                min_area_for_density = 100.0
                density_per_ha = None
                density_reliable = False
                density_note = None

                if area_m2 is not None:
                    if area_m2 >= min_area_for_density and area_m2 > 0:
                        density_per_ha = (len(contained) / area_m2) * 10000.0
                        density_reliable = True
                    else:
                        density_note = f"AOI area {area_m2:.1f} m² < 100 m², density may be unstable"
                else:
                    # For IR images without GSD, we can't calculate density
                    density_note = "GSD not available (IR image), density calculation skipped"

                aoi_results.append({
                    "aoi_id": aoi_id,
                    "object_ids": contained,
                    "object_count": len(contained),
                    "aoi_area_m2": round(area_m2, 2) if area_m2 is not None else None,
                    "object_density_per_hectare": round(density_per_ha, 4) if density_per_ha is not None else None,
                    "density_reliable": density_reliable,
                    "density_note": density_note,
                    "min_area_threshold_m2": 100.0
                })
                all_contained_ids.extend(contained)

            # Global summary
            unique_in_any_aoi = sorted(set(all_contained_ids))

            # Calculate total AOI area (handle None values for IR images)
            total_aoi_area_m2 = sum(r["aoi_area_m2"] for r in aoi_results if r["aoi_area_m2"] is not None)
            if total_aoi_area_m2 == 0 and any(r["aoi_area_m2"] is None for r in aoi_results):
                # If all areas are None (IR image), set total to None
                total_aoi_area_m2 = None

            overall_density = None
            overall_density_reliable = False
            if total_aoi_area_m2 is not None and total_aoi_area_m2 >= 100.0 and total_aoi_area_m2 > 0:
                overall_density = round(len(unique_in_any_aoi) / total_aoi_area_m2 * 10000.0, 4)
                overall_density_reliable = True

            # Optional previews (limit total count)
            previews = {}
            if inp.include_cleaned_geometries:
                previews = self._geometry_previews(
                    objects, aois, object_ids, aoi_ids,
                    limit=inp.cleaned_geometries_limit,
                    include_holes=inp.include_holes_in_geometry
                )

            result = {
                "success": True,
                "tool": "object_count_aoi",
                "input": {
                    "meters_per_pixel": inp.meters_per_pixel,
                    "tolerance": inp.tolerance,
                    "min_object_area_sqm": inp.min_object_area_sqm,
                    "min_aoi_area_sqm": inp.min_aoi_area_sqm,
                    "counting_rule": inp.counting_rule,
                    "len_objects": len(objects),
                    "len_aois": len(aois),
                    "include_cleaned_geometries": inp.include_cleaned_geometries,
                    "cleaned_geometries_limit": inp.cleaned_geometries_limit,
                    "include_holes_in_geometry": inp.include_holes_in_geometry
                },
                "output": {
                    "aoi_results": aoi_results,
                    "summary": {
                        "total_objects": len(objects),
                        "objects_in_any_aoi": len(unique_in_any_aoi),
                        "objects_outside_aois": len(objects) - len(unique_in_any_aoi),
                        "total_aoi_regions": len(aois),
                        "total_aoi_area_m2": round(total_aoi_area_m2, 2) if total_aoi_area_m2 is not None else None,
                        "overall_density_per_hectare": overall_density,
                        "overall_density_reliable": overall_density_reliable,
                        "reliable_aoi_count": sum(1 for r in aoi_results if r["density_reliable"]),
                        "total_aoi_count": len(aoi_results),
                        "min_area_threshold_m2": 100.0
                    },
                    **({"geometry_previews": previews} if previews else {})
                },
                "metadata": {
                    "validation_warnings": warnings,
                    "processing_summary": (
                        f"Counted objects under rule '{inp.counting_rule}': "
                        f"{len(unique_in_any_aoi)}/{len(objects)} objects in {len(aois)} AOIs."
                    )
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

            # Build the unified arguments field with ONLY 4 fields
            arguments = {
                "image_path": inp.image_path,
                "classes_used": classes_used_sorted,
                "total_objects": len(objects),
                "total_aois": len(aois)
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps(self._fail(f"Object count in AOI failed: {e}"), indent=2)

    # ---------- Geometry building & cleaning ----------

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
                warnings.append(f"{label}[{i}] dropped (< min area).")
                continue
            out.append(g)
        return out

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

    # ---------- Structure checks & ring ops ----------

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

    # ---------- Cleaning ----------

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

    # ---------- Rules ----------

    def _matches_rule(self, obj: Union[Polygon, MultiPolygon], aoi: Union[Polygon, MultiPolygon], rule: str) -> bool:
        try:
            if rule == "intersects":
                inter = obj.intersection(aoi)
                return (not inter.is_empty) and getattr(inter, "area", 0.0) > 0.0
            elif rule == "contains_centroid":
                return aoi.contains(obj.centroid)
            elif rule == "covered_by":
                # covers/within semantics (allow boundary-touch to count as inside)
                return aoi.covers(obj)
            else:
                return False
        except Exception:
            return False

    # ---------- Previews & Utils ----------

    def _geometry_previews(
        self,
        objects: List[Union[Polygon, MultiPolygon]],
        aois: List[Union[Polygon, MultiPolygon]],
        object_ids: List[str],
        aoi_ids: List[str],
        limit: Optional[int],
        include_holes: bool
    ) -> Dict[str, Any]:
        cap = limit if (limit is not None and limit >= 0) else None
        out_obj, out_aoi = [], []
        count = 0

        for i, g in enumerate(objects):
            if cap is not None and count >= cap:
                break
            out_obj.append({
                "id": object_ids[i],
                "geometry": self._geom_to_coords(g, include_holes=include_holes)
            })
            count += 1

        for j, g in enumerate(aois):
            if cap is not None and count >= cap:
                break
            out_aoi.append({
                "id": aoi_ids[j],
                "geometry": self._geom_to_coords(g, include_holes=include_holes)
            })
            count += 1

        previews = {}
        if out_obj:
            previews["objects"] = out_obj
        if out_aoi:
            previews["aois"] = out_aoi
        return previews

    def _geom_to_coords(self, geom: Union[Polygon, MultiPolygon], include_holes: bool = True) -> PolygonLike:
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

    def _create_aoi_from_query(self, query_text: str, image_path: str) -> Optional[List[List[int]]]:
        """
        Create AOI geometry based on spatial keywords in query text.

        Detects keywords like "right half", "left half", "upper half", "lower half", "top half", "bottom half"
        and generates corresponding polygon coordinates.

        Args:
            query_text: Natural language query text
            image_path: Path to the image file (to get dimensions)

        Returns:
            Polygon coordinates in format [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x1, y1]] or None if no keywords found
        """
        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                return None

            height, width = image.shape[:2]
            query_lower = query_text.lower()

            # Define AOI based on spatial descriptions in the query
            if "lower half" in query_lower or "bottom half" in query_lower:
                # Lower half of the scene
                aoi_coords = [
                    [0, height // 2],           # Top-left of lower half
                    [width, height // 2],       # Top-right of lower half
                    [width, height],            # Bottom-right
                    [0, height],                # Bottom-left
                    [0, height // 2]            # Close polygon
                ]
                return aoi_coords

            elif "upper half" in query_lower or "top half" in query_lower:
                # Upper half of the scene
                aoi_coords = [
                    [0, 0],                     # Top-left
                    [width, 0],                 # Top-right
                    [width, height // 2],       # Bottom-right of upper half
                    [0, height // 2],           # Bottom-left of upper half
                    [0, 0]                      # Close polygon
                ]
                return aoi_coords

            elif "left half" in query_lower:
                # Left half of the scene
                aoi_coords = [
                    [0, 0],                     # Top-left
                    [width // 2, 0],            # Top-right of left half
                    [width // 2, height],       # Bottom-right of left half
                    [0, height],                # Bottom-left
                    [0, 0]                      # Close polygon
                ]
                return aoi_coords

            elif "right half" in query_lower:
                # Right half of the scene
                aoi_coords = [
                    [width // 2, 0],            # Top-left of right half
                    [width, 0],                 # Top-right
                    [width, height],            # Bottom-right
                    [width // 2, height],       # Bottom-left of right half
                    [width // 2, 0]             # Close polygon
                ]
                return aoi_coords

            elif "center" in query_lower or "middle" in query_lower:
                # Central area (middle 50% of the image)
                margin_x = width // 4
                margin_y = height // 4
                aoi_coords = [
                    [margin_x, margin_y],                    # Top-left of center
                    [width - margin_x, margin_y],            # Top-right of center
                    [width - margin_x, height - margin_y],   # Bottom-right of center
                    [margin_x, height - margin_y],           # Bottom-left of center
                    [margin_x, margin_y]                     # Close polygon
                ]
                return aoi_coords

            # No spatial keywords found
            return None

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to create AOI from query: {e}")
            return None

    def _sqm_to_pixels(self, area_sqm: Optional[float], meters_per_pixel: Optional[float]) -> Optional[float]:
        if area_sqm is None or meters_per_pixel is None:
            return None
        return area_sqm / (meters_per_pixel ** 2)

    def _fail(self, msg: str, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "success": False,
            "tool": "object_count_aoi",
            "error": msg,
            "metadata": {
                "validation_warnings": warnings or []
            }
        }
