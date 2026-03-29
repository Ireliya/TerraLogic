"""
BufferTool for creating buffer zones around detected geometries.
Creates buffer zones around detected objects for spatial analysis.
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Union, Type, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


class BufferAnalysisInput(BaseModel):
    """
    Input arguments for BufferTool modern execution path.
    Processes all detected geometries of a specified class for buffer analysis.
    """
    perception_results: Dict[str, Any] = Field(
        ..., description="Results from perception tools (detection/segmentation/classification)"
    )
    classes_used: List[str] = Field(
        ..., description="Classes to analyze for buffer creation"
    )
    buffer_distance_meters: float = Field(
        ..., description="Buffer distance in meters", gt=0
    )
    image_path: str = Field(
        ..., description="Path to input satellite image"
    )
    meters_per_pixel: float = Field(
        default=0.3, description="Ground resolution in meters per pixel", gt=0
    )
    query_text: Optional[str] = Field(
        default="", description="Original query for role assignment"
    )


class BufferTool(BaseTool):
    """
    Creates buffer zones around detected geometries for spatial analysis.
    Processes all detected geometries of a specified class for comprehensive buffer analysis.
    """
    name: str = "buffer"
    description: str = (
        "Create buffer zones around detected geometries from perception tools. "
        "Processes all geometries of specified classes for comprehensive spatial analysis. "
        "Returns buffered geometry coordinates, area calculations, and unified buffer zones."
    )
    args_schema: Type[BaseModel] = BufferAnalysisInput

    def _run(self, tool_input: Union[BufferAnalysisInput, Dict[str, Any], str]) -> str:
        """
        Run comprehensive buffer analysis on ALL detected geometries.

        This method processes all detected geometries of the buffer class, creates individual
        buffers for each, and combines them into unified buffer zones for comprehensive
        spatial relationship analysis.

        PYDANTIC VALIDATION:
        ====================
        Input is automatically validated against BufferAnalysisInput schema:
        - perception_results: Required, must be a dict
        - classes_used: Required, must be a non-empty list
        - buffer_distance_meters: Required, must be > 0
        - image_path: Required, must be a string
        - meters_per_pixel: Optional, default 0.3, must be > 0
        - query_text: Optional, default ""

        Args:
            tool_input: Can be one of:
                - BufferAnalysisInput: Pydantic model (already validated)
                - Dict[str, Any]: Dictionary with required fields (will be validated)
                - str: JSON string (will be parsed and validated)

        Returns:
            JSON string containing comprehensive multi-geometry buffer analysis results
        """
        from spatialreason.tools.spatialops import preprocess_all_geometries_for_spatial_relations
        from create_data.generate_gt.semantics.role_assigner import RoleAssigner

        try:
            # Parse and validate input using Pydantic
            if isinstance(tool_input, str):
                # Parse JSON string and validate with Pydantic
                params_dict = json.loads(tool_input)
                params = BufferAnalysisInput(**params_dict)
            elif isinstance(tool_input, dict):
                # Validate dictionary with Pydantic
                params = BufferAnalysisInput(**tool_input)
            else:
                # Already a BufferAnalysisInput instance
                params = tool_input

            # Extract validated parameters directly from Pydantic model
            perception_results = params.perception_results
            classes_used = params.classes_used
            buffer_distance_meters = params.buffer_distance_meters
            image_path = params.image_path
            meters_per_pixel = params.meters_per_pixel
            query_text = params.query_text or ""

            # DEBUG: Log input data for troubleshooting
            print(f"🔍 [DEBUG] Buffer tool input data (Pydantic validated):")
            print(f"   Classes used: {classes_used}")
            print(f"   Perception results keys: {list(perception_results.keys())}")
            for tool_name, result in perception_results.items():
                print(f"   {tool_name} result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                if isinstance(result, dict) and 'result' in result:
                    inner_result = result['result']
                    print(f"   {tool_name} inner result keys: {list(inner_result.keys()) if isinstance(inner_result, dict) else type(inner_result)}")

            # Preprocess to get all geometries
            preprocessing_result = preprocess_all_geometries_for_spatial_relations(
                perception_results, classes_used
            )

            print(f"🔍 [DEBUG] Preprocessing result:")
            print(f"   Success: {preprocessing_result['success']}")
            if preprocessing_result["success"]:
                all_geometries = preprocessing_result["all_geometries"]
                print(f"   Available geometry classes: {list(all_geometries.keys())}")
                for class_name, geometries in all_geometries.items():
                    print(f"   {class_name}: {len(geometries)} geometries")
            else:
                print(f"   Error: {preprocessing_result.get('error', 'Unknown error')}")

            if not preprocessing_result["success"]:
                raise ValueError(f"Geometry preprocessing failed: {preprocessing_result['error']}")

            all_geometries = preprocessing_result["all_geometries"]

            # Role assignment for buffer class determination
            role_assigner = RoleAssigner()
            role_assigner.initialize()

            # Use query_text from Pydantic model (already extracted above)
            if not query_text:
                query_text = f"Are any {classes_used[0]} within {buffer_distance_meters} m of {classes_used[1] if len(classes_used) > 1 else classes_used[0]}"

            role_result = role_assigner.process({
                "query_text": query_text,
                "classes_used": classes_used
            })

            buffer_class = classes_used[0]  # Default fallback
            if role_result.success:
                source_classes = role_result.data.get("source_classes", [])
                if source_classes:
                    buffer_class = source_classes[0]

            if buffer_class not in all_geometries:
                raise ValueError(f"Required buffer source class not detected: {buffer_class}")

            # Execute comprehensive multi-geometry buffer analysis
            return self._execute_comprehensive_buffer_analysis(
                all_geometries, buffer_class, buffer_distance_meters, image_path, meters_per_pixel,
                query_text=query_text, classes_used=classes_used
            )

        except Exception as e:
            error_result = {
                "success": False,
                "tool_name": "buffer",
                "error": str(e),
                "summary": f"Buffer analysis failed: {str(e)}"
            }
            return json.dumps(error_result, indent=2)
    def _execute_comprehensive_buffer_analysis(self,
                                               all_geometries: Dict[str, List[Dict[str, Any]]],
                                               buffer_class: str,
                                               buffer_distance_meters: float,
                                               image_path: str,
                                               meters_per_pixel: float = 0.3,
                                               query_text: str = "",
                                               classes_used: List[str] = None) -> str:
        """
        Execute comprehensive buffer analysis on ALL geometries of the specified buffer class.

        This is the main buffer analysis method that processes all detected geometries of the
        buffer class, creates individual buffers for each, and then combines them into a
        unified buffer zone for comprehensive spatial relationship analysis.

        OUTPUT STRUCTURE DOCUMENTATION:
        ================================
        The output contains the following key fields:

        1. **buffer_union_geometry** (CRITICAL for downstream tools)
           - Unified buffer zone coordinates as [[x1, y1], [x2, y2], ...]
           - Used by: overlap_tool, containment_tool, object_count_aoi
           - This is the PRIMARY output for spatial relationship analysis

        2. **buffered_geometries_info** (Per-buffer statistics)
           - Array of individual buffer metadata: bbox, area, centroid, expansion ratio
           - Provides detailed statistics for each buffered geometry
           - Used for analysis and debugging

        3. **unified_buffer** (Unified buffer summary)
           - Total area, bbox, centroid of the unified buffer zone
           - Provides high-level statistics for the combined buffer

        4. **totals** (Aggregate statistics)
           - Original area, individual buffers total area, unified buffer area
           - Area saved by union operation (overlap elimination)

        Args:
            all_geometries: Dictionary mapping class names to lists of all geometry objects
            buffer_class: Class name of geometries to buffer
            buffer_distance_meters: Buffer distance in meters
            image_path: Path to input image
            meters_per_pixel: Ground resolution

        Returns:
            JSON string containing comprehensive multi-geometry buffer analysis results
        """

        try:
            if buffer_class not in all_geometries:
                raise ValueError(f"Buffer class '{buffer_class}' not found in geometries")

            buffer_geometries = all_geometries[buffer_class]
            if not buffer_geometries:
                raise ValueError(f"No geometries found for buffer class '{buffer_class}'")

            # CRITICAL FIX: Validate coordinate system parameters
            meters_per_pixel, buffer_distance_meters = self._validate_coordinate_system_parameters(
                meters_per_pixel, buffer_distance_meters)

            print(f"🔧 Executing comprehensive multi-geometry buffer analysis for '{buffer_class}'")
            print(f"   Processing {len(buffer_geometries)} geometries")
            print(f"   Buffer distance: {buffer_distance_meters}m")
            print(f"   Ground resolution: {meters_per_pixel}m/pixel")

            # Convert buffer distance to pixels
            buffer_distance_pixels = self._convert_to_pixels(buffer_distance_meters, meters_per_pixel)

            # Process each geometry individually
            individual_buffers = []
            shapely_buffers = []
            total_original_area = 0
            total_buffered_area = 0

            for i, geometry in enumerate(buffer_geometries):
                # Convert geometry to Shapely format
                shapely_geom = self._convert_geometry_to_shapely(geometry)
                if shapely_geom is None:
                    continue

                # Calculate original area
                original_area = shapely_geom.area
                total_original_area += original_area

                # Create buffer
                buffered_geom = shapely_geom.buffer(buffer_distance_pixels)
                shapely_buffers.append(buffered_geom)

                # Calculate buffered area
                buffered_area = buffered_geom.area
                total_buffered_area += buffered_area

                # Extract buffer properties
                bounds = buffered_geom.bounds
                centroid = buffered_geom.centroid

                individual_buffer = {
                    "geometry_id": f"{buffer_class}_{i+1}",
                    "original_geometry_id": geometry.get("object_id", f"{buffer_class}_{i+1}"),
                    "bbox": {
                        "x_min": round(bounds[0], 1),
                        "y_min": round(bounds[1], 1),
                        "x_max": round(bounds[2], 1),
                        "y_max": round(bounds[3], 1),
                        "width": round(bounds[2] - bounds[0], 1),
                        "height": round(bounds[3] - bounds[1], 1)
                    },
                    "centroid": {
                        "x": round(centroid.x, 1),
                        "y": round(centroid.y, 1)
                    },
                    "original_area_pixels": round(original_area, 2),
                    "original_area_sqm": round(self._convert_to_meters_squared(original_area, meters_per_pixel), 2),
                    "buffered_area_pixels": round(buffered_area, 2),
                    "buffered_area_sqm": round(self._convert_to_meters_squared(buffered_area, meters_per_pixel), 2),
                    "buffer_expansion_ratio": round(buffered_area / original_area if original_area > 0 else 0, 2)
                }
                individual_buffers.append(individual_buffer)

            # Create unified buffer from all individual buffers
            if shapely_buffers:
                unified_buffer = unary_union(shapely_buffers)

                # IMAGE BOUNDARY CLIPPING STRATEGY:
                # ===================================
                # CRITICAL FIX: DISABLED image boundary clipping
                #
                # REASON: Clipping the unified buffer to image bounds using Shapely's intersection()
                # was corrupting the buffer geometry, causing it to become a convex hull covering
                # almost the entire image instead of the actual buffer zone.
                #
                # SOLUTION: Skip clipping and let downstream tools handle geometry validation.
                # - ObjectCountAOITool can handle geometries that extend beyond image bounds
                # - OverlapTool can handle geometries that extend beyond image bounds
                # - ContainmentTool can handle geometries that extend beyond image bounds
                # - Downstream tools perform their own geometry cleaning and validation
                #
                # This preserves the integrity of the buffer geometry and ensures correct
                # spatial analysis results.
                print(f"   ✅ Unified buffer geometry preserved (no clipping)")

                unified_area_pixels = unified_buffer.area
                unified_area_sqm = self._convert_to_meters_squared(unified_area_pixels, meters_per_pixel)

                # Calculate unified buffer properties
                unified_bounds = unified_buffer.bounds
                unified_centroid = unified_buffer.centroid

                unified_buffer_info = {
                    "bbox": {
                        "x_min": round(unified_bounds[0], 1),
                        "y_min": round(unified_bounds[1], 1),
                        "x_max": round(unified_bounds[2], 1),
                        "y_max": round(unified_bounds[3], 1),
                        "width": round(unified_bounds[2] - unified_bounds[0], 1),
                        "height": round(unified_bounds[3] - unified_bounds[1], 1)
                    },
                    "centroid": {
                        "x": round(unified_centroid.x, 1),
                        "y": round(unified_centroid.y, 1)
                    },
                    "area_pixels": round(unified_area_pixels, 2),
                    "area_sqm": round(unified_area_sqm, 2),
                    "geometry_count": len(individual_buffers),
                    "overlap_eliminated": unified_area_pixels < total_buffered_area
                }
            else:
                unified_buffer_info = None

            # Convert areas to square meters
            total_original_area_sqm = self._convert_to_meters_squared(total_original_area, meters_per_pixel)
            total_buffered_area_sqm = self._convert_to_meters_squared(total_buffered_area, meters_per_pixel)

            # CRITICAL FIX: Create buffer union geometry for object_count_aoi connectivity
            buffer_union_geometry = None
            if shapely_buffers:
                try:
                    # Convert unified buffer to coordinate format for AOI connectivity
                    if hasattr(unified_buffer, 'exterior'):
                        # Single polygon
                        coords = list(unified_buffer.exterior.coords)
                        buffer_union_geometry = [[float(x), float(y)] for x, y in coords]
                    elif hasattr(unified_buffer, 'geoms'):
                        # MultiPolygon - take the largest polygon
                        largest_poly = max(unified_buffer.geoms, key=lambda p: p.area)
                        coords = list(largest_poly.exterior.coords)
                        buffer_union_geometry = [[float(x), float(y)] for x, y in coords]

                    print(f"   🔗 Created buffer union geometry for AOI connectivity (area: {unified_area_sqm:.2f} sqm)")
                except Exception as union_error:
                    print(f"   ⚠️ Failed to create buffer union geometry: {union_error}")

            # ========== COMPUTE OBJECTS WITHIN BUFFER ==========
            # This is the KEY improvement for proper ReAct termination
            # The LLM can use these counts directly to answer the query
            objects_within_buffer = {}
            total_objects_in_buffer = 0

            if shapely_buffers and unified_buffer is not None:
                # Check each non-buffer class for objects within the buffer
                for other_class, other_geometries in all_geometries.items():
                    if other_class == buffer_class:
                        continue  # Skip the buffer source class

                    count_in_buffer = 0
                    for other_geom in other_geometries:
                        other_shapely = self._convert_geometry_to_shapely(other_geom)
                        if other_shapely is not None:
                            # Check if centroid is within buffer OR if geometries intersect
                            try:
                                if unified_buffer.contains(other_shapely.centroid) or unified_buffer.intersects(other_shapely):
                                    count_in_buffer += 1
                            except Exception:
                                pass

                    if count_in_buffer > 0:
                        objects_within_buffer[other_class] = count_in_buffer
                        total_objects_in_buffer += count_in_buffer

                print(f"   🎯 Objects within buffer: {objects_within_buffer} (total: {total_objects_in_buffer})")
            # ========== END OBJECTS WITHIN BUFFER ==========

            # Prepare comprehensive results
            result = {
                "success": True,
                "tool_name": "buffer",
                "processing_mode": "multi_geometry",
                "buffer_class": buffer_class,
                "buffer_distance_meters": buffer_distance_meters,
                "buffer_distance_pixels": buffer_distance_pixels,
                "meters_per_pixel": meters_per_pixel,
                "geometry_count": len(individual_buffers),
                "individual_buffers": individual_buffers,
                "unified_buffer": unified_buffer_info,
                "buffer_union_geometry": buffer_union_geometry,  # CRITICAL: For object_count_aoi connectivity
                "objects_within_buffer": objects_within_buffer,  # NEW: Direct counts for ReAct termination
                "total_objects_in_buffer": total_objects_in_buffer,  # NEW: Total count for easy access
                "totals": {
                    "original_area_sqm": round(total_original_area_sqm, 2),
                    "individual_buffers_total_area_sqm": round(total_buffered_area_sqm, 2),
                    "unified_buffer_area_sqm": round(unified_area_sqm, 2) if unified_buffer_info else 0,
                    "area_saved_by_union_sqm": round(total_buffered_area_sqm - unified_area_sqm, 2) if unified_buffer_info else 0
                },
                "summary": f"Created {buffer_distance_meters}m buffer around {len(individual_buffers)} {buffer_class} objects. {total_objects_in_buffer} objects from other classes found within buffer: {objects_within_buffer}"
            }

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics
            # This matches the format used in perception tools for consistency

            # Ensure classes_used is properly initialized
            if classes_used is None:
                classes_used = [buffer_class]

            # Sort classes alphabetically for consistency
            classes_used_sorted = sorted(classes_used)

            # Build the unified arguments field
            arguments = {
                "image_path": image_path,
                "classes_used": classes_used_sorted,
                "meters_per_pixel": meters_per_pixel if meters_per_pixel else None,
                "buffer_distance_meters": buffer_distance_meters,
                "buffer_class": buffer_class,
                "total_geometries_buffered": len(individual_buffers),
                "unified_buffer_area_sqm": round(unified_area_sqm, 2) if unified_buffer_info else 0.0
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            return json.dumps(result, indent=2)

        except Exception as e:
            error_result = {
                "success": False,
                "tool_name": "buffer",
                "processing_mode": "multi_geometry",
                "error": str(e),
                "summary": f"Multi-geometry buffer analysis failed: {str(e)}"
            }
            return json.dumps(error_result, indent=2)

    def _convert_geometry_to_shapely(self, geometry: Dict[str, Any]):
        """Convert a geometry dictionary to a Shapely geometry object."""
        try:
            # Try polygon coordinates first (from segmentation)
            if "polygon" in geometry and geometry["polygon"]:
                polygon_data = geometry["polygon"]
                if isinstance(polygon_data, str):
                    polygon_data = json.loads(polygon_data)

                if isinstance(polygon_data, list) and len(polygon_data) >= 3:
                    # Convert coordinate pairs to Shapely polygon
                    coords = [(float(x), float(y)) for x, y in polygon_data]
                    return Polygon(coords)

            # Try bounding box
            elif "bbox" in geometry:
                bbox = geometry["bbox"]
                return box(bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"])

            # Try centroid as point (will need small buffer to create area)
            elif "centroid" in geometry:
                centroid = geometry["centroid"]
                if isinstance(centroid, dict) and "x" in centroid and "y" in centroid:
                    # Create small square around centroid
                    x, y = centroid["x"], centroid["y"]
                    size = 1.0  # 1 pixel square
                    return box(x - size/2, y - size/2, x + size/2, y + size/2)

            return None

        except Exception:
            return None

    def _convert_to_pixels(self, distance_meters: float, meters_per_pixel: float) -> float:
        """
        Convert distance from meters to pixels with validation.

        STANDARDIZED CONVERSION:
        - All distance conversions use this method for consistency
        - Formula: pixel_distance = distance_meters / meters_per_pixel
        - Validates meters_per_pixel and applies default if invalid

        Args:
            distance_meters: Distance in meters
            meters_per_pixel: Ground Sample Distance (GSD)

        Returns:
            Distance in pixels
        """
        if meters_per_pixel <= 0:
            print(f"⚠️  Invalid meters_per_pixel value: {meters_per_pixel}, using default 0.3")
            meters_per_pixel = 0.3

        pixel_distance = distance_meters / meters_per_pixel
        print(f"🔧 Converting {distance_meters}m to {pixel_distance:.1f} pixels (GSD: {meters_per_pixel}m/pixel)")
        return pixel_distance

    def _convert_to_meters_squared(self, area_pixels: float, meters_per_pixel: float) -> float:
        """
        Convert area from pixels to square meters with validation.

        STANDARDIZED CONVERSION:
        - All area conversions use this method for consistency
        - Formula: area_meters = area_pixels * (meters_per_pixel ** 2)
        - Validates meters_per_pixel and applies default if invalid

        Args:
            area_pixels: Area in square pixels
            meters_per_pixel: Ground Sample Distance (GSD)

        Returns:
            Area in square meters
        """
        if meters_per_pixel <= 0:
            print(f"⚠️  Invalid meters_per_pixel value: {meters_per_pixel}, using default 0.3")
            meters_per_pixel = 0.3

        area_meters = area_pixels * (meters_per_pixel ** 2)
        print(f"🔧 Converting {area_pixels:.1f} pixel² to {area_meters:.2f} m² (GSD: {meters_per_pixel}m/pixel)")
        return area_meters

    def _validate_coordinate_system_parameters(self, meters_per_pixel: float, buffer_distance_meters: float) -> tuple:
        """
        Validate and correct coordinate system parameters.

        PARAMETER VALIDATION STRATEGY:
        ==============================
        - Ensures meters_per_pixel (GSD) is within reasonable range for satellite imagery
        - Ensures buffer_distance_meters is positive and reasonable
        - Applies sensible defaults for invalid values
        - Warns about unusual but potentially valid values

        Typical satellite imagery GSD ranges:
        - High resolution: 0.1-1 m/pixel (e.g., Maxar, Planet)
        - Medium resolution: 1-10 m/pixel (e.g., Sentinel-2)
        - Low resolution: 10-100 m/pixel (e.g., Landsat, MODIS)

        Args:
            meters_per_pixel: Ground Sample Distance (GSD)
            buffer_distance_meters: Buffer distance in meters

        Returns:
            Tuple of (validated_meters_per_pixel, validated_buffer_distance)
        """
        # Validate meters_per_pixel (typical satellite imagery: 0.1-10 m/pixel)
        if meters_per_pixel <= 0 or meters_per_pixel > 100:
            print(f"⚠️  Invalid meters_per_pixel: {meters_per_pixel}, using default 0.3 m/pixel")
            meters_per_pixel = 0.3
        elif meters_per_pixel > 10:
            print(f"⚠️  Unusually large meters_per_pixel: {meters_per_pixel} m/pixel (typical range: 0.1-10)")

        # Validate buffer distance (should be reasonable for satellite imagery analysis)
        if buffer_distance_meters <= 0:
            print(f"⚠️  Invalid buffer distance: {buffer_distance_meters}m, using default 30m")
            buffer_distance_meters = 30.0
        elif buffer_distance_meters > 10000:  # 10km seems excessive for most analyses
            print(f"⚠️  Very large buffer distance: {buffer_distance_meters}m, this may cause performance issues")

        print(f"🔧 Validated parameters: GSD={meters_per_pixel}m/pixel, buffer={buffer_distance_meters}m")
        return meters_per_pixel, buffer_distance_meters
