"""
Parameter mapping module for spatialreason.plan

This module contains the ParameterMapper class which handles mapping and
normalization of tool parameters.

Classes:
    ParameterMapper: Maps and normalizes parameters for tools
"""

import logging
import os
import re
from typing import Dict, Any, Optional

# Setup module logger
logger = logging.getLogger(__name__)


class ParameterMapper:
    """
    Maps and normalizes tool parameters for execution.
    
    This class handles:
    - Fixing placeholder parameters
    - Normalizing parameter values
    - Ensuring required parameters are present
    - Converting between different parameter formats
    """
    
    def __init__(self, parameter_extractor=None):
        """
        Initialize the parameter mapper.
        
        Args:
            parameter_extractor: Optional ParameterExtractor instance for context
        """
        self.parameter_extractor = parameter_extractor
    
    def fix_placeholder_parameters(self, action_input: Dict[str, Any], 
                                   tool_name: str, parameter_extractor=None) -> Dict[str, Any]:
        """
        Fix placeholder parameter values by extracting real values from context.
        
        Args:
            action_input: Dictionary of tool arguments that may contain placeholders
            tool_name: Name of the tool being called
            parameter_extractor: Optional ParameterExtractor for context
        
        Returns:
            Dictionary with corrected parameter values
        """
        extractor = parameter_extractor or self.parameter_extractor
        
        # Create a copy to avoid modifying the original
        fixed_input = action_input.copy()
        
        # CRITICAL FIX: Ensure required parameters are present for perception tools
        if tool_name in ["segmentation", "detection", "classification"]:
            # Extract image path from context
            if extractor:
                image_path = extractor.extract_image_path_from_context()
            else:
                image_path = None

            # Add image_path if missing or placeholder
            if "image_path" not in fixed_input or not fixed_input["image_path"]:
                if image_path:
                    logger.info(f"Adding missing image_path parameter: {image_path}")
                    fixed_input["image_path"] = image_path
                else:
                    logger.error(f"❌ CRITICAL: No image_path available for {tool_name} tool")
                    raise ValueError(f"Missing required image_path parameter for {tool_name} tool")

        elif tool_name == "change_detection":
            # Change detection requires two image paths (T1 and T2)
            # If not provided, derive them from the single image_path

            import os
            from pathlib import Path

            # CRITICAL FIX: Validate and correct paths even if they're already present
            # The LLM sometimes generates incorrect paths with _T1/_T2 suffixes instead of /t1/ and /t2/ directories
            needs_derivation = False

            if ("image_path_t1" not in fixed_input or not fixed_input["image_path_t1"]) or \
               ("image_path_t2" not in fixed_input or not fixed_input["image_path_t2"]):
                needs_derivation = True
                logger.info(f"🔍 Change detection paths missing, will derive them")
            else:
                # Validate existing paths - check for common LLM mistakes
                t1_path = fixed_input.get("image_path_t1", "")
                t2_path = fixed_input.get("image_path_t2", "")

                # Check if paths have incorrect _T1/_T2 suffix format (e.g., "dataset/images/13_cd_T1.png")
                # instead of correct directory format (e.g., "dataset/hrscd_images/t1/13_cd.png")
                if "_T1" in t1_path or "_T2" in t2_path or "_t1" in t1_path or "_t2" in t2_path:
                    logger.warning(f"⚠️ Detected incorrect path format with _T1/_T2 suffixes:")
                    logger.warning(f"   T1: {t1_path}")
                    logger.warning(f"   T2: {t2_path}")
                    needs_derivation = True
                # Check if paths don't exist
                elif not os.path.exists(t1_path) or not os.path.exists(t2_path):
                    logger.warning(f"⚠️ Provided paths do not exist:")
                    logger.warning(f"   T1: {t1_path} (exists: {os.path.exists(t1_path)})")
                    logger.warning(f"   T2: {t2_path} (exists: {os.path.exists(t2_path)})")
                    needs_derivation = True
                else:
                    logger.info(f"✅ Change detection parameters already present and valid: T1={t1_path}, T2={t2_path}")

            if needs_derivation:
                # Try to extract base image path from context
                if extractor:
                    base_image_path = extractor.extract_image_path_from_context()
                else:
                    base_image_path = None

                # Also check if there's an image_path in the input
                if not base_image_path and "image_path" in fixed_input:
                    base_image_path = fixed_input["image_path"]

                # Try to extract from existing incorrect paths
                if not base_image_path and "image_path_t1" in fixed_input:
                    # Extract base filename from incorrect path (e.g., "dataset/images/13_cd_T1.png" -> "13_cd.png")
                    t1_path = fixed_input["image_path_t1"]
                    filename = os.path.basename(t1_path)
                    # Remove _T1, _T2, _t1, _t2 suffixes
                    filename = filename.replace("_T1", "").replace("_T2", "").replace("_t1", "").replace("_t2", "")
                    base_image_path = f"dataset/hrscd_images/t1/{filename}"
                    logger.info(f"🔍 Extracted base path from incorrect T1 path: {base_image_path}")

                if base_image_path:
                    # Derive T1 and T2 paths from base path
                    # Expected format: "dataset/images/13_cd.png" -> "dataset/hrscd_images/t1/13_cd.png" and "dataset/hrscd_images/t2/13_cd.png"
                    # OR: "dataset/hrscd_images/t1/13_cd.png" (already in T1 format)

                    # Extract filename (e.g., "13_cd.png")
                    filename = os.path.basename(base_image_path)

                    # Check if the path already contains t1 or t2 subdirectory
                    if "/t1/" in base_image_path:
                        # Already in T1 format, derive T2 by replacing t1 with t2
                        image_path_t1 = base_image_path
                        image_path_t2 = base_image_path.replace("/t1/", "/t2/")
                    elif "/t2/" in base_image_path:
                        # Already in T2 format, derive T1 by replacing t2 with t1
                        image_path_t1 = base_image_path.replace("/t2/", "/t1/")
                        image_path_t2 = base_image_path
                    else:
                        # Generic format (e.g., "dataset/images/13_cd.png")
                        # Construct T1 and T2 paths
                        image_path_t1 = f"dataset/hrscd_images/t1/{filename}"
                        image_path_t2 = f"dataset/hrscd_images/t2/{filename}"

                    # Set the derived paths
                    fixed_input["image_path_t1"] = image_path_t1
                    fixed_input["image_path_t2"] = image_path_t2

                    logger.info(f"✅ Derived bi-temporal paths from {base_image_path}:")
                    logger.info(f"   T1: {image_path_t1}")
                    logger.info(f"   T2: {image_path_t2}")
                else:
                    logger.error(f"❌ CRITICAL: No image path available to derive T1/T2 paths for change_detection tool")
                    raise ValueError(f"Missing required image_path_t1 and image_path_t2 parameters for change_detection tool")
            
            # Add text_prompt if missing
            if "text_prompt" not in fixed_input or not fixed_input["text_prompt"]:
                if extractor:
                    text_prompt = extractor.extract_text_prompt_from_context(tool_name)
                else:
                    text_prompt = None
                
                if text_prompt:
                    logger.info(f"Adding missing text_prompt parameter: {text_prompt}")
                    fixed_input["text_prompt"] = text_prompt
                else:
                    # Use a generic prompt as fallback
                    fallback_prompt = "objects"
                    logger.warning(f"Using fallback text_prompt: {fallback_prompt}")
                    fixed_input["text_prompt"] = fallback_prompt
        
        # Fix image_path parameter if it's a placeholder
        if "image_path" in fixed_input:
            current_path = fixed_input["image_path"]
            # Check if it's a placeholder value
            placeholder_patterns = [
                "path_to_image", "path/to/image", "IMAGE", "actual_image_path",
                "image.jpg", "image.png", "sample_image", "input_image",
                "Image: /path/to/image", "Image: path"
            ]
            
            is_placeholder = any(placeholder in current_path.lower() for placeholder in placeholder_patterns)
            
            if is_placeholder:
                if extractor:
                    image_path = extractor.extract_image_path_from_context()
                else:
                    image_path = None
                
                if image_path:
                    logger.info(f"Replacing placeholder '{current_path}' with real image path: {image_path}")
                    fixed_input["image_path"] = image_path
                else:
                    logger.warning(f"Found placeholder '{current_path}' but no real image path available")
                    # Try to extract from the placeholder itself if it contains a real path
                    if current_path.startswith("Image: "):
                        potential_path = current_path[7:]  # Remove "Image: " prefix
                        if potential_path and not any(p in potential_path.lower() for p in ["path_to_image", "path/to/image"]):
                            logger.info(f"Extracting path from placeholder: {potential_path}")
                            fixed_input["image_path"] = potential_path
        
        # Fix text_prompt parameter if it's too generic
        if "text_prompt" in fixed_input:
            current_prompt = fixed_input["text_prompt"]
            generic_prompts = ["objects", "classify objects", "detect objects", "segment objects"]
            
            if current_prompt.lower() in generic_prompts:
                # Try to extract a more specific prompt from the query
                if extractor:
                    better_prompt = extractor.extract_text_prompt_from_context(tool_name)
                else:
                    better_prompt = None
                
                if better_prompt:
                    logger.info(f"Replacing generic prompt '{current_prompt}' with: {better_prompt}")
                    fixed_input["text_prompt"] = better_prompt
        
        # Ensure default values for optional parameters
        if tool_name in ["detection", "classification", "segmentation"]:
            if "confidence_threshold" not in fixed_input:
                if tool_name == "detection":
                    fixed_input["confidence_threshold"] = 0.6
                elif tool_name == "classification":
                    fixed_input["confidence_threshold"] = 0.5
                elif tool_name == "segmentation":
                    fixed_input["confidence_threshold"] = 0.5
            else:
                # Override confidence thresholds that are too high for segmentation
                if tool_name == "segmentation" and fixed_input.get("confidence_threshold", 0.5) > 0.8:
                    logger.warning(f"Overriding high confidence threshold {fixed_input['confidence_threshold']} with 0.5 for segmentation")
                    fixed_input["confidence_threshold"] = 0.5

        # CRITICAL FIX: Extract meters_per_pixel from query instead of using hardcoded default
        # This ensures query-specified GSD values are used when LLM doesn't provide them
        if "meters_per_pixel" not in fixed_input:
            # Try to extract from query first (if parameter_extractor is available)
            if extractor and extractor.input_query:
                # Extract GSD from query using regex patterns
                extracted_gsd = self._extract_gsd_from_query(extractor.input_query)
                fixed_input["meters_per_pixel"] = extracted_gsd
                logger.debug(f"Extracted meters_per_pixel from query: {extracted_gsd}")
            else:
                # Fallback to default only if extraction is not available
                fixed_input["meters_per_pixel"] = 0.3
                logger.debug("Using default meters_per_pixel: 0.3")
        else:
            logger.debug(f"Preserving LLM-provided meters_per_pixel: {fixed_input['meters_per_pixel']}")

        return fixed_input

    def _extract_gsd_from_query(self, query_text: str) -> float:
        """
        Extract GSD (meters per pixel) from query text.

        Args:
            query_text: Natural language query string

        Returns:
            Extracted GSD value in meters per pixel, or 0.3 as default
        """
        if not query_text:
            return 0.3

        query_lower = query_text.lower()

        # Look for GSD = X.XX m/px pattern (with optional parentheses)
        # Matches: "(GSD = 0.05 m/px)", "GSD = 0.05 m/px", "GSD: 0.05 m/px"
        gsd_match = re.search(r'gsd\s*[=:]\s*([0-9.]+)\s*m\s*/\s*px', query_lower)
        if gsd_match:
            gsd_value = float(gsd_match.group(1))
            logger.debug(f"Extracted GSD from query: {gsd_value} m/px")
            return gsd_value

        # Look for other GSD patterns
        gsd_patterns = [
            r'([0-9.]+)\s*m\s*/\s*px',  # "0.05 m/px"
            r'([0-9.]+)\s*m\s*/\s*pixel',  # "0.05 m/pixel"
            r'([0-9.]+)\s*meters?\s*per\s*pixel',  # "0.05 meters per pixel"
            r'resolution[:\s]*([0-9.]+)\s*m'  # "resolution: 0.05 m"
        ]

        for pattern in gsd_patterns:
            match = re.search(pattern, query_lower)
            if match:
                gsd_value = float(match.group(1))
                logger.debug(f"Extracted GSD from query using pattern '{pattern}': {gsd_value} m/px")
                return gsd_value

        # Default GSD value when not found in query
        logger.debug("No GSD found in query, using default: 0.3 m/px")
        return 0.3
    
    def filter_tool_arguments_to_benchmark_format(self, tool_name: str, 
                                                  tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter tool arguments to only include fields in the benchmark format.
        
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments dictionary
        
        Returns:
            Filtered arguments dictionary
        """
        # Define benchmark format fields for each tool
        benchmark_fields = {
            "detection": ["image_path", "text_prompt", "confidence_threshold", "meters_per_pixel"],
            "segmentation": ["image_path", "text_prompt", "confidence_threshold", "meters_per_pixel"],
            "classification": ["image_path", "text_prompt", "confidence_threshold", "meters_per_pixel"],
            "buffer": ["distance", "buffer_class", "geometry_count", "meters_per_pixel"],
            "overlap": ["source_class", "target_class", "source_polygon_count", "target_polygon_count", "meters_per_pixel"],
            "containment": ["container_class", "contained_class", "meters_per_pixel"],
            "object_count_aoi": ["object_class", "aoi_class", "meters_per_pixel"],
        }
        
        allowed_fields = benchmark_fields.get(tool_name, list(tool_args.keys()))
        
        filtered_args = {k: v for k, v in tool_args.items() if k in allowed_fields}
        
        return filtered_args
    
    def ensure_complete_tool_arguments(self, tool_name: str, tool_args: Dict[str, Any], 
                                      step_plan: str = "") -> Dict[str, Any]:
        """
        Ensure tool arguments have all required fields for benchmark format.
        
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments dictionary
            step_plan: The step plan description
        
        Returns:
            Complete tool arguments dictionary
        """
        complete_args = tool_args.copy()
        
        # Define required fields for each tool
        required_fields = {
            "detection": ["image_path", "text_prompt"],
            "segmentation": ["image_path", "text_prompt"],
            "classification": ["image_path", "text_prompt"],
            "buffer": ["distance", "buffer_class"],
            "overlap": ["source_class", "target_class"],
            "containment": ["container_class", "contained_class"],
            "object_count_aoi": ["object_class", "aoi_class"],
        }
        
        required = required_fields.get(tool_name, [])
        
        for field in required:
            if field not in complete_args or complete_args[field] is None:
                logger.warning(f"Missing required field '{field}' for tool '{tool_name}'")
                # Try to provide a default or extract from context
                if self.parameter_extractor:
                    if field == "text_prompt":
                        complete_args[field] = self.parameter_extractor.extract_text_prompt_from_context(tool_name)
                    elif field == "image_path":
                        complete_args[field] = self.parameter_extractor.extract_image_path_from_context()
        
        return complete_args

