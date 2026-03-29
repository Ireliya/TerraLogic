"""
LLM Response Enhancement Utility
Provides scene context and domain knowledge to enhance LLM responses for spatial reasoning queries.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging


class LLMResponseEnhancer:
    """
    Utility class to enhance LLM responses with domain-specific scene context
    and expert-level interpretations for remote sensing spatial analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the response enhancer with scene context configuration.
        
        Args:
            config_path: Path to scene context configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.scene_contexts = {}
        self.enhancement_guidelines = {}
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "scene_context_config.yaml"
        
        self.load_config(config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load scene context configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Extract scene contexts and guidelines
            self.scene_contexts = self.config.get('scene_context_types', {})
            self.enhancement_guidelines = self.config.get('llm_response_enhancement_guidelines', {})
            
            self.logger.info(f"Loaded scene context configuration with {len(self.scene_contexts)} scene types")
            
        except Exception as e:
            self.logger.error(f"Failed to load scene context config from {config_path}: {e}")
            # Provide minimal fallback
            self.scene_contexts = {}
            self.enhancement_guidelines = {
                'general_principles': [
                    'Provide context about what spatial patterns mean in real-world terms',
                    'Connect numerical results to practical implications'
                ]
            }
    
    def identify_relevant_scenes(self, spatial_analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify which scene types are relevant based on spatial analysis results.
        
        Args:
            spatial_analysis_results: Results from spatial analysis tools
            
        Returns:
            List of relevant scene contexts with matching criteria
        """
        relevant_scenes = []
        
        # Extract key spatial metrics from results
        metrics = self._extract_spatial_metrics(spatial_analysis_results)
        
        # Check each scene type for relevance
        for scene_id, scene_config in self.scene_contexts.items():
            relevance_score = self._calculate_scene_relevance(scene_config, metrics)
            
            if relevance_score > 0.3:  # Threshold for relevance
                relevant_scenes.append({
                    'scene_id': scene_id,
                    'scene_name': scene_config.get('name', scene_id),
                    'relevance_score': relevance_score,
                    'context': scene_config,
                    'matching_patterns': self._get_matching_patterns(scene_config, metrics)
                })
        
        # Sort by relevance score
        relevant_scenes.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return relevant_scenes
    
    def generate_context_prompt(self, spatial_analysis_results: Dict[str, Any], 
                               query: str = "") -> str:
        """
        Generate a context prompt to enhance LLM responses based on spatial analysis results.
        
        Args:
            spatial_analysis_results: Results from spatial analysis tools
            query: Original user query for additional context
            
        Returns:
            Context prompt string to prepend to LLM input
        """
        relevant_scenes = self.identify_relevant_scenes(spatial_analysis_results)
        
        if not relevant_scenes:
            return self._get_general_enhancement_prompt()
        
        # Build context prompt
        context_parts = [
            "SPATIAL ANALYSIS CONTEXT:",
            "Based on the spatial analysis results, consider the following scene contexts when formulating your response:\n"
        ]
        
        # Add relevant scene contexts
        for i, scene in enumerate(relevant_scenes[:3], 1):  # Limit to top 3 most relevant
            scene_context = scene['context']
            context_parts.extend([
                f"{i}. {scene['scene_name']}:",
                f"   Pattern: {scene_context.get('spatial_pattern', 'N/A')}",
                f"   Context: {scene_context.get('interpretation_context', '').strip()}",
                f"   Enhancement: {scene_context.get('response_enhancement', '').strip()}",
                ""
            ])
        
        # Add general guidelines
        guidelines = self.enhancement_guidelines.get('general_principles', [])
        if guidelines:
            context_parts.extend([
                "RESPONSE GUIDELINES:",
                *[f"- {guideline}" for guideline in guidelines],
                ""
            ])
        
        context_parts.extend([
            "Please provide an expert-level interpretation that goes beyond just numerical values,",
            "incorporating the relevant scene context and practical implications.\n"
        ])
        
        return "\n".join(context_parts)
    
    def enhance_response(self, llm_response: str, spatial_analysis_results: Dict[str, Any]) -> str:
        """
        Post-process an LLM response to add additional context if needed.
        
        Args:
            llm_response: Original LLM response
            spatial_analysis_results: Spatial analysis results for context
            
        Returns:
            Enhanced response with additional context
        """
        # For now, return the original response
        # This could be extended to add post-processing enhancements
        return llm_response
    
    def get_scene_context_summary(self) -> Dict[str, Any]:
        """Get a summary of available scene contexts."""
        summary = {
            'total_scene_types': len(self.scene_contexts),
            'scene_types': {}
        }
        
        for scene_id, scene_config in self.scene_contexts.items():
            summary['scene_types'][scene_id] = {
                'name': scene_config.get('name', scene_id),
                'description': scene_config.get('description', ''),
                'spatial_pattern': scene_config.get('spatial_pattern', '')
            }
        
        return summary
    
    def _extract_spatial_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key spatial metrics from analysis results."""
        metrics = {}
        
        # Try to extract common metrics from different tool outputs
        if isinstance(results, dict):
            # Look for area proportions
            if 'area_proportion' in results:
                metrics.update(results['area_proportion'])
            
            # Look for buffer analysis results
            if 'buffer_analysis' in results:
                buffer_data = results['buffer_analysis']
                if isinstance(buffer_data, dict):
                    metrics.update(buffer_data)
            
            # Look for segmentation results with area calculations
            if 'segmentation_results' in results:
                seg_results = results['segmentation_results']
                if isinstance(seg_results, dict):
                    for category, data in seg_results.items():
                        if isinstance(data, dict) and 'area_proportion' in data:
                            metrics[f"{category}_proportion"] = data['area_proportion']
        
        return metrics
    
    def _calculate_scene_relevance(self, scene_config: Dict[str, Any], 
                                  metrics: Dict[str, float]) -> float:
        """Calculate relevance score for a scene type based on spatial metrics."""
        # Simple relevance calculation based on pattern matching
        # This could be made more sophisticated
        
        spatial_pattern = scene_config.get('spatial_pattern', '').lower()
        relevance_score = 0.0
        
        # Check for mentions of land cover types in the pattern
        land_cover_types = ['building', 'road', 'water', 'forest', 'agriculture', 'barren']
        
        for land_cover in land_cover_types:
            if land_cover in spatial_pattern:
                # Check if we have metrics for this land cover type
                proportion_key = f"{land_cover}_proportion"
                if proportion_key in metrics:
                    relevance_score += 0.2
                    
                    # Bonus for high proportions
                    if metrics[proportion_key] > 0.3:
                        relevance_score += 0.1
        
        return min(relevance_score, 1.0)  # Cap at 1.0
    
    def _get_matching_patterns(self, scene_config: Dict[str, Any], 
                              metrics: Dict[str, float]) -> List[str]:
        """Get list of matching spatial patterns for a scene."""
        patterns = []
        spatial_pattern = scene_config.get('spatial_pattern', '')
        
        if spatial_pattern:
            patterns.append(f"Matches pattern: {spatial_pattern}")
        
        return patterns
    
    def _get_general_enhancement_prompt(self) -> str:
        """Get general enhancement prompt when no specific scenes are identified."""
        guidelines = self.enhancement_guidelines.get('general_principles', [])
        
        prompt_parts = [
            "SPATIAL ANALYSIS CONTEXT:",
            "Please provide an expert-level interpretation of the spatial analysis results that:",
            *[f"- {guideline}" for guideline in guidelines],
            ""
        ]
        
        return "\n".join(prompt_parts)


# Convenience function for easy usage
def enhance_llm_prompt(spatial_analysis_results: Dict[str, Any], 
                      query: str = "", 
                      config_path: Optional[str] = None) -> str:
    """
    Convenience function to generate context prompt for LLM enhancement.
    
    Args:
        spatial_analysis_results: Results from spatial analysis tools
        query: Original user query
        config_path: Path to scene context configuration file
        
    Returns:
        Context prompt string
    """
    enhancer = LLMResponseEnhancer(config_path)
    return enhancer.generate_context_prompt(spatial_analysis_results, query)


# Example usage
if __name__ == "__main__":
    # Example spatial analysis results
    example_results = {
        'segmentation_results': {
            'building': {'area_proportion': 0.45},
            'road': {'area_proportion': 0.15},
            'forest': {'area_proportion': 0.35}
        }
    }
    
    enhancer = LLMResponseEnhancer()
    context_prompt = enhancer.generate_context_prompt(example_results, "Analyze the urban development pattern")
    print("Generated Context Prompt:")
    print(context_prompt)
