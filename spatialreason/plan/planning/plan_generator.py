"""
Plan generation module for spatialreason.plan

This module contains the PlanGenerator class which handles generation of
execution plans using semantic filtering.

Classes:
    PlanGenerator: Generates execution plans using semantic filtering
"""

import json
import logging
from typing import List, Dict, Any, Optional

# Setup module logger
logger = logging.getLogger(__name__)


class PlanGenerator:
    """
    Generates execution plans using semantic filtering.
    
    This class handles:
    - Plan generation using LLM
    - Semantic filtering of tools
    - Toolkit diversity management
    - Plan validation and fixing
    """
    
    def __init__(self, planner_llm=None, semantic_filter=None, toolkit_list=None):
        """
        Initialize the plan generator.
        
        Args:
            planner_llm: LLM instance for plan generation
            semantic_filter: Semantic filter for tool selection
            toolkit_list: Available toolkits
        """
        self.planner_llm = planner_llm
        self.semantic_filter = semantic_filter
        self.toolkit_list = toolkit_list
        self.devise_plan = ""
        self.have_plan = 0
    
    def generate_plan(self, input_query: str, prompt_templates: Dict[str, str]) -> str:
        """
        Generate execution plan using semantic filtering.
        
        Args:
            input_query: The input query
            prompt_templates: Dictionary of prompt templates
        
        Returns:
            Generated plan string
        """
        if self.devise_plan == "":
            # Apply semantic filtering to get relevant tools only
            filtered_toolkits_prompt = self._get_semantically_filtered_toolkits(input_query)
            
            # Build system prompt
            system = self._build_system_prompt(
                base_template=prompt_templates.get("PROMPT_OF_PLAN_MAKING", ""),
                toolkit_list=filtered_toolkits_prompt
            )
            
            # Build user prompt
            user = self._build_user_prompt(
                template=prompt_templates.get("PROMPT_OF_PLAN_MAKING_USER", ""),
                user_query=input_query
            )
            
            # Reset conversation and feed new prompts
            if self.planner_llm:
                self.planner_llm._reset_and_feed(system, user)
                
                # Generate plan
                self.devise_plan = self.planner_llm.predict()
                self.have_plan = 1
                
                logger.info("GeneratePlan: Plan generated using semantically filtered tools only")
        
        return self.devise_plan
    
    def _get_semantically_filtered_toolkits(self, input_query: str) -> str:
        """
        Apply semantic filtering to toolkits for plan generation.
        
        Args:
            input_query: The input query
        
        Returns:
            Filtered toolkit prompt string
        """
        if not self.semantic_filter or not self.semantic_filter.retrieval_available:
            logger.info("Semantic filtering not available, using all toolkits for planning")
            return self._generate_fallback_toolkit_prompt()
        
        try:
            logger.info(f"Applying semantic filtering to toolkits for query: '{input_query[:50]}...'")
            
            # Get all available tools
            all_tools = []
            for toolkit in self.toolkit_list.tool_kits:
                all_tools.extend(toolkit.tool_lists)
            
            # Apply semantic filtering
            filtered_tools = self.semantic_filter.filter_tools_by_relevance(
                query=input_query,
                available_tools=all_tools,
                top_k=5
            )
            
            if not filtered_tools:
                logger.warning("No relevant tools found, using all toolkits")
                return self._generate_fallback_toolkit_prompt()
            
            # Ensure tool diversity
            diverse_tools = self._ensure_toolkit_diversity(filtered_tools)
            
            # Build filtered toolkit prompt
            filtered_prompt = "Available toolkits and their tools:\n"
            
            # Group tools by category
            perception_tools = [t for t in diverse_tools if t.api_dest.get('category') == 'perception']
            spatial_tools = [t for t in diverse_tools if t.api_dest.get('category') == 'spatial_relations']
            analysis_tools = [t for t in diverse_tools if t.api_dest.get('category') not in ['perception', 'spatial_relations']]
            
            # Add tools to prompt
            if perception_tools:
                filtered_prompt += "Toolkit 0 (Perception): "
                filtered_prompt += ", ".join([t.api_dest['name'] for t in perception_tools]) + "\n"
            
            if spatial_tools:
                filtered_prompt += "Toolkit 1 (Spatial Relations): "
                filtered_prompt += ", ".join([t.api_dest['name'] for t in spatial_tools]) + "\n"
            
            if analysis_tools:
                filtered_prompt += "Toolkit 2 (Analysis): "
                filtered_prompt += ", ".join([t.api_dest['name'] for t in analysis_tools]) + "\n"
            
            return filtered_prompt
        
        except Exception as e:
            logger.warning(f"Failed to apply semantic filtering: {e}")
            return self._generate_fallback_toolkit_prompt()
    
    def _ensure_toolkit_diversity(self, filtered_tools: List[Any]) -> List[Any]:
        """
        Ensure diversity across toolkits in filtered tools.
        
        Args:
            filtered_tools: List of filtered tools
        
        Returns:
            Diverse list of tools
        """
        # Group by category
        by_category = {}
        for tool in filtered_tools:
            category = tool.api_dest.get('category', 'other')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(tool)
        
        # Select diverse tools
        diverse = []
        for category, tools in by_category.items():
            # Take up to 2 tools per category
            diverse.extend(tools[:2])
        
        return diverse
    
    def _generate_fallback_toolkit_prompt(self) -> str:
        """Generate fallback toolkit prompt with all available tools."""
        fallback_prompt = "Available toolkits and their tools:\n"
        
        for i, toolkit in enumerate(self.toolkit_list.tool_kits):
            toolkit_name = "Perception" if i == 0 else "Spatial Relations" if i == 1 else "Analysis"
            tool_names = [tool.api_dest["name"] for tool in toolkit.tool_lists]
            fallback_prompt += f"Toolkit {i} ({toolkit_name}): {', '.join(tool_names)}\n"
        
        return fallback_prompt
    
    def _build_system_prompt(self, base_template: str, toolkit_list: str = "") -> str:
        """
        Build system prompt for plan generation.
        
        Args:
            base_template: Base prompt template
            toolkit_list: Toolkit list string
        
        Returns:
            Complete system prompt
        """
        return base_template.replace("{toolkit_list}", toolkit_list)
    
    def _build_user_prompt(self, template: str, user_query: str) -> str:
        """
        Build user prompt for plan generation.
        
        Args:
            template: Prompt template
            user_query: User query
        
        Returns:
            Complete user prompt
        """
        return template.replace("{user_query}", user_query)
    
    def validate_and_fix_toolkit_ids(self, plan_json: List[Dict[str, Any]], 
                                    available_toolkits: int) -> List[Dict[str, Any]]:
        """
        Validate and fix toolkit IDs in the plan.
        
        Args:
            plan_json: Plan JSON list
            available_toolkits: Number of available toolkits
        
        Returns:
            Fixed plan JSON list
        """
        fixed_plan = []
        
        for step in plan_json:
            if isinstance(step, dict) and "toolkit_id" in step:
                toolkit_id = step["toolkit_id"]
                
                # Fix out-of-bounds toolkit IDs
                if toolkit_id >= available_toolkits:
                    logger.warning(f"Fixing out-of-bounds toolkit_id {toolkit_id} to {available_toolkits - 1}")
                    step["toolkit_id"] = available_toolkits - 1
            
            fixed_plan.append(step)
        
        return fixed_plan

