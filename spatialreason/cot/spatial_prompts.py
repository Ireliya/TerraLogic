"""
Spatial analysis specific ReAct prompts for Chain-of-Thought reasoning.
Adapted from the original ReAct prompts for remote sensing and spatial analysis tasks.
"""

# System prompt for spatial analysis with function calling
SPATIAL_REACT_SYSTEM_PROMPT = """You are an expert remote sensing analyst with access to advanced spatial analysis tools. Your task is to analyze satellite imagery and provide comprehensive insights.

Available Tools:
- segmentation: Identify and segment objects in satellite imagery (buildings, water bodies, vegetation, etc.)
- detection: Detect and locate specific objects with bounding boxes
- classification: Classify land cover types and regions
- Finish: Complete the analysis with your final answer
3
Analysis Process:
At each step, you need to:
1. Think about the current status and what to do next
2. Choose the most appropriate tool for the current analysis step
3. Execute the tool and analyze the results
4. Continue reasoning until you have enough information for a comprehensive answer

Format Requirements:
Use this exact format for each reasoning step:

Thought: [Your analysis of the current situation and what you need to do next - keep it concise, max 3 sentences]
Action: [tool_name]
Action Input: {{"text_prompt": "description of what to analyze", "confidence_threshold": 0.6}}

Note: Use proper confidence thresholds - detection: 0.6, classification: 0.5, segmentation: 0.5
Do NOT confuse confidence_threshold with meters_per_pixel (0.3) - they are separate parameters!

After each tool execution, you'll receive an observation with the results. Continue this Thought-Action cycle until you have sufficient information, then use:

Action: Finish
Action Input: {{"return_type": "give_answer", "final_answer": "Your comprehensive analysis and conclusions"}}

Important Guidelines:
1. Each thought should be focused and actionable
2. Choose tools strategically based on the user's request
3. Build upon previous observations to create comprehensive insights
4. Provide detailed, expert-level analysis in your final answer
5. If you encounter issues, you can restart by saying "I give up and restart"

Task: {task_description}"""

# User prompt template
SPATIAL_REACT_USER_PROMPT = """
Analyze the following satellite imagery request:
{input_description}

Begin your step-by-step analysis!
"""

# Alternative system prompt for simpler queries
SPATIAL_REACT_SIMPLE_SYSTEM_PROMPT = """You are an expert remote sensing analyst. Analyze satellite imagery using the available tools.

Available tools: segmentation, detection, classification, Finish

Use this format:
Thought: [What you need to analyze]
Action: [tool_name] 
Action Input: {{"text_prompt": "what to find"}}

Continue until you have enough information, then:
Action: Finish
Action Input: {{"return_type": "give_answer", "final_answer": "your analysis"}}

Task: {task_description}"""

# Prompt templates for different analysis types
FLOOD_RISK_ANALYSIS_PROMPT = """You are analyzing satellite imagery for flood risk assessment. Your goal is to:
1. Identify buildings and structures
2. Identify water bodies (rivers, lakes, ponds)
3. Analyze proximity between buildings and water
4. Assess flood risk levels based on distance and topography

Use segmentation tools to identify both buildings and water bodies, then provide a comprehensive flood risk assessment."""

URBAN_PLANNING_ANALYSIS_PROMPT = """You are analyzing satellite imagery for urban planning purposes. Your goal is to:
1. Identify different land use types (residential, commercial, industrial)
2. Detect infrastructure elements (roads, buildings)
3. Classify vegetation and open spaces
4. Provide insights for urban development planning

Use a combination of detection, segmentation, and classification tools to provide comprehensive urban analysis."""

ENVIRONMENTAL_MONITORING_PROMPT = """You are analyzing satellite imagery for environmental monitoring. Your goal is to:
1. Identify and classify different land cover types
2. Detect changes in vegetation or water bodies
3. Assess environmental health indicators
4. Provide recommendations for environmental management

Focus on using classification and segmentation tools to understand the environmental characteristics."""

# Function to get appropriate prompt based on query type
def get_spatial_prompt(query: str, task_description: str = "") -> tuple[str, str]:
    """
    Get appropriate spatial analysis prompt based on the query type.
    
    Args:
        query: User's analysis request
        task_description: Specific task description
        
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    query_lower = query.lower()
    
    # Determine analysis type and select appropriate prompt
    if any(keyword in query_lower for keyword in ['flood', 'risk', 'water', 'proximity']):
        system_prompt = SPATIAL_REACT_SYSTEM_PROMPT.replace("{task_description}", 
                                                           task_description or FLOOD_RISK_ANALYSIS_PROMPT)
    elif any(keyword in query_lower for keyword in ['urban', 'planning', 'development', 'infrastructure']):
        system_prompt = SPATIAL_REACT_SYSTEM_PROMPT.replace("{task_description}", 
                                                           task_description or URBAN_PLANNING_ANALYSIS_PROMPT)
    elif any(keyword in query_lower for keyword in ['environment', 'vegetation', 'land cover', 'monitoring']):
        system_prompt = SPATIAL_REACT_SYSTEM_PROMPT.replace("{task_description}", 
                                                           task_description or ENVIRONMENTAL_MONITORING_PROMPT)
    else:
        # Use simple prompt for basic queries
        system_prompt = SPATIAL_REACT_SIMPLE_SYSTEM_PROMPT.replace("{task_description}", 
                                                                  task_description or "Analyze the satellite imagery as requested")
    
    user_prompt = SPATIAL_REACT_USER_PROMPT.replace("{input_description}", query)
    
    return system_prompt, user_prompt


# Example usage and validation
def validate_prompts():
    """Validate that prompts are properly formatted."""
    test_queries = [
        "analyze flood risk for buildings near water",
        "segment buildings in urban area", 
        "classify land cover types",
        "detect vehicles in parking lot"
    ]
    
    for query in test_queries:
        system_prompt, user_prompt = get_spatial_prompt(query)
        assert "{task_description}" not in system_prompt, f"Template not replaced in system prompt for: {query}"
        assert "{input_description}" not in user_prompt, f"Template not replaced in user prompt for: {query}"
        print(f"✅ Prompt validation passed for: {query}")


if __name__ == "__main__":
    validate_prompts()
    print("🎉 All spatial prompts validated successfully!")
