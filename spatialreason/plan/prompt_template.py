"""
Prompt templates for the spatial reasoning agent.

This module contains all prompt templates used by the planner for:
- Plan generation
- Tool calling
- Error handling
"""

# Import shared CLASS_VOCABULARY for consistent class naming
from spatialreason.config.class_vocabulary import get_class_vocabulary_prompt


FORMAT_TOOL_FUNCTIONARITY_FUNCTION = """

You will be provided with a tool, its description, name, and API documentation.
Based on this information, briefly tell me the functionality of this tool without outputting unnecessary information,
only describing the API functionality.

"""

FORMAT_TOOL_FUNCTIONARITY_USER_FUNCTION = """
Tool Name: {tool_name}.
Tool Description: {pack_description},{tool_description}.
API documentation:: {api_doc}.
Please remember, tell me the functionality of this tool without outputting unnecessary information,
ONLY describing the API functionality in short description.
"""

FORMAT_TOOLKIT_FUNCTIONARITY_FUNCTION = """
You will be provided a toolkit with a list tool, including tool functionality and name.
Based on this information, briefly tell me the functionality of this toolkit without outputting unnecessary information,
only describing the toolkit functionality.
"""

FORMAT_TOOLKIT_FUNCTIONARITY_USER_FUNCTION = """
Tool List: {tool_list}.
Please remember, tell me the functionality of this tool without outputting unnecessary information,
ONLY describing the API functionality in short description.
"""

PROMPT_OF_PLAN_MAKING = """
You will be provided with the toolkits, the clustered names of toolkits, and the descriptions of
the function of the toolkits.Your task is to interact with API toolkits to construct user queries
and use the functionalities of the toolkits to answer the queries. 
The toolkits list: {toolkit_list}

You need to identify the most suitable toolkits based on the user’s requirements, and then outline your solution plan based
on the toolkits you’ve selected.Remember, your goal is not to directly answer the query but
to identify the toolkits and provide a solution plan. 

CRITICAL INSTRUCTIONS FOR PLAN FORMAT:
- You must create a plan as a JSON array of steps
- Each step must reference a toolkit ID: 0, 1, 2, 3, or 4 (valid toolkit IDs based on available toolkits)
- Toolkit 0 = perception tools (segmentation, detection, classification, change_detection) for OPTICAL imagery
- Toolkit 1 = spatial relation tools (containment, overlap, buffer)
- Toolkit 2 = spatial statistics tools (measurement, distance, counting)
- Toolkit 3 = SAR tools (sar_detection, sar_classification) for SAR/RADAR imagery
- Toolkit 4 = IR tools (infrared_detection) for INFRARED/THERMAL imagery

REQUIRED OUTPUT FORMAT - use exactly this structure:
[{"Toolkit 0": "your plan step using perception tools"}, {"Toolkit 1": "your plan step using spatial tools"}]

TOOLKIT CAPABILITIES:
- Toolkit 0 (Perception): Contains segmentation, detection, classification, and change_detection tools for identifying objects and regions in OPTICAL imagery
- Toolkit 1 (Spatial Relations): Contains containment, overlap, and buffer tools for spatial analysis
- Toolkit 2 (Spatial Statistics): Contains measurement, distance, and counting tools for quantitative analysis
- Toolkit 3 (SAR Tools): Contains sar_detection and sar_classification for SAR/RADAR imagery analysis
- Toolkit 4 (IR Tools): Contains infrared_detection for infrared/thermal imagery analysis

MODALITY-SPECIFIC TOOLKIT SELECTION:
- For OPTICAL queries: Use Toolkit 0 (Perception) for detection/segmentation/classification/change_detection
- For BI-TEMPORAL queries: Use Toolkit 0 (Perception) with change_detection tool for temporal change analysis
- For SAR queries: Use Toolkit 3 (SAR Tools) for sar_detection or sar_classification
- For IR queries: Use Toolkit 4 (IR Tools) for infrared_detection
- For spatial analysis: Use Toolkit 1 (Spatial Relations) or Toolkit 2 (Spatial Statistics) after perception

EFFICIENCY GUIDELINES:
- When the query mentions multiple object types, use ONE perception tool call to process ALL objects
- Combine related object classes in a single text prompt for efficiency
- Choose the most appropriate perception tool based on the specific requirements of the query
- For SAR queries, ALWAYS use Toolkit 3 (SAR Tools), NOT Toolkit 0 (Optical Perception)
- For IR queries, ALWAYS use Toolkit 4 (IR Tools), NOT Toolkit 0 (Optical Perception)

STRICT REQUIREMENTS:
- Use toolkit IDs 0, 1, 2, 3, or 4 based on available toolkits
- Each step must be a JSON object with exactly one key-value pair
- The key must be "Toolkit X" where X is a valid toolkit ID
- CRITICAL: For SAR queries, use Toolkit 3, NOT Toolkit 0
- CRITICAL: For IR queries, use Toolkit 4, NOT Toolkit 0

=== CRITICAL DEPENDENCY RULES (PHASE 1 CONSTRAINT) ===
IMPORTANT: Spatial relation tools and analysis tools have HARD DEPENDENCIES on perception tools.

DEPENDENCY CONSTRAINT:
- Toolkit 1 (Spatial Relations: buffer, overlap, containment) REQUIRES Toolkit 0/3/4 (Perception) to execute FIRST
- Toolkit 2 (Spatial Statistics: count, area_measurement, distance_calculation, object_count_aoi) REQUIRES Toolkit 0/3/4 (Perception) to execute FIRST
- Toolkit 0/3/4 (Perception) has NO dependencies - can execute first

VALID TOOL SEQUENCES (ALWAYS follow these patterns):
1. Perception → Spatial Relations → Spatial Statistics
   Example: [{"Toolkit 0": "segment water and forest"}, {"Toolkit 1": "buffer water by 100m"}, {"Toolkit 2": "measure forest area in buffer"}]

2. Perception → Spatial Relations
   Example: [{"Toolkit 0": "detect buildings"}, {"Toolkit 1": "check overlap with roads"}]

3. Perception → Spatial Statistics
   Example: [{"Toolkit 0": "classify land cover"}, {"Toolkit 2": "count buildings"}]

4. Perception only
   Example: [{"Toolkit 0": "segment all objects"}]

5. SAR Perception → Spatial Relations → Spatial Statistics
   Example: [{"Toolkit 3": "detect bridges in SAR"}, {"Toolkit 1": "check overlap with water"}, {"Toolkit 2": "count bridges"}]

6. IR Perception → Spatial Statistics
   Example: [{"Toolkit 4": "detect infrared targets"}, {"Toolkit 2": "count targets in lower half"}]

INVALID TOOL SEQUENCES (NEVER use these patterns):
- ❌ Starting with Toolkit 1 (buffer, overlap, containment) without Toolkit 0/3/4 first
- ❌ Starting with Toolkit 2 (count, area_measurement) without Toolkit 0/3/4 first
- ❌ Using Toolkit 1 or 2 without any Toolkit 0/3/4 step in the plan
- ❌ Using Toolkit 0 (Optical) for SAR queries - use Toolkit 3 instead
- ❌ Using Toolkit 0 (Optical) for IR queries - use Toolkit 4 instead

EXAMPLES OF INVALID PLANS (DO NOT GENERATE):
- ❌ [{"Toolkit 1": "buffer water by 100m"}] - Missing perception step
- ❌ [{"Toolkit 2": "count buildings"}] - Missing perception step
- ❌ [{"Toolkit 0": "detect bridges in SAR"}] - SAR query using Optical toolkit
- ❌ [{"Toolkit 0": "detect infrared targets"}] - IR query using Optical toolkit

EXAMPLES OF VALID PLANS (ALWAYS GENERATE):
- ✅ [{"Toolkit 0": "segment water and forest"}, {"Toolkit 1": "buffer water by 100m"}]
- ✅ [{"Toolkit 0": "detect buildings"}, {"Toolkit 2": "count buildings"}]
- ✅ [{"Toolkit 3": "detect bridges in SAR"}, {"Toolkit 1": "check overlap with water"}]
- ✅ [{"Toolkit 4": "detect infrared targets"}, {"Toolkit 2": "count targets"}]

Divise a plan to resolve the problem:
"""

PROMPT_OF_PLAN_MAKING_USER = """
Only give the plan in the format and DO NOT generate unnecessary information.
Here is the user’s question: {user_query}.
"""

def get_prompt_of_calling_one_tool_system() -> str:
    """
    Generate the system prompt for calling one tool with dynamic CLASS_VOCABULARY.

    Returns:
        System prompt string with current CLASS_VOCABULARY
    """
    class_vocab = get_class_vocabulary_prompt()

    return f"""
You are Tool-GPT, and you can solve specific problems using given tools (functions).
You will receive a problem description and the specific method of function calls to execute the solution from the toolkit.
By invoking this API, you can obtain the results of the thought process regarding this part of the problem.
Task description: {{task_description}}.

IMPORTANT - CLASS VOCABULARY CONSTRAINT:
When calling perception tools (detection, segmentation, classification, sar_detection, sar_classification, infrared_detection),
you MUST use ONLY the following class names in the text_prompt parameter:

VALID CLASS NAMES (only use these exact names):
{class_vocab}

DO NOT use any class names outside this vocabulary. Use the exact spelling and form shown above.
Examples:
- Correct: "cars", "trees", "building", "road"
- Incorrect: "car", "tree", "buildings", "roads" (these are NOT in the vocabulary)

"""


# Legacy constant for backward compatibility - now calls the function
PROMPT_OF_CALLING_ONE_TOOL_SYSTEM = get_prompt_of_calling_one_tool_system()

PROMPT_OF_CALLING_ONE_TOOL_USER = """
Thought: {thought_text}.
Provide the accurate API input and message as you can!
"""

PROMPT_OF_PLAN_EXPLORATION = """
Let’s begin executing this step of the plan. You will be provided with documentation for all
the APIs contained within this step’s toolkit, along with the parameters required to call the
APIs. Please randomly select one API from this toolkit to satisfy the user’s requirements,
or select the specified API if the user has indicated one. Consult the usage documentation
for this API, then make the API call and provide the response. Afterward, briefly analyze
the current status and determine the next step. If the API call is successful, proceed to the
next step as planned. If it fails, invoke another API from the toolkit. If all APIs in the toolkit
have been tried and failed, revert to the previous node and revise this step. Keep the analysis
concise, ideally no more than three sentences.
"""

PROMPT_OF_THE_INTOOLKIT_ERROR_OCCURS = """
This is not your first attempt at this task. The previously called APIs have all failed, and you
are now in the intermediate state of an In-Toolkit plan exploration. Before you decide on
new actions, I will first show you the actions you have taken previously for this state. Then,
you must develop an action that is different from all these previous actions. Here are some
previous candidate actions: [previous API]. Now, please analyze the current state and then
call another API within the same toolkit where the previously failed APIs are located.
"""

PROMPT_OF_THE_CROSSTOOLKIT_ERROR_OCCURS = """
This is not your first attempt at this task. All the APIs planned within the previous toolkits
have failed, and you are now in the intermediate state of a Cross-Toolkit plan exploration.
Before you decide on new actions, I will first show you the actions you have taken previously
for this state. Then, you must develop an action that is different from all these previous
actions. Here are some previous candidate actions: [previous API, previous toolkit]. Now,
please revert to the previous node, revise the plan for this step, and use a different toolkit.
"""

PROMPT_OF_THE_OUTPUTS = """
If you believe you have obtained the result capable of answering the task, please invoke this
function to provide the final answer. Remember: the only part displayed to the user is the
final answer, so it should contain sufficient information.
Task_Description:{task_description}
"""

PROMPT_OF_THE_OUTPUTS_USER = """
Resolving Procedure:{response_thought_list}
Please provide the final answer.
"""