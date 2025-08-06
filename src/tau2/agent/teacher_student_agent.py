"""
Teacher-Student Agent implementation for τ²-bench.
Student selectively uses teacher's thinking traces.
"""
from typing import List, Optional, Dict, Any
import json
import re
from pathlib import Path
from copy import deepcopy

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import (
    LLMSoloAgent, 
    LLMGTAgent,
    LLMAgentState, 
    SYSTEM_PROMPT_SOLO,
    AGENT_SOLO_INSTRUCTION,
    SYSTEM_PROMPT_GT,
    AGENT_GT_INSTRUCTION
)
from tau2.data_model.message import (
    AssistantMessage, 
    SystemMessage, 
    Message,
    MultiToolMessage
)
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate

class TeacherStudentSoloAgent(LLMSoloAgent):
    """
    Enhanced solo agent that uses teacher thinking traces.
    Student selectively extracts relevant parts to avoid confusion.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        thinking_traces: Dict[str, str],
        student_llm: str,
        student_llm_args: Optional[dict] = None,
        trace_extraction_llm: Optional[str] = None
    ):
        """Initialize with thinking traces."""
        # Store attributes before calling super().__init__
        self.thinking_traces = thinking_traces
        
        # Debug: Log task ID and available traces
        logger.info(f"Task ID from task object: {task.id}")
        logger.info(f"Available trace keys (first 5): {list(thinking_traces.keys())[:5]}")
        
        self.current_trace = thinking_traces.get(task.id, "")
        
        if not self.current_trace:
            logger.warning(f"No trace found for task ID: {task.id}")
        else:
            logger.info(f"Found trace for task ID: {task.id} (length: {len(self.current_trace)} chars)")
        
        # Now call parent init
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            llm=student_llm,
            llm_args=student_llm_args
        )
        
    
    @property
    def system_prompt(self) -> str:
        """Override system prompt to include teacher trace as context."""
        
        # Build base prompt using parent's format
        agent_instruction = AGENT_SOLO_INSTRUCTION.format(
            stop_function_name=self.STOP_FUNCTION_NAME,
            stop_token=self.STOP_TOKEN,
        )
        
        base_prompt = SYSTEM_PROMPT_SOLO.format(
            agent_instruction=agent_instruction,
            domain_policy=self.domain_policy,
            ticket=self.task.ticket,
        )
        
        # Add teacher trace if available
        if self.current_trace:
            enhanced_prompt = f"""{base_prompt}

## IMPORTANT: FORMAT TRANSLATION GUIDE
The teacher's instructions use human-readable function notation. You MUST convert these to proper JSON tool calls:

### Translation Rules:
1. When you see: function_name()
   Generate tool call: {{"name": "function_name", "arguments": {{}}}}

2. When you see: function_name({{"key": "value", "key2": "value2"}})
   Generate tool call: {{"name": "function_name", "arguments": {{"key": "value", "key2": "value2"}}}}

3. When you see: done()
   Generate tool call: {{"name": "done", "arguments": {{}}}}

### Examples:
- toggle_airplane_mode() → {{"name": "toggle_airplane_mode", "arguments": {{}}}}
- grant_app_permission({{"app_name": "messaging", "permission": "sms"}}) → {{"name": "grant_app_permission", "arguments": {{"app_name": "messaging", "permission": "sms"}}}}
- check_network_status() → {{"name": "check_network_status", "arguments": {{}}}}

### Execution Instructions:
1. Read each "Step N:" in sequence
2. Extract ONLY the function call (ignore the "- **Why**:" explanations)
3. Convert the function call to proper JSON format using the rules above
4. Generate exactly ONE tool call per step
5. When you see "Completion Signal" or instructions to call done(), generate the done() tool call

## TEACHER'S INSTRUCTIONS
{self.current_trace}

## YOUR TASK
Execute the teacher's solution by:
1. Processing each step sequentially
2. Converting each function notation to a proper JSON tool call
3. Ignoring explanatory text (it's for context only)
4. Calling done() when indicated in the completion signal"""
        else:
            enhanced_prompt = base_prompt
            
        return enhanced_prompt
    
    def generate_next_message(
        self, 
        message: Optional[ValidAgentInputMessage], 
        state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """Generate next message using parent's logic - no special processing."""
        # Log the state before generating
        logger.info(f"=== GENERATING MESSAGE FOR TASK: {self.task.id} ===")
        logger.info(f"Number of messages in state: {len(state.messages)}")
        logger.info(f"Message type: {type(message).__name__ if message else 'None'}")
        
        # Log the full system prompt being used
        if len(state.messages) == 0:  # First message
            logger.info(f"System prompt length: {len(self.system_prompt)} chars")
            logger.debug(f"Full system prompt:\n{self.system_prompt[:500]}...")
            
            # Check for any special characters in teaching
            if self.current_trace:
                special_chars = ['```', '"""', "'''", '\\', '\n\n\n']
                for char in special_chars:
                    if char in self.current_trace:
                        logger.warning(f"Teaching contains special sequence: {repr(char)}")
        
        # Log tool availability
        logger.info(f"Available tools: {[tool.name for tool in self.tools]}")
        
        try:
            result = super().generate_next_message(message, state)
            logger.info(f"Successfully generated message for task: {self.task.id}")
            
            # Log the generated message details
            assistant_msg = result[0]
            if assistant_msg.tool_calls:
                logger.info(f"Generated tool calls: {[tc.name for tc in assistant_msg.tool_calls]}")
            else:
                logger.info(f"Generated content: {assistant_msg.content[:100] if assistant_msg.content else 'None'}")
            
            return result
        except Exception as e:
            logger.error(f"Error generating message for task {self.task.id}: {str(e)}")
            raise


class TeacherStudentGTAgent(LLMGTAgent):
    """
    Oracle plan agent enhanced with collaboration insights.
    For Objective 2: Better human-agent collaboration.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        student_llm: str,
        student_llm_args: Optional[dict] = None,
        provide_function_args: bool = True
    ):
        """Initialize with enhanced collaboration prompting."""
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            llm=student_llm,
            llm_args=student_llm_args,
            provide_function_args=provide_function_args
        )
    
    @property
    def system_prompt(self) -> str:
        """Enhanced prompt focusing on collaboration."""
        
        # Get base oracle plan instructions
        resolution_steps = self.make_agent_instructions_from_actions()
        
        enhanced_instruction = f"""{AGENT_GT_INSTRUCTION}

CRITICAL COLLABORATION PRINCIPLES:

1. USER COMMUNICATION:
   - Assume ZERO technical knowledge
   - Use simple, clear language (e.g., "turn on" not "toggle")
   - Confirm understanding before proceeding
   - Be patient with confused responses

2. ACTION GUIDANCE:
   - Break down EVERY user action into tiny steps
   - Wait for explicit confirmation after each step
   - Handle "I don't know how" with detailed guidance
   - Provide visual cues: "You should see...", "Look for..."

3. VERIFICATION OBSESSION:
   - NEVER assume an action succeeded
   - Always ask "What do you see now?"
   - Have a Plan B ready for failures
   - Use tool calls to verify state when possible

4. EMOTIONAL INTELLIGENCE:
   - Acknowledge frustration: "I understand this is frustrating"
   - Provide progress updates: "We're making good progress"
   - Stay encouraging: "You're doing great"
   - Celebrate small wins

5. COMMON PITFALLS TO AVOID:
   - Don't use technical jargon
   - Don't skip verification steps
   - Don't assume user did it correctly
   - Don't rush through steps

Remember: Success = User Success, not just technical completion."""
        
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=enhanced_instruction,
            domain_policy=self.domain_policy,
            resolution_steps=resolution_steps
        )


class TeacherStudentHardPersonaAgent(LLMGTAgent):
    """
    Special variant for Hard Persona tasks.
    Extra patience and simplification.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        student_llm: str,
        student_llm_args: Optional[dict] = None,
        provide_function_args: bool = True
    ):
        """Initialize with hard persona adaptations."""
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            llm=student_llm,
            llm_args=student_llm_args,
            provide_function_args=provide_function_args
        )
        
        # Check if this is a hard persona task
        self.is_hard_persona = "[PERSONA:Hard]" in task.id
    
    @property
    def system_prompt(self) -> str:
        """Ultra-patient prompt for hard personas."""
        
        resolution_steps = self.make_agent_instructions_from_actions()
        
        if self.is_hard_persona:
            hard_persona_instruction = f"""{AGENT_GT_INSTRUCTION}

HARD PERSONA MODE - EXTREME PATIENCE REQUIRED:

The user has VERY LOW technical knowledge and gets easily flustered.

SPECIAL ADAPTATIONS:

1. ULTRA-SIMPLE LANGUAGE:
   - "Press the button that looks like three lines" (not "tap the menu icon")
   - "The round button at the bottom of your phone" (not "home button")
   - Use analogies: "It's like turning on a light switch"

2. MICRO-STEPS:
   - Break actions into 3x more steps than normal
   - Example: "Look at your phone" → "Do you see the screen?" → "Good!"
   - Pause between each micro-step

3. CONSTANT REASSURANCE:
   - "There's no rush, take your time"
   - "It's okay if you're not sure, let's figure it out together"
   - "You're doing perfectly fine"
   - "This is tricky for everyone at first"

4. HANDLE CONFUSION:
   - User says "I don't see it": "That's okay, let's try a different way"
   - User gets frustrated: "I completely understand. Let's take a quick break if you need"
   - Wrong action: "No problem at all, that happens. Let's just..."

5. VERIFY EVERYTHING TWICE:
   - First check: "What do you see?"
   - Second check: "Just to make sure, can you describe what's on the screen?"

6. FALLBACK STRATEGIES:
   - Always have 2-3 alternative ways to explain
   - Be ready to start over without frustration
   - Consider simpler alternatives if user is stuck

Remember: Go SLOW. Be KIND. VERIFY everything."""
        else:
            hard_persona_instruction = AGENT_GT_INSTRUCTION
        
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=hard_persona_instruction,
            domain_policy=self.domain_policy,
            resolution_steps=resolution_steps
        )