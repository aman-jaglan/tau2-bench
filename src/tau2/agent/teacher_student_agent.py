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

## TEACHER'S ANALYSIS
A teacher has previously analyzed a similar task. Use their thinking to guide your approach:

{self.current_trace}

## YOUR TASK
Important: The teacher's analysis shows a comprehensive approach, but you should focus on the specific issue in the ticket. Execute only the necessary actions to resolve the ticket requirements, then call done() immediately. Do not continue with extra diagnostic steps once the issue is resolved.

Key instruction: After each action, check if the ticket requirements are met. If yes, call done() without executing additional steps from the teacher's analysis. You can see all previous actions in the conversation history."""
        else:
            enhanced_prompt = base_prompt
            
        return enhanced_prompt
    
    def generate_next_message(
        self, 
        message: Optional[ValidAgentInputMessage], 
        state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """Generate next message using parent's logic - no special processing."""
        return super().generate_next_message(message, state)


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