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


class TeacherStudentAgentState(LLMAgentState):
    """Extended state for teacher-student agent with execution plan tracking."""
    execution_plan: Optional[List[Dict[str, Any]]] = None
    current_step: int = 0
    total_steps: int = 0

class TeacherStudentSoloAgent(LLMSoloAgent):
    """
    Enhanced solo agent that uses teacher thinking traces.
    Implements stateful orchestration for sequential execution.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        execution_plans: Dict[str, List[Dict[str, Any]]],
        student_llm: str,
        student_llm_args: Optional[dict] = None,
        trace_extraction_llm: Optional[str] = None
    ):
        """Initialize with execution plans."""
        # Store attributes before calling super().__init__
        self.execution_plans = execution_plans
        
        # Debug: Log task ID and available plans
        logger.info(f"Task ID from task object: {task.id}")
        logger.info(f"Available plan keys (first 5): {list(execution_plans.keys())[:5]}")
        
        self.current_plan = execution_plans.get(task.id, [])
        
        if not self.current_plan:
            logger.warning(f"No execution plan found for task ID: {task.id}")
        else:
            logger.info(f"Found execution plan for task ID: {task.id} ({len(self.current_plan)} steps)")
        
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
        """Simple prompt for single-step execution."""
        
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
        
        # If we have an execution plan, modify the instruction
        if self.current_plan:
            enhanced_prompt = f"""{base_prompt}

## TASK
The teacher has provided a plan. Your job is to execute the current step.

## INSTRUCTIONS
- Execute the SINGLE tool call that will be provided in the state
- Do NOT add any other tool calls
- After you receive the tool's response, the system will provide you with the next step

CRITICAL: The teacher's argument values are examples. You must use real values from the ticket and environment.

## IMPORTANT
You will receive ONE step at a time. Execute only that step."""
        else:
            enhanced_prompt = base_prompt
            
        return enhanced_prompt
    
    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> TeacherStudentAgentState:
        """Get the initial state with execution plan."""
        if message_history is None:
            message_history = []
        
        # Initialize the extended state
        state = TeacherStudentAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
            execution_plan=self.current_plan,
            current_step=0,
            total_steps=len(self.current_plan)
        )
        
        return state
    
    def generate_next_message(
        self, 
        message: Optional[ValidAgentInputMessage], 
        state: TeacherStudentAgentState
    ) -> tuple[AssistantMessage, TeacherStudentAgentState]:
        """Generate next message with stateful orchestration."""
        
        # Log the current state
        logger.info(f"=== GENERATING MESSAGE FOR TASK: {self.task.id} ===")
        logger.info(f"Current step: {state.current_step} / {state.total_steps}")
        logger.info(f"Number of messages in state: {len(state.messages)}")
        
        # If no execution plan, fall back to normal behavior
        if not state.execution_plan or state.total_steps == 0:
            logger.warning(f"No execution plan for task {self.task.id}, using default behavior")
            return super().generate_next_message(message, state)
        
        # Check if we've completed all steps
        if state.current_step >= state.total_steps:
            logger.info(f"All steps completed for task {self.task.id}")
            # Create a done message
            done_message = AssistantMessage(
                role="assistant",
                content=self.STOP_TOKEN,
                tool_calls=None
            )
            state.messages.append(done_message)
            return done_message, state
        
        # Get the current step from the execution plan
        current_tool_call = state.execution_plan[state.current_step]
        logger.info(f"Executing step {state.current_step + 1}: {current_tool_call['name']}")
        
        # Create a focused prompt that includes only the current step
        step_instruction = f"""## CURRENT STEP TO EXECUTE

You must execute this specific tool call:

Tool: {current_tool_call['name']}
Arguments: {json.dumps(current_tool_call['arguments'])}

IMPORTANT: 
- Execute ONLY this single tool call
- Do not add any other tool calls

CRITICAL: The arguments shown above are examples. You must use real values from the ticket and environment."""
        
        # Create a temporary system message with the step instruction
        focused_system_message = SystemMessage(
            role="system",
            content=self.system_prompt + "\n\n" + step_instruction
        )
        
        # Update messages if needed
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is None:
            assert len(state.messages) == 0, "Message history should be empty"
        else:
            state.messages.append(message)
        
        # Generate with focused instruction
        messages = [focused_system_message] + state.messages
        
        try:
            assistant_message = generate(
                model=self.llm,
                tools=self.tools,
                messages=messages,
                tool_choice="required",
                **self.llm_args,
            )
            
            if not assistant_message.is_tool_call():
                raise ValueError("LLMSoloAgent only supports tool calls.")
            
            # Check if it's a stop message
            assistant_message = self._check_if_stop_toolcall(assistant_message)
            
            # Update state
            state.messages.append(assistant_message)
            state.current_step += 1
            
            # Log progress
            if assistant_message.tool_calls:
                logger.info(f"Successfully executed: {[tc.name for tc in assistant_message.tool_calls]}")
                logger.info(f"Progress: {state.current_step}/{state.total_steps} steps completed")
            
            return assistant_message, state
            
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