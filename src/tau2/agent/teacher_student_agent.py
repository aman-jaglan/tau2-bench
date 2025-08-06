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
- Execute the EXACT tool call specified by the teacher
- Do NOT substitute with a different tool name
- Do NOT add any other tool calls
- After you receive the tool's response, the system will provide you with the next step

CRITICAL: 
- You MUST use the exact tool name provided by the teacher
- Only the argument VALUES should be replaced with real values from the ticket

## IMPORTANT
You will receive ONE step at a time. Execute only that step.

## VERIFICATION NOTE
When executing the final done() step, remember that you should first verify the issue mentioned in the ticket has been resolved by using appropriate assertion tools available to you."""
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
        
        # Check if we need to do assertion before done()
        need_assertion = False
        if current_tool_call['name'] == 'done' and state.current_step > 0:
            # Check if we've already done any assertions in recent messages
            recent_assertions = []
            for msg in state.messages[-5:]:  # Check last 5 messages for assertions
                if isinstance(msg, AssistantMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.name.startswith('assert_'):
                            recent_assertions.append(tc.name)
            
            if not recent_assertions:
                need_assertion = True
                logger.info("Need to verify issue resolution before done()")
            else:
                logger.info(f"Recent assertions found: {recent_assertions}")
        
        if need_assertion:
            # Create assertion instruction
            step_instruction = f"""## VERIFICATION REQUIRED

Before completing the task, you must verify the issue is resolved.

Current teacher step: done()

However, FIRST you need to:
1. Look at the ticket to understand what issue needs to be resolved
2. Call an appropriate assertion tool to verify the fix worked

The ticket states: {self.task.ticket}

Choose an assertion tool that verifies the issue mentioned in the ticket.

IMPORTANT: 
- Execute ONE assertion tool call at a time
- After the assertion, the system will determine if more verification is needed
- Continue with assertions until the issue is fully verified"""
            
        else:
            # Normal step execution
            step_instruction = f"""## CURRENT STEP TO EXECUTE

You must execute this EXACT tool:

Tool Name: {current_tool_call['name']}
Arguments Template: {json.dumps(current_tool_call['arguments'])}

CRITICAL INSTRUCTIONS:
1. You MUST call the tool named "{current_tool_call['name']}" - no substitutions
2. Do NOT call a different tool even if you think it would help
3. Do NOT check status or gather information first
4. For the arguments: Replace placeholder values (like "C1001", "L1002") with real values from the ticket
5. Execute ONLY this single tool call

Example: If the teacher says "grant_app_permission", you MUST call grant_app_permission, not check_app_permissions or any other tool."""
        
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
            
            # Handle state progression
            if assistant_message.tool_calls:
                executed_assertion = any(tc.name.startswith('assert_') for tc in assistant_message.tool_calls)
                current_step_is_done = (state.current_step < len(state.execution_plan) and 
                                       state.execution_plan[state.current_step]['name'] == 'done')
                
                if executed_assertion and current_step_is_done:
                    # We just executed an assertion before done(), don't increment
                    # We'll execute done() in the next iteration
                    logger.info("Assertion completed, will execute done() next")
                else:
                    # Normal step increment
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