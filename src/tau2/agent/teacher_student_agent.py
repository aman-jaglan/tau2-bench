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
        elif len(self.current_plan) == 1 and self.current_plan[0]['name'] == 'done':
            logger.warning(f"Task {task.id} has no assistant actions, only done()")
        else:
            logger.info(f"Found execution plan for task ID: {task.id} ({len(self.current_plan)} steps)")
            # Log the actual steps for debugging
            for i, step in enumerate(self.current_plan):
                logger.debug(f"  Step {i+1}: {step['name']}({step.get('arguments', {})})")
        
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
        """Guided prompt for following teacher's execution plan."""
        
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
        
        # If we have an execution plan, provide clear guidance
        if self.current_plan:
            enhanced_prompt = f"""{base_prompt}

## EXECUTION CONTEXT
A teacher has already analyzed this issue and provided the exact sequence of tools to resolve it.
Your expertise is needed to execute this plan with the correct customer-specific values.

## KEY PRINCIPLE
Trust the teacher's tool selection completely. The teacher has already determined which tools are needed.
Your role is to execute them with the correct values from the ticket.

## CRITICAL GUIDANCE
1. Tool Selection: Use the EXACT tool name provided by the teacher - no substitutions
2. Argument Values: Extract REAL customer data from the ticket (customer IDs, phone numbers, etc.)
3. Sequential Execution: Execute only the current step, then wait for the next instruction
4. No Additional Tools: The plan is complete - do not add diagnostic or verification tools

## COMMON PATTERNS TO AVOID
- Calling check_* or verify_* tools when not in the plan
- Substituting similar tools (e.g., using toggle_data instead of reset_apn_settings)
- Using placeholder values like "C1001" instead of extracting real customer data
- Adding multiple tool calls when only one is specified

## EXECUTION FLOW
You will receive one tool at a time to execute. Focus solely on executing that specific tool correctly."""
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
        
        # Check if the tool exists
        tool_names = [t.name for t in self.tools]
        if current_tool_call['name'] not in tool_names and current_tool_call['name'] != 'done':
            logger.warning(f"Tool '{current_tool_call['name']}' not found in available tools: {tool_names[:10]}...")
            # Skip this step
            state.current_step += 1
            return self.generate_next_message(message, state)
        
        # Check if we need to do assertion before done()
        need_assertion = False
        if current_tool_call['name'] == 'done' and state.current_step > 0:
            # Check if we've already done any assertions in recent messages
            recent_assertions = []
            for msg in state.messages[-5:]:  # Check last 5 messages for assertions
                if isinstance(msg, AssistantMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.name.startswith('assert_') or tc.name == 'can_send_mms':
                            recent_assertions.append(tc.name)
            
            if not recent_assertions:
                need_assertion = True
                logger.info("Need to verify issue resolution before done()")
            else:
                logger.info(f"Recent assertions found: {recent_assertions}")
        
        if need_assertion:
            # Create assertion instruction for MMS issues
            step_instruction = f"""## VERIFICATION REQUIRED

Before completing the task, you must verify the issue is resolved.

Current teacher step: done()

However, FIRST you need to verify the MMS issue is fixed.

The ticket states: {self.task.ticket}

Since this is an MMS issue, you MUST call: can_send_mms()

IMPORTANT: 
- Call can_send_mms() to verify MMS functionality
- After verification, the system will execute done() automatically"""
            
        else:
            # Normal step execution - clear and focused instruction
            step_instruction = f"""## CURRENT STEP IN PLAN

The teacher has determined you need to call: {current_tool_call['name']}

Expected arguments structure: {json.dumps(current_tool_call['arguments'])}

IMPORTANT REMINDERS:
1. This is the ONLY tool to call right now - {current_tool_call['name']}
2. Extract real values from the ticket to replace any placeholders
3. Do not add any other tool calls or diagnostics
4. Trust that this tool is correct for solving the issue

Execute {current_tool_call['name']} now with appropriate values from the ticket."""
        
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
        
        # Limit retries to avoid infinite loops
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
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
                
                # Validate the tool call matches expected
                executed_tools = [tc.name for tc in assistant_message.tool_calls] if assistant_message.tool_calls else []
                expected_tool = current_tool_call['name']
                
                # Special case: assertions are acceptable before done()
                if expected_tool == 'done' and need_assertion:
                    if any(tc in executed_tools for tc in ['can_send_mms', 'assert_can_send_mms']):
                        logger.info(f"Valid assertion executed: {executed_tools}")
                    else:
                        logger.warning(f"Expected MMS assertion but got: {executed_tools}")
                        retry_count += 1
                        continue
                # Check if the exact expected tool was called
                elif expected_tool not in executed_tools:
                    logger.warning(f"Tool mismatch! Expected: {expected_tool}, Got: {executed_tools}")
                    logger.warning(f"Retry {retry_count + 1}/{max_retries}: Reinforcing instruction")
                    
                    # Make instruction even more explicit on retry
                    step_instruction = f"""## CORRECTION NEEDED

You called {executed_tools[0] if executed_tools else 'unknown'}, but the teacher's plan specifically requires {expected_tool}.

The teacher has already determined that {expected_tool} is the correct tool for this step.
Please trust the plan and call {expected_tool} instead.

Tool to call: {expected_tool}
Arguments structure: {json.dumps(current_tool_call['arguments'])}

Remember: Use real customer values from the ticket, not placeholder values."""
                    
                    focused_system_message = SystemMessage(
                        role="system",
                        content=self.system_prompt + "\n\n" + step_instruction
                    )
                    messages[0] = focused_system_message
                    retry_count += 1
                    continue
                
                # Update state
                state.messages.append(assistant_message)
                
                # Handle state progression
                if assistant_message.tool_calls:
                    executed_assertion = any(tc.name in ['can_send_mms', 'assert_can_send_mms'] for tc in assistant_message.tool_calls)
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
                    logger.info(f"Successfully executed: {executed_tools}")
                    logger.info(f"Progress: {state.current_step}/{state.total_steps} steps completed")
                
                return assistant_message, state
                
            except Exception as e:
                logger.error(f"Error generating message for task {self.task.id}: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise
        
        # If we exhausted retries, skip this step
        logger.error(f"Failed to execute correct tool after {max_retries} retries, skipping step")
        state.current_step += 1
        return self.generate_next_message(message, state)


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