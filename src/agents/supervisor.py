"""
Supervisor Agent with Plan-and-Execute Architecture.

This agent:
1. Analyzes the user query
2. Creates a dynamic execution plan (list of steps)
3. Executes each step, passing context between them
4. Combines results into a final response

NO hardcoded query-specific logic - the LLM decides everything.
"""

from typing import Any, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.llm.providers import get_fast_llm, get_reasoning_llm, LLMProvider
from src.utils.logger import get_logger
from src.data_loader import DataLoader
from src.core.state import AgentState, ExecutionPlan, PlanStep
from src.core.data_context import DataContext
from src.agents.data_explorer import DataExplorerAgent
from src.agents.sql_agent import SQLQueryAgent
from src.agents.viz_agent import VisualizationAgent
from src.agents.analytics_agent import AnalyticsAgent

logger = get_logger("cpg_agent.supervisor")


PLANNER_PROMPT = """You are a Senior Data Analyst planning how to answer a business question.

DATA: {data_context}

AGENTS (your tools):
1. data_explorer - Quick data overview, schema, quality check
2. sql_agent - Query data, aggregate, filter, calculate metrics
3. viz_agent - Create charts (bar, line, pie, heatmap, scatter)
4. analytics_agent - Statistical analysis, correlations, KPIs

THINK LIKE A DATA ANALYST:
- What's the user really asking? What insight do they need?
- What's the BEST way to answer: table, chart, or both?
- For comparisons/rankings → Bar chart
- For trends over time → Line chart  
- For distributions → Histogram
- For correlations → Scatter or heatmap
- For proportions → Donut chart (only if <6 categories)
- For exact numbers → Just show the query result

PLANNING RULES:
- 2-3 steps max. Don't over-engineer.
- Always aggregate data BEFORE visualizing
- Choose ONE best visualization, not multiple
- If user asks "summarize" → give key stats + 1-2 relevant charts

QUERY: {query}

Plan the most effective way to answer this question."""


class SupervisorAgent:
    """
    Supervisor agent using Plan-and-Execute pattern.
    
    Flow:
    1. Planner node: Creates execution plan from user query
    2. Executor node: Executes current step
    3. Check node: Decides if more steps remain or if done
    4. Respond node: Combines all results into final response
    """
    
    def __init__(
        self,
        data_loader: DataLoader,
        llm_provider: LLMProvider | None = None,
    ):
        self.data_loader = data_loader
        self.data_context = DataContext(data_loader)
        self.llm_provider = llm_provider
        
        # Planner uses reasoning model for complex planning
        self.planner_llm = get_fast_llm(provider=llm_provider).with_structured_output(ExecutionPlan)
        
        # Initialize worker agents
        self.agents = {
            "data_explorer": DataExplorerAgent(data_loader, llm_provider),
            "sql_agent": SQLQueryAgent(data_loader, llm_provider),
            "viz_agent": VisualizationAgent(data_loader, llm_provider),
            "analytics_agent": AnalyticsAgent(data_loader, llm_provider),
        }
        
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        
        logger.info(f"SupervisorAgent initialized with Plan-and-Execute pattern")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with plan-and-execute pattern."""
        workflow = StateGraph(AgentState)
        
        # Nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("respond", self._respond_node)
        
        # Entry point
        workflow.set_entry_point("planner")
        
        # Edges
        workflow.add_edge("planner", "executor")
        
        # Conditional edge from executor: continue or respond
        workflow.add_conditional_edges(
            "executor",
            self._should_continue,
            {
                "continue": "executor",
                "respond": "respond",
            }
        )
        
        workflow.add_edge("respond", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _planner_node(self, state: AgentState) -> dict[str, Any]:
        """
        Planner node: Analyzes query and creates execution plan.
        
        The LLM decides:
        - Which agents to use
        - In what order
        - What each step should accomplish
        """
        logger.info(f"Planning execution for: {state.current_query[:80]}...")
        
        # Get compact context for planning
        data_context = self.data_context.get_compact_context()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_PROMPT),
            ("human", "{query}"),
        ])
        
        messages = prompt.format_messages(
            data_context=data_context,
            query=state.current_query,
        )
        
        try:
            plan = self.planner_llm.invoke(messages)
            
            # Log the plan
            logger.info(f"Execution Plan ({len(plan.steps)} steps):")
            for i, step in enumerate(plan.steps, 1):
                logger.info(f"  Step {i}: [{step.agent}] {step.task}")
            logger.info(f"Reasoning: {plan.reasoning[:100]}...")
            
            return {
                "execution_plan": plan,
                "current_step_index": 0,
                "step_results": [],
                "data_schema": data_context,
            }
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback: single-step plan with sql_agent
            fallback_plan = ExecutionPlan(
                steps=[PlanStep(agent="sql_agent", task=state.current_query)],
                reasoning="Fallback plan due to planning error",
                is_simple=True,
            )
            return {
                "execution_plan": fallback_plan,
                "current_step_index": 0,
                "step_results": [],
                "data_schema": data_context,
            }
    
    def _executor_node(self, state: AgentState) -> dict[str, Any]:
        """
        Executor node: Executes the current step in the plan.
        
        Passes context from previous steps to current step.
        """
        plan = state.execution_plan
        step_index = state.current_step_index
        
        if not plan or step_index >= len(plan.steps):
            return {"current_step_index": step_index}
        
        current_step = plan.steps[step_index]
        logger.info(f"Executing Step {step_index + 1}/{len(plan.steps)}: [{current_step.agent}] {current_step.task}")
        
        # Build context for this step (include previous step results if needed)
        step_context = self._build_step_context(state, current_step)
        
        # Execute the appropriate agent
        agent = self.agents.get(current_step.agent)
        if not agent:
            logger.error(f"Unknown agent: {current_step.agent}")
            result = {"success": False, "error": f"Unknown agent: {current_step.agent}"}
        else:
            result = self._execute_agent(agent, current_step.agent, current_step.task, step_context)
        
        # Store result
        new_results = state.step_results + [{
            "step": step_index,
            "agent": current_step.agent,
            "task": current_step.task,
            "result": result,
        }]
        
        logger.info(f"Step {step_index + 1} completed | Success: {result.get('success', False)}")
        
        return {
            "step_results": new_results,
            "current_step_index": step_index + 1,
            "agent_scratchpad": {
                **state.agent_scratchpad,
                f"step_{step_index}_result": result,
            },
        }
    
    def _build_step_context(self, state: AgentState, current_step: PlanStep) -> str:
        """Build context for a step, including previous results if needed."""
        context_parts = [state.data_schema]
        
        # If this step depends on previous results, include them
        if current_step.depends_on_previous and state.step_results:
            last_result = state.step_results[-1]["result"]
            context_parts.append("\n--- PREVIOUS STEP RESULTS ---")
            
            # Include response text
            if "response" in last_result:
                context_parts.append(f"Previous analysis:\n{last_result['response'][:2000]}")
            
            # Include data if available (limited)
            if "data" in last_result and last_result.get("success"):
                data = last_result["data"]
                if isinstance(data, list) and data:
                    context_parts.append(f"Data from previous step ({len(data)} rows available)")
        
        return "\n".join(context_parts)
    
    def _execute_agent(self, agent: Any, agent_name: str, task: str, context: str) -> dict[str, Any]:
        """Execute an agent with the given task and context."""
        try:
            if agent_name == "data_explorer":
                return agent.explore(task, context)
            elif agent_name == "sql_agent":
                return agent.query(task, context)
            elif agent_name == "viz_agent":
                return agent.visualize(task, context)
            elif agent_name == "analytics_agent":
                return agent.analyze(task, context)
            else:
                return {"success": False, "error": f"Unknown agent: {agent_name}"}
        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            return {"success": False, "error": str(e), "response": f"Error in {agent_name}: {str(e)}"}
    
    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue executing steps or respond."""
        plan = state.execution_plan
        
        if not plan:
            return "respond"
        
        if state.current_step_index >= len(plan.steps):
            return "respond"
        
        # Check for errors that should stop execution
        if state.step_results:
            last_result = state.step_results[-1]["result"]
            if not last_result.get("success", True) and last_result.get("error"):
                logger.warning(f"Stopping due to error: {last_result.get('error')}")
                return "respond"
        
        return "continue"
    
    def _respond_node(self, state: AgentState) -> dict[str, Any]:
        """
        Respond node: Combines step results into a concise final response.
        """
        logger.info("Generating final response from step results")
        
        response_parts = []
        all_figures = []
        
        for step_result in state.step_results:
            result = step_result["result"]
            agent = step_result.get("agent", "")
            
            if result.get("response"):
                response_text = result["response"]
                # Skip verbose viz agent responses when we have figures
                if agent == "viz_agent" and result.get("figures"):
                    # Just note that charts were created, actual content is in figures
                    continue
                response_parts.append(response_text)
            
            if result.get("figures"):
                all_figures.extend(result["figures"])
        
        # Combine responses concisely
        if response_parts:
            # Use single newline for tighter formatting
            final_response = "\n\n".join(response_parts)
            # Add note about charts if we have figures
            if all_figures:
                final_response += f"\n\n*{len(all_figures)} chart(s) generated below.*"
        else:
            final_response = "I couldn't complete the analysis. Please try rephrasing."
        
        return {
            "messages": [AIMessage(content=final_response)],
            "agent_scratchpad": {
                **state.agent_scratchpad,
                "final_figures": all_figures,
            },
            "iteration_count": state.iteration_count + 1,
        }
    
    def run(self, query: str, thread_id: str = "default") -> dict[str, Any]:
        """
        Run the supervisor workflow for a query.
        
        Args:
            query: User's query.
            thread_id: Thread ID for conversation persistence.
        
        Returns:
            Dictionary with the response and metadata.
        """
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "current_query": query,
            "data_schema": "",
            "available_datasets": self.data_loader.list_datasets(),
            "active_dataset": self.data_loader.list_datasets()[0] if self.data_loader.list_datasets() else None,
            "execution_plan": None,
            "current_step_index": 0,
            "step_results": [],
            "last_result": None,
            "current_agent": None,
            "agent_scratchpad": {},
            "error_message": None,
            "iteration_count": 0,
            "max_iterations": 10,
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            final_state = self.graph.invoke(initial_state, config)
            
            # Extract response
            response = ""
            if final_state.get("messages"):
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        response = msg.content
                        break
            
            # Extract figures from scratchpad
            scratchpad = final_state.get("agent_scratchpad", {})
            figures = scratchpad.get("final_figures", [])
            
            # Also check step results for figures
            if not figures:
                for step_result in final_state.get("step_results", []):
                    result = step_result.get("result", {})
                    if result.get("figures"):
                        figures.extend(result["figures"])
            
            # Get plan summary
            plan = final_state.get("execution_plan")
            plan_summary = None
            if plan:
                plan_summary = {
                    "steps": [{"agent": s.agent, "task": s.task} for s in plan.steps],
                    "reasoning": plan.reasoning,
                }
            
            logger.info(f"Query completed | Steps: {len(final_state.get('step_results', []))} | Figures: {len(figures)}")
            
            return {
                "success": True,
                "response": response,
                "figures": figures,
                "plan": plan_summary,
                "step_results": final_state.get("step_results", []),
                "scratchpad": scratchpad,
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred: {str(e)}",
                "figures": [],
            }
    
    def get_conversation_history(self, thread_id: str = "default") -> list[dict]:
        """Get conversation history for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.graph.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                return [
                    {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                    for m in messages
                ]
        except Exception:
            pass
        return []
