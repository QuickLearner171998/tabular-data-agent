"""LangGraph state definitions for the CPG Data Analysis Agent."""

from typing import Annotated, Literal, Any
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class PlanStep(BaseModel):
    """A single step in an execution plan."""
    
    agent: Literal[
        "data_explorer", 
        "sql_agent", 
        "viz_agent", 
        "analytics_agent"
    ] = Field(description="The agent to execute this step")
    
    task: str = Field(description="What this step should accomplish")
    
    depends_on_previous: bool = Field(
        default=False, 
        description="Whether this step needs results from the previous step"
    )


class ExecutionPlan(BaseModel):
    """Plan for executing a complex query."""
    
    steps: list[PlanStep] = Field(
        description="Ordered list of steps to execute",
        default_factory=list
    )
    
    reasoning: str = Field(
        description="Explanation of why this plan was chosen"
    )
    
    is_simple: bool = Field(
        default=True,
        description="True if query can be handled by a single agent"
    )


class RouteDecision(BaseModel):
    """Decision model for routing to specialized agents."""
    
    next_agent: Literal[
        "data_explorer", 
        "sql_agent", 
        "viz_agent", 
        "analytics_agent",
        "summarize_with_viz",  # Combined data prep + visualization
        "respond"
    ] = Field(description="The next agent to route to")
    
    reasoning: str = Field(description="Reasoning for the routing decision")
    
    requires_aggregation: bool = Field(
        default=False,
        description="Whether the query needs data aggregation before visualization"
    )


class AnalysisResult(BaseModel):
    """Result from an analysis operation."""
    
    success: bool = Field(default=True)
    result_type: Literal["dataframe", "plot", "text", "error"] = Field(default="text")
    content: Any = Field(default=None)
    summary: str = Field(default="")
    code_executed: str | None = Field(default=None)


class AgentState(BaseModel):
    """
    The shared state for the multi-agent system.
    
    This state is passed between all agents and contains:
    - Conversation messages
    - Current query and context
    - Execution plan for multi-step queries
    - Analysis results from each agent
    - Data schema information
    """
    
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    
    current_query: str = Field(default="")
    
    data_schema: str = Field(default="")
    
    available_datasets: list[str] = Field(default_factory=list)
    
    active_dataset: str | None = Field(default=None)
    
    # Plan-and-execute fields
    execution_plan: ExecutionPlan | None = Field(default=None)
    current_step_index: int = Field(default=0)
    step_results: list[dict[str, Any]] = Field(default_factory=list)
    
    last_result: AnalysisResult | None = Field(default=None)
    
    current_agent: str | None = Field(default=None)
    
    agent_scratchpad: dict[str, Any] = Field(default_factory=dict)
    
    error_message: str | None = Field(default=None)
    
    iteration_count: int = Field(default=0)
    max_iterations: int = Field(default=10)
    
    class Config:
        arbitrary_types_allowed = True


def create_initial_state(
    query: str,
    data_schema: str = "",
    available_datasets: list[str] | None = None,
    active_dataset: str | None = None,
) -> dict:
    """
    Create an initial state dictionary for the agent graph.
    
    Args:
        query: The user's query.
        data_schema: String description of available data schema.
        available_datasets: List of available dataset names.
        active_dataset: Currently active dataset name.
    
    Returns:
        A dictionary suitable for initializing AgentState.
    """
    return {
        "messages": [],
        "current_query": query,
        "data_schema": data_schema,
        "available_datasets": available_datasets or [],
        "active_dataset": active_dataset,
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
