"""Visualization Agent for creating charts and visual insights."""

from typing import Any
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from src.llm.providers import get_fast_llm, LLMProvider
from src.utils.logger import get_logger
from src.data_loader import DataLoader
from src.tools.viz_tools import create_viz_tools
from src.tools.sql_tools import create_sql_tools

logger = get_logger("cpg_agent.viz_agent")


VIZ_AGENT_PROMPT = """You are a visualization expert. Create charts that answer the question directly.

Context: {data_context}

Task: {query}

CRITICAL: Visualize ONLY what answers the question.
- Asked about "months with >20% decline"? → Show ONLY those months, not all data
- Asked about "top 5"? → Show exactly 5 bars, not 20
- Highlight the answer visually (color, annotation)

IMPORTANT: After SQL queries run, results are saved as "_last_query_result" dataset.
Use dataset_name="_last_query_result" to create visualizations from query results.

CHART SELECTION:
| Question Type | Chart | Notes |
|---------------|-------|-------|
| Rankings/Top N | Horizontal Bar | Sort descending, limit to N |
| Trends over time | Line | Highlight anomalies |
| Proportions | Donut | Max 5-6 slices |
| Comparisons | Grouped Bar | Side by side |

EXECUTION:
1. Query ONLY the data needed to answer the question
2. Filter to relevant subset (not full dataset)
3. Create ONE chart that directly answers the question
4. Use title that states the insight: "January 2024: Largest Revenue Drop (-35.6%)"

SQL RULES (DuckDB):
- Date columns are VARCHAR: CAST(date AS DATE)
- STRFTIME(CAST(date AS DATE), '%Y-%m') for month grouping

NO chart is better than a chart that doesn't answer the question."""


class VisualizationAgent:
    """
    Agent for creating data visualizations.
    
    Works best when:
    - Previous step has already aggregated data (from sql_agent)
    - Or for simple visualizations on small datasets
    """
    
    def __init__(
        self,
        data_loader: DataLoader,
        llm_provider: LLMProvider | None = None,
    ):
        self.data_loader = data_loader
        self.llm = get_fast_llm(provider=llm_provider)
        # Include both viz and sql tools (sql for data prep if needed)
        self.tools = create_viz_tools(data_loader) + create_sql_tools(data_loader)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", VIZ_AGENT_PROMPT),
            ("human", "{query}"),
        ])
        
        logger.info("VisualizationAgent initialized")
    
    def visualize(self, query: str, data_context: str) -> dict[str, Any]:
        """
        Create visualization(s) based on the user query.
        
        Args:
            query: User's visualization request or task description.
            data_context: Context about available data or previous step results.
        
        Returns:
            Dictionary with visualization results.
        """
        logger.info(f"VisualizationAgent processing: {query[:60]}...")
        
        messages = [
            HumanMessage(content=self.prompt.format(
                data_context=data_context[:4000],  # Limit context to avoid overflow
                query=query,
            ))
        ]
        
        max_iterations = 5
        iteration = 0
        tool_results = []
        figures = []
        
        while iteration < max_iterations:
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            if not response.tool_calls:
                logger.debug(f"No more tool calls after iteration {iteration}")
                break
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.debug(f"Calling tool: {tool_name}")
                
                tool_func = next(
                    (t for t in self.tools if t.name == tool_name),
                    None
                )
                
                if tool_func:
                    try:
                        result = tool_func.invoke(tool_args)
                        tool_results.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result": result,
                        })
                        
                        # Collect figures
                        if isinstance(result, dict) and "figure" in result:
                            figures.append({
                                "type": result.get("chart_type", "unknown"),
                                "figure_json": result["figure"],
                                "summary": result.get("summary", ""),
                            })
                            logger.info(f"Created {result.get('chart_type', 'unknown')} chart")
                        
                        # Limit result size for context
                        result_str = str(result)
                        if len(result_str) > 1500:
                            result_str = result_str[:1500] + "... [truncated]"
                        
                        messages.append(ToolMessage(
                            content=result_str,
                            tool_call_id=tool_call["id"],
                        ))
                    except Exception as e:
                        logger.warning(f"Tool {tool_name} failed: {e}")
                        messages.append(ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call["id"],
                        ))
                else:
                    logger.warning(f"Tool not found: {tool_name}")
            
            iteration += 1
        
        final_response = messages[-1].content if hasattr(messages[-1], 'content') else "Visualization created."
        
        logger.info(f"VizAgent complete | Iterations: {iteration} | Charts: {len(figures)}")
        if figures:
            chart_types = [f.get("type", "unknown") for f in figures]
            logger.info(f"VizAgent charts created: {chart_types}")
        
        return {
            "success": True,
            "response": final_response,
            "figures": figures,
            "tool_results": tool_results,
            "iterations": iteration,
        }
    
    def suggest_visualizations(self, dataset_name: str) -> list[dict[str, str]]:
        """
        Suggest appropriate visualizations for a dataset based on its schema.
        
        Args:
            dataset_name: Name of the dataset.
        
        Returns:
            List of visualization suggestions.
        """
        schema = self.data_loader.get_schema(dataset_name)
        suggestions = []
        
        # Categorize columns
        numeric_cols = [
            col for col, info in schema["columns"].items()
            if "int" in info["dtype"] or "float" in info["dtype"]
        ]
        
        categorical_cols = [
            col for col, info in schema["columns"].items()
            if info["dtype"] == "object" and info["unique_count"] < 50
        ]
        
        date_cols = [
            col for col, info in schema["columns"].items()
            if "date" in col.lower() or "time" in col.lower()
        ]
        
        # Generate suggestions based on column types
        for num_col in numeric_cols[:3]:
            suggestions.append({
                "type": "histogram",
                "description": f"Distribution of {num_col}",
                "columns": [num_col],
            })
        
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "scatter",
                "description": f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                "columns": numeric_cols[:2],
            })
            
            suggestions.append({
                "type": "heatmap",
                "description": "Correlation between numeric variables",
                "columns": numeric_cols,
            })
        
        for cat_col in categorical_cols[:2]:
            if numeric_cols:
                suggestions.append({
                    "type": "bar",
                    "description": f"{numeric_cols[0]} by {cat_col}",
                    "columns": [cat_col, numeric_cols[0]],
                })
        
        if date_cols and numeric_cols:
            suggestions.append({
                "type": "line",
                "description": f"{numeric_cols[0]} trend over {date_cols[0]}",
                "columns": [date_cols[0], numeric_cols[0]],
            })
        
        return suggestions
