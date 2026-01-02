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


VIZ_AGENT_PROMPT = """You are a Data Visualization Specialist. Create charts that tell a story.

Context: {data_context}

Task: {query}

IMPORTANT: After SQL queries run, results are saved as "_last_query_result" dataset.
Use this dataset name to create visualizations from query results.

CHOOSE THE RIGHT CHART:
| Data Type | Best Chart | When to Use |
|-----------|------------|-------------|
| Rankings/Comparisons | Horizontal Bar | "Top 10 by...", "Compare X vs Y" |
| Trends | Line | Data over time (dates on x-axis) |
| Proportions | Donut | Parts of whole, <6 categories |
| Distribution | Histogram | "How is X distributed?" |
| Correlation | Scatter | "Relationship between X and Y" |

DECISION RULES:
1. Rankings → Horizontal bar chart (sorted)
2. Time series → Line chart 
3. Category breakdown → Bar chart
4. Proportions (<6 items) → Pie/Donut

EXECUTION:
1. If previous step ran SQL, use dataset_name="_last_query_result"
2. Otherwise, query the main dataset first
3. Create ONE focused chart
4. Give 1-line insight after creating chart

SQL RULES (if querying data) - Date columns are VARCHAR:
- EXTRACT: EXTRACT(MONTH FROM CAST(date AS DATE))
- STRFTIME: STRFTIME(CAST(date AS DATE), '%Y-%m')

Example: create_bar_chart(dataset_name="_last_query_result", x_column="category", y_column="total_revenue", title="Revenue by Category")

Quality over quantity. One great chart beats three mediocre ones."""


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
