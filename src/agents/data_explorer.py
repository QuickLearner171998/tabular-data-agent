"""Data Explorer Agent for understanding and profiling datasets."""

from typing import Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.llm.providers import get_fast_llm, LLMProvider
from src.utils.logger import get_logger

logger = get_logger("cpg_agent.data_explorer")
from src.data_loader import DataLoader
from src.tools.pandas_tools import create_pandas_tools
from src.tools.sql_tools import create_sql_tools


DATA_EXPLORER_PROMPT = """You are a Data Analyst examining a new dataset.

Schema: {data_context}

Question: {query}

RESPOND LIKE A DATA ANALYST:
When asked to explore/describe data, provide:

**Dataset at a Glance**
- 📊 **Size**: X rows × Y columns
- 📅 **Time Range**: [if date column exists]
- 💰 **Key Metrics**: [numeric columns like revenue, profit]
- 🏷️ **Dimensions**: [categorical columns like category, region]

**Data Quality**
- Missing values: X columns have nulls
- Unique values: [for key dimensions]

**Ready for Analysis**
- Can answer: [2-3 example questions this data can answer]

Keep it scannable. Use emojis sparingly for visual hierarchy.
Don't list every column - focus on what matters for business analysis."""


class DataExplorerAgent:
    """Agent for exploring and profiling datasets."""
    
    def __init__(
        self, 
        data_loader: DataLoader,
        llm_provider: LLMProvider | None = None,
    ):
        self.data_loader = data_loader
        # Use fast model for data exploration (gpt-4.1-mini / gemini-2.0-flash)
        self.llm = get_fast_llm(provider=llm_provider)
        self.tools = create_pandas_tools(data_loader) + create_sql_tools(data_loader)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", DATA_EXPLORER_PROMPT),
            ("human", "{query}"),
        ])
    
    def explore(self, query: str, data_context: str) -> dict[str, Any]:
        """
        Explore the data based on the user query.
        
        Args:
            query: User's exploration query.
            data_context: Context about available data.
        
        Returns:
            Dictionary with exploration results.
        """
        messages = [
            HumanMessage(content=self.prompt.format(
                data_context=data_context,
                query=query,
            ))
        ]
        
        max_iterations = 5
        iteration = 0
        tool_results = []
        
        logger.debug(f"Starting data exploration for query: {query[:50]}...")
        
        while iteration < max_iterations:
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            if not response.tool_calls:
                break
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                tool_func = next(
                    (t for t in self.tools if t.name == tool_name), 
                    None
                )
                
                if tool_func:
                    result = tool_func.invoke(tool_args)
                    tool_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                    })
                    
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    ))
            
            iteration += 1
        
        final_response = messages[-1].content if messages else "No response generated"
        
        logger.info(f"DataExplorer complete | Iterations: {iteration} | Tools: {len(tool_results)}")
        logger.info(f"DataExplorer output: {final_response[:200]}..." if len(final_response) > 200 else f"DataExplorer output: {final_response}")
        
        return {
            "success": True,
            "response": final_response,
            "tool_results": tool_results,
            "iterations": iteration,
        }
    
    def get_quick_profile(self, dataset_name: str) -> dict[str, Any]:
        """
        Get a quick profile of a dataset.
        
        Args:
            dataset_name: Name of the dataset.
        
        Returns:
            Dictionary with profile information.
        """
        schema = self.data_loader.get_schema(dataset_name)
        sample = self.data_loader.get_sample(dataset_name, n=5)
        
        profile = {
            "name": dataset_name,
            "rows": schema["row_count"],
            "columns": len(schema["columns"]),
            "memory_mb": schema["memory_usage_mb"],
            "column_details": [],
            "sample_data": sample.to_dict("records"),
        }
        
        for col_name, col_info in schema["columns"].items():
            col_detail = {
                "name": col_name,
                "type": col_info["dtype"],
                "nulls": col_info["null_count"],
                "unique": col_info["unique_count"],
            }
            
            if "min" in col_info:
                col_detail["min"] = col_info["min"]
                col_detail["max"] = col_info["max"]
                col_detail["mean"] = col_info["mean"]
            
            if "sample_values" in col_info:
                col_detail["sample_values"] = col_info["sample_values"]
            
            profile["column_details"].append(col_detail)
        
        return profile
