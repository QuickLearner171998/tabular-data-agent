"""Analytics Agent for statistical analysis and advanced insights."""

from typing import Any
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.llm.providers import get_fast_llm, get_reasoning_llm, LLMProvider
from src.utils.logger import get_logger

logger = get_logger("cpg_agent.analytics")
from src.data_loader import DataLoader
from src.tools.stats_tools import create_stats_tools
from src.tools.pandas_tools import create_pandas_tools
from src.tools.sql_tools import create_sql_tools


ANALYTICS_AGENT_PROMPT = """You are a Senior Data Scientist providing analytical insights.

Schema: {data_context}

Question: {query}

RESPOND LIKE A DATA SCIENTIST:

**TL;DR**: [One sentence executive summary]

**Analysis**:
| Metric | Value | Insight |
|--------|-------|---------|
| Key 1  | $X.XM | context |
| Key 2  | X%    | context |

**Key Findings**:
1. [Most important insight with number]
2. [Second insight if relevant]

**Recommendation**: [One actionable point]

ANALYSIS APPROACH:
- For "why" questions → Look for correlations, compare segments
- For "what's happening" → Show trends, top performers
- For "predict/forecast" → Use historical patterns
- For KPIs → Calculate: revenue, profit margin, growth rate, avg transaction

STATISTICAL TOOLS:
- Correlations: identify relationships between metrics
- Outliers: flag unusual values
- Trends: calculate growth rates

SQL RULES (DuckDB) - Date columns are VARCHAR:
- EXTRACT: EXTRACT(MONTH FROM CAST(date AS DATE))
- STRFTIME: STRFTIME(CAST(date AS DATE), '%Y-%m')

Keep numbers prominent. Format currency ($1.2M) and percentages (15.3%) clearly."""

# Keywords that indicate complex reasoning is needed
REASONING_KEYWORDS = [
    "why", "explain", "analyze why", "root cause", "hypothesis",
    "predict", "forecast", "trend analysis", "what if",
    "compare and contrast", "deep dive", "strategic",
    "recommendation", "optimize", "anomaly", "unusual pattern",
    "correlation analysis", "regression", "causation", "summarize", "key insights",
    "analysis", "insights", "recommendation", "optimize", "anomaly", "unusual pattern",
]


def requires_reasoning(query: str) -> bool:
    """Determine if a query requires the reasoning model (o3-mini)."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in REASONING_KEYWORDS)


class AnalyticsAgent:
    """
    Agent for statistical analysis and business insights.
    
    Uses tiered model approach:
        - Fast model (gpt-4.1-mini): Simple stats, aggregations, KPIs
        - Reasoning model (o3-mini): Complex analysis, predictions, root cause
    """
    
    def __init__(
        self,
        data_loader: DataLoader,
        llm_provider: LLMProvider | None = None,
    ):
        self.data_loader = data_loader
        self.llm_provider = llm_provider
        
        # Initialize both tiers of models
        self.fast_llm = get_fast_llm(provider=llm_provider)
        self.reasoning_llm = get_reasoning_llm(provider=llm_provider)
        
        self.tools = (
            create_stats_tools(data_loader) + 
            create_pandas_tools(data_loader) +
            create_sql_tools(data_loader)
        )
        
        # Bind tools to both models
        self.fast_llm_with_tools = self.fast_llm.bind_tools(self.tools)
        self.reasoning_llm_with_tools = self.reasoning_llm.bind_tools(self.tools)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYTICS_AGENT_PROMPT),
            ("human", "{query}"),
        ])
    
    def analyze(self, query: str, data_context: str) -> dict[str, Any]:
        """
        Perform analysis based on the user query.
        
        Automatically selects model tier based on query complexity:
        - Simple queries: Fast model (gpt-4.1-mini / gemini-2.0-flash)
        - Complex reasoning: Reasoning model (o3-mini / gemini-thinking)
        
        Args:
            query: User's analysis question.
            data_context: Context about available data.
        
        Returns:
            Dictionary with analysis results.
        """
        # Select model based on query complexity
        use_reasoning = requires_reasoning(query)
        llm_with_tools = self.reasoning_llm_with_tools if use_reasoning else self.fast_llm_with_tools
        logger.info(f"Using {'reasoning' if use_reasoning else 'fast'} model for analysis")
        
        messages = [
            HumanMessage(content=self.prompt.format(
                data_context=data_context,
                query=query,
            ))
        ]
        
        max_iterations = 7
        iteration = 0
        tool_results = []
        insights = []
        
        while iteration < max_iterations:
            response = llm_with_tools.invoke(messages)
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
                    logger.debug(f"Calling tool: {tool_name} with args: {tool_args}")
                    result = tool_func.invoke(tool_args)
                    tool_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                    })
                    
                    if isinstance(result, dict):
                        if "strong_correlations" in result and result["strong_correlations"]:
                            insights.append({
                                "type": "correlation",
                                "data": result["strong_correlations"],
                            })
                        if "outlier_count" in result and result.get("outlier_count", 0) > 0:
                            insights.append({
                                "type": "outliers",
                                "count": result["outlier_count"],
                                "percentage": result.get("outlier_percentage", 0),
                            })
                    
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    ))
            
            iteration += 1
        
        final_response = messages[-1].content if messages else "No response generated"
        
        logger.info(f"AnalyticsAgent complete | Iterations: {iteration} | Tools: {len(tool_results)} | Insights: {len(insights)}")
        logger.info(f"AnalyticsAgent output: {final_response[:200]}..." if len(final_response) > 200 else f"AnalyticsAgent output: {final_response}")
        
        return {
            "success": True,
            "response": final_response,
            "insights": insights,
            "tool_results": tool_results,
            "iterations": iteration,
            "model_tier": "reasoning" if use_reasoning else "fast",
        }
    
    def calculate_cpg_kpis(self, dataset_name: str) -> dict[str, Any]:
        """
        Calculate common CPG KPIs for a dataset.
        
        Args:
            dataset_name: Name of the dataset.
        
        Returns:
            Dictionary with calculated KPIs.
        """
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            kpis = {}
            
            revenue_cols = [c for c in df.columns if "revenue" in c.lower() or "sales" in c.lower()]
            if revenue_cols:
                kpis["total_revenue"] = float(df[revenue_cols[0]].sum())
                kpis["avg_revenue"] = float(df[revenue_cols[0]].mean())
            
            profit_cols = [c for c in df.columns if "profit" in c.lower() or "margin" in c.lower()]
            if profit_cols:
                kpis["total_profit"] = float(df[profit_cols[0]].sum())
                if revenue_cols:
                    kpis["profit_margin_pct"] = float(
                        df[profit_cols[0]].sum() / df[revenue_cols[0]].sum() * 100
                    )
            
            quantity_cols = [c for c in df.columns if "quantity" in c.lower() or "units" in c.lower()]
            if quantity_cols:
                kpis["total_units"] = int(df[quantity_cols[0]].sum())
                if revenue_cols:
                    kpis["avg_price_per_unit"] = float(
                        df[revenue_cols[0]].sum() / df[quantity_cols[0]].sum()
                    )
            
            kpis["total_transactions"] = len(df)
            
            category_cols = [c for c in df.columns if "category" in c.lower()]
            if category_cols:
                kpis["unique_categories"] = int(df[category_cols[0]].nunique())
            
            product_cols = [c for c in df.columns if "product" in c.lower()]
            if product_cols:
                kpis["unique_products"] = int(df[product_cols[0]].nunique())
            
            return {
                "success": True,
                "dataset": dataset_name,
                "kpis": kpis,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
