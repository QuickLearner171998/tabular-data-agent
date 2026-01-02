"""SQL Query Agent for text-to-SQL generation and data querying."""

from typing import Any
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.llm.providers import get_fast_llm, LLMProvider
from src.utils.logger import get_logger

logger = get_logger("cpg_agent.sql_agent")
from src.data_loader import DataLoader
from src.tools.sql_tools import create_sql_tools


SQL_AGENT_PROMPT = """You are an expert data consultant. Answer questions directly and concisely.

Schema: {data_context}

Question: {query}

RESPONSE FORMAT - BE DIRECT:

**Answer:** [Direct answer to the question in 1-2 sentences with specific numbers]

**Why:** [Brief explanation if asked, otherwise skip]

| Only | Relevant | Columns |
|------|----------|---------|  
[Show ONLY rows that answer the question - not full dataset]

CRITICAL RULES:
1. ANSWER THE EXACT QUESTION FIRST - don't show unrelated data
2. Filter results to ONLY what was asked (e.g., "months with >20% decline" = show ONLY those months)
3. If asked "which/what/how many" → give the specific answer, not a data dump
4. Maximum 5-10 rows unless explicitly asked for more
5. No filler text or excessive context

SQL RULES (DuckDB):
- Date columns are VARCHAR: CAST(date AS DATE) before operations
- STRFTIME(CAST(date AS DATE), '%Y-%m') for month grouping
- Always LIMIT results appropriately
- ROUND numbers: ROUND(value, 2)

FORMATTING:
- Currency: $1.2M, $450K  
- Percentages: 15.3%
- Bold the answer: **January 2024 (-35.6%)**

Think like a consultant: client asks a question, give them the answer - not a research paper."""


class SQLQueryAgent:
    """Agent for generating and executing SQL queries."""
    
    def __init__(
        self,
        data_loader: DataLoader,
        llm_provider: LLMProvider | None = None,
    ):
        self.data_loader = data_loader
        # Use fast model for SQL generation (gpt-4.1-mini / gemini-2.0-flash)
        self.llm = get_fast_llm(provider=llm_provider)
        self.tools = create_sql_tools(data_loader)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SQL_AGENT_PROMPT),
            ("human", "{query}"),
        ])
    
    def query(self, query: str, data_context: str) -> dict[str, Any]:
        """
        Generate and execute a SQL query based on natural language.
        
        Args:
            query: User's natural language query.
            data_context: Context about available data.
        
        Returns:
            Dictionary with query results.
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
        executed_queries = []
        
        logger.info(f"SQLAgent starting | Query: {query[:80]}...")
        
        while iteration < max_iterations:
            logger.info(f"SQLAgent iteration {iteration + 1}/{max_iterations} | Calling LLM...")
            
            try:
                response = self.llm_with_tools.invoke(messages)
                messages.append(response)
                logger.debug(f"LLM response received | Tool calls: {len(response.tool_calls) if response.tool_calls else 0}")
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                break
            
            if not response.tool_calls:
                logger.info(f"SQLAgent complete after {iteration + 1} iterations (no more tool calls)")
                break
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"SQLAgent calling tool: {tool_name}")
                
                if tool_name == "execute_sql_query" and "query" in tool_args:
                    executed_queries.append(tool_args["query"])
                    logger.debug(f"SQL: {tool_args['query'][:100]}...")
                
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
        
        logger.info(f"SQLAgent complete | Iterations: {iteration} | Queries: {len(executed_queries)}")
        logger.info(f"SQLAgent output: {final_response[:200]}..." if len(final_response) > 200 else f"SQLAgent output: {final_response}")
        
        return {
            "success": True,
            "response": final_response,
            "executed_queries": executed_queries,
            "tool_results": tool_results,
            "iterations": iteration,
        }
    
    def execute_direct(self, sql_query: str) -> dict[str, Any]:
        """
        Execute a SQL query directly without LLM.
        
        Args:
            sql_query: SQL query to execute.
        
        Returns:
            Dictionary with query results.
        """
        try:
            df = self.data_loader.execute_sql(sql_query)
            return {
                "success": True,
                "data": df.head(100).to_dict("records"),
                "columns": list(df.columns),
                "row_count": len(df),
                "query": sql_query,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": sql_query,
            }
