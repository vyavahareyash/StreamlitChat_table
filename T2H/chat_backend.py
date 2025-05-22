import pandas as pd
from datetime import datetime
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field

import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "llama3.2"  # You can change to your preferred model
TEMPERATURE = 0.0

# Initialize LLM
llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)


# Load Schema Information
def load_schema(schema_path: str) -> Dict[str, Any]:
    """Load the schema information from a JSON file."""
    try:
        with open(schema_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load schema from {schema_path}: {str(e)}")

# Helper function to load CSV data
def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV data into a pandas DataFrame."""
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to load CSV from {csv_path}: {str(e)}")

# Define state schema
class AgentState(BaseModel):
    """State for the multi-agent table data analyzer system."""
    user_query: str = Field(description="The original user query in natural language")
    sql_query: Optional[str] = Field(default=None, description="Generated SQL query")
    query_results: Optional[Any] = Field(default=None, description="Results from executing SQL query")
    final_response: Optional[str] = Field(default=None, description="Final response to user query")
    error: Optional[str] = Field(default=None, description="Error message if something goes wrong")

def sql_generation_agent(state: AgentState) -> AgentState:
    """Generates SQL query based on the query understanding and schema information."""
    if state.error:
        # print("error")
        return state
    
    schema = str(load_schema("data/schema.json")).replace('{','{{').replace('}','}}')
    
    prompt = ChatPromptTemplate([
        ("system", """You are a SQL Generation Agent specialized in converting natural language queries into precise, executable SQL queries.
        
        INSTRUCTIONS:
        1. Analyze the query understanding and schema information carefully
        2. Generate a correct SQL query that will retrieve exactly the data needed to answer the user's question
        3. The SQL query should be executable using the pandasql library on a pandas DataFrame
        
        
        IMPORTANT REQUIREMENTS:
        - Your query must reference columns EXACTLY as they appear in the schema
        - Use appropriate SQL functions (AVG, SUM, COUNT, etc.) based on the query intent
        - Include proper WHERE clauses for any filtering conditions
        - Use GROUP BY for aggregation queries
        - Use ORDER BY for sorting results
        - Ensure all table and column names are correctly quoted if they contain spaces or special characters
        - Double-check that your query syntax is valid for pandasql
        
        Your output should contain ONLY the SQL query without any explanation or comment. Do not include markdown code blocks or any other text."""),
        ("human", f"""Schema information: {schema}
        
        Generate a SQL query to answer the original question: "{state.user_query}".""")
    ])
    
    try:
        response = llm.invoke(prompt.format())

        sql_query = response.content.strip()
        
        return AgentState(**{**state.model_dump(), "sql_query": sql_query})
    except Exception as e:
        return AgentState(**{**state.model_dump(), "error": f"Query execution failed: {str(e)}"})


def execution_agent(state: AgentState) -> AgentState:
    """Executes the SQL query on the CSV data."""
    if state.error:
        print("error")
        return state

    try:
        # Import here to avoid issues if package is missing
        from pandasql import sqldf
        
        # Load the CSV data
        SKA1 = load_csv_data("data/SKA1.csv")
        
        # Pre-process the SQL query to handle potential issues
        sql_query = state.sql_query
        
        local_env = {"SKA1": SKA1}
        
        query_results = sqldf(sql_query, local_env)
        
        if query_results is None or query_results.empty:
            # Handle empty results - this might be valid in some cases
            results_json =[ ]
        else:
            # Convert results to JSON for easier serialization
            results_json = json.loads(query_results.to_json(orient="records"))
        
        
        return AgentState(**{**state.model_dump(), "query_results": results_json})
    except Exception as e:
        return AgentState(**{**state.model_dump(), "error": f"Query execution failed: {str(e)}"})


def response_generation_agent(state: AgentState) -> AgentState:
    """Generates a natural language response based on the query results."""
    if state.error:
        return state

    # Prepare context about the number of results
    result_count = len(state.query_results) if state.query_results else 0
    result_summary = f"Query returned {result_count} results."
    
    # For very large result sets, limit what we send to the LLM
    display_results = state.query_results
    if result_count > 20:
        display_results = state.query_results[:20]
        result_summary += f" Showing the first 20 for analysis."
    
    # Include result statistics for numeric data
    result_stats = ""
    if result_count > 0 and isinstance(state.query_results, list) and len(state.query_results) > 0:
        # Try to compute simple statistics for numeric fields
        try:
            import pandas as pd
            result_df = pd.DataFrame(state.query_results)
            
            # Find numeric columns
            numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                stats = {}
                for col in numeric_cols:
                    col_stats = {
                        "mean": result_df[col].mean() if not result_df[col].empty else "N/A",
                        "min": result_df[col].min() if not result_df[col].empty else "N/A",
                        "max": result_df[col].max() if not result_df[col].empty else "N/A"
                    }
                    stats[col] = col_stats
                
                result_stats = f"Statistical summary of numeric fields: {json.dumps(stats)}"
        except Exception:
            # If statistics calculation fails, just proceed without it
            pass
    
    display_results = str(display_results).replace('{','{{').replace('}','}}')
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Response Generation Agent specialized in converting data query results into clear, insightful natural language answers.
        
        INSTRUCTIONS:
        1. Review the original user query carefully
        2. Analyze the provided SQL query and its results
        3. Generate a comprehensive yet concise response that directly answers the user's question
        
        Your response should:
        - Start by directly answering the question
        - Include specific data points and numbers from the results
        - Explain any trends, patterns, or notable findings
        - Be conversational and easy to understand
        - Include relevant summary statistics when appropriate
        - Provide context for understanding the results
        
        IMPORTANT:
        - If the result set is empty, explain what that means in the context of the question
        - Be precise with numbers and units
        - Avoid technical jargon unless necessary
        - Don't reference the SQL query in your response unless directly relevant to the user's understanding
        - If possible keep it to one sentence"""),
        ("user", f"""Original user query: "{state.user_query}"
        
        SQL query executed: "{state.sql_query}"
        
        Query results: {display_results}
        
        
        {result_summary}
        
        {result_stats}
        
        Generate a natural language response that answers the original query.""")
    ])
    try:
        response = llm.invoke(prompt.format())
        final_response = response.content
        
        return AgentState(**{**state.model_dump(), "final_response": final_response})
    except Exception as e:
        return AgentState(**{**state.model_dump(), "error": f"Response generation failed: {str(e)}"})


def handle_error(state: AgentState) -> str:
    """Determines if there's an error in the state."""
    if state.error:
        return "error"
    return "continue"

def build_graph() -> StateGraph:
    """Build the LangGraph state graph for the CSV data analyzer."""
    graph = StateGraph(AgentState)
    
    graph.add_node("sql_generation", sql_generation_agent)
    graph.add_node("execution", execution_agent)
    graph.add_node("response_generation", response_generation_agent)
    graph.add_node("error_node", lambda state: AgentState(**{**state.model_dump(), "final_response": f"I encountered an error: {state.error}"}))
    
    graph.add_edge(START, "sql_generation")
    graph.add_edge("sql_generation", "execution")
    graph.add_edge("execution", "response_generation")
    graph.add_edge("response_generation", END)
    
    # Add error handling edges
    graph.add_conditional_edges(
        "sql_generation",
        handle_error,
        {
            "error": "error_node",
            "continue": "execution"
        }
    )
    graph.add_conditional_edges(
        "execution",
        handle_error,
        {
            "error": "error_node",
            "continue": "response_generation"
        }
    )
    graph.add_conditional_edges(
        "response_generation",
        handle_error,
        {
            "error": "error_node",
            "continue": END
        }
    )
    
    graph.add_edge("error_node", END)
    
    # Set the entry point
    graph.set_entry_point("sql_generation")
    
    return graph


def process_query(query: str) -> str:
    """Process a natural language query about CSV data and return the answer."""
    graph = build_graph()
    
    # Compile the graph
    app = graph.compile()
    
    # from IPython.display import Image
    # Image(app.get_graph().draw_png())
    
    # Run the graph
    inputs = AgentState(user_query=query)
    result = app.invoke(inputs)

    if result['error']:
        return f"Error: {result['error']}"
    
    return result

def process_user_question(question: str) -> dict:
    """
    Process the user's question and return a response.
    This is a mock implementation - replace with actual backend logic.
    """
    # Mock response (replace with your actual backend call)
    result = process_query(question)
    
    answer = result['final_response']
    sample_prompt = "Answer the question based on the database"
    query = result['sql_query']
    data = pd.DataFrame(result['query_results'])
    
    return {
        "role": "assistant",
        "content": answer,
        "prompt": sample_prompt,
        "query": query,
        "data": data,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

def create_user_message(question: str) -> dict:
    """
    Create a user message object.
    """
    return {
        "role": "user",
        "content": question,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    } 