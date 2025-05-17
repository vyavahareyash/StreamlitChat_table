import pandas as pd
from datetime import datetime

def process_user_question(question: str) -> dict:
    """
    Process the user's question and return a response.
    This is a mock implementation - replace with actual backend logic.
    """
    # Mock response (replace with your actual backend call)
    answer = f"This is a sample answer to your question: '{question}'"
    sample_prompt = "Answer the question based on the database"
    sample_query = "SELECT * FROM products WHERE category = 'electronics'"
    sample_data = pd.DataFrame({
        'Product': ['Laptop', 'Phone', 'Tablet'],
        'Sales': [1200, 3500, 800]
    })
    
    return {
        "role": "assistant",
        "content": answer,
        "prompt": sample_prompt,
        "query": sample_query,
        "data": sample_data,
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