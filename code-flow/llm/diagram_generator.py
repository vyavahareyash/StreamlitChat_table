import openai
from typing import List, Dict
import os

class DiagramGenerator:
    def __init__(self, azure_endpoint: str, api_key: str, deployment_name: str):
        openai.api_type = "azure"
        openai.api_base = azure_endpoint
        openai.api_key = api_key
        openai.api_version = "2023-05-15"  # Replace if using newer version
        self.deployment_name = deployment_name

    def generate_mermaid(self, chunks: List[str], diagram_type: str = "class") -> str:
        full_diagram_code = ""

        for i, chunk in enumerate(chunks):
            prompt = f"""
You are an expert developer. Analyze the following code and generate a Mermaid {diagram_type} diagram 
to represent relationships, flow, or structure.

Respond ONLY with valid mermaid code.

### CODE CHUNK {i+1} ###
{chunk}
"""
            try:
                response = openai.ChatCompletion.create(
                    engine=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You generate architecture diagrams from code."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )
                mermaid = response["choices"][0]["message"]["content"]
                full_diagram_code += f"\n\n%% CHUNK {i+1} %%\n" + mermaid
            except Exception as e:
                print(f"Error in chunk {i+1}: {e}")

        return full_diagram_code
