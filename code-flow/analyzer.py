import os
import git
import shutil
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from fnmatch import fnmatch
import openai
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GitConfig:
    """Configuration for Git repository access"""
    repo_url: str
    username: str
    personal_access_token: str
    local_path: str = "./temp_repo"

@dataclass
class AnalysisConfig:
    """Configuration for code analysis"""
    openai_api_key: str
    model: str = "gpt-4"
    max_file_size: int = 50000  # Skip files larger than 50KB
    supported_extensions: List[str] = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = [
                '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
                '.cs', '.php', '.rb', '.go', '.rs', '.kt', '.swift', '.scala',
                '.vue', '.html', '.css', '.scss', '.less', '.sql', '.sh', '.yaml', '.yml'
            ]

class GitIgnoreParser:
    """Parser for .gitignore patterns"""
    
    def __init__(self, gitignore_path: str):
        self.patterns = []
        self.load_gitignore(gitignore_path)
    
    def load_gitignore(self, gitignore_path: str):
        """Load patterns from .gitignore file"""
        if not os.path.exists(gitignore_path):
            return
        
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    self.patterns.append(line)
    
    def should_ignore(self, file_path: str, repo_root: str) -> bool:
        """Check if file should be ignored based on .gitignore patterns"""
        relative_path = os.path.relpath(file_path, repo_root)
        
        for pattern in self.patterns:
            if fnmatch(relative_path, pattern) or fnmatch(os.path.basename(relative_path), pattern):
                return True
        return False

class GitRepoFetcher:
    """Handles Git repository cloning and file extraction"""
    
    def __init__(self, config: GitConfig):
        self.config = config
        self.repo = None
    
    def clone_repository(self) -> bool:
        """Clone the Git repository with authentication"""
        try:
            # Clean up existing directory
            if os.path.exists(self.config.local_path):
                shutil.rmtree(self.config.local_path)
            
            # Create authenticated URL
            if self.config.repo_url.startswith('https://github.com/'):
                auth_url = self.config.repo_url.replace(
                    'https://github.com/',
                    f'https://{self.config.username}:{self.config.personal_access_token}@github.com/'
                )
            else:
                # Generic HTTPS authentication
                auth_url = self.config.repo_url.replace('https://', f'https://{self.config.username}:{self.config.personal_access_token}@')
            
            logger.info(f"Cloning repository to {self.config.local_path}")
            self.repo = git.Repo.clone_from(auth_url, self.config.local_path)
            logger.info("Repository cloned successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clone repository: {str(e)}")
            return False
    
    def get_file_paths(self, analysis_config: AnalysisConfig) -> List[str]:
        """Get list of relevant files respecting .gitignore"""
        if not self.repo:
            return []
        
        gitignore_parser = GitIgnoreParser(os.path.join(self.config.local_path, '.gitignore'))
        file_paths = []
        
        for root, dirs, files in os.walk(self.config.local_path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if file should be ignored
                if gitignore_parser.should_ignore(file_path, self.config.local_path):
                    continue
                
                # Check file extension
                if any(file.endswith(ext) for ext in analysis_config.supported_extensions):
                    # Check file size
                    if os.path.getsize(file_path) <= analysis_config.max_file_size:
                        file_paths.append(file_path)
        
        logger.info(f"Found {len(file_paths)} files to analyze")
        return file_paths
    
    def cleanup(self):
        """Clean up cloned repository"""
        if os.path.exists(self.config.local_path):
            shutil.rmtree(self.config.local_path)
            logger.info("Cleaned up temporary repository")

class CodeAnalyzer:
    """Analyzes code files and generates understanding"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        
        self.code_understanding_prompt = """
You are a senior software architect specializing in code analysis and documentation. Your role is to read codebase files and generate comprehensive, structured code understanding.

For the given file, provide the following structured analysis:

## File: {filename}
### Purpose & Responsibility
- Core functionality and role in the system
- Business logic or technical purpose

### Code Structure
- Classes/functions/modules defined
- Main entry points and public interfaces
- Key algorithms or logic patterns

### Dependencies & Imports
- External libraries imported
- Internal modules/files referenced
- Database connections or external services

### Data Flow
- Input parameters and their sources
- Data transformations performed
- Output/return values and their destinations
- Side effects (file writes, API calls, state changes)

### Relationships & Interactions
- Which components this file interacts with
- How it's called by other parts of the system
- What it calls or depends on
- Event handling or callback patterns

### Configuration & Environment
- Environment variables used
- Configuration files or settings referenced
- Runtime dependencies

### Error Handling & Edge Cases
- Exception handling patterns
- Validation logic
- Fallback mechanisms

Analyze this code file:
```
{code_content}
```
"""
    
    def analyze_file(self, file_path: str) -> Dict[str, str]:
        """Analyze a single file and return structured understanding"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            relative_path = file_path.replace(os.getcwd(), '').lstrip('/')
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "user", "content": self.code_understanding_prompt.format(
                        filename=relative_path,
                        code_content=content
                    )}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            logger.info(f"Analyzed file: {relative_path}")
            
            return {
                "file_path": relative_path,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {str(e)}")
            return {
                "file_path": file_path,
                "analysis": f"Error analyzing file: {str(e)}"
            }
    
    def combine_analyses(self, file_analyses: List[Dict[str, str]]) -> str:
        """Combine individual file analyses into system-level understanding"""
        combined_analysis = "# Complete System Code Understanding\n\n"
        
        # Add individual file analyses
        for analysis in file_analyses:
            combined_analysis += f"{analysis['analysis']}\n\n---\n\n"
        
        # Generate system-level synthesis
        synthesis_prompt = f"""
Based on the following individual file analyses, provide a system-level synthesis:

## Overall System Architecture

### Core Components
- List major system components and their roles
- Identify service layers (presentation, business, data)

### Inter-Component Dependencies
- Map how components depend on each other
- Identify circular dependencies or coupling issues
- Note communication patterns (sync/async, events, direct calls)

### Data Flow Patterns
- Trace major data pipelines through the system
- Identify data storage and retrieval patterns
- Note data transformation points

### External Integrations
- APIs, databases, file systems accessed
- Third-party services and libraries
- Configuration and deployment dependencies

### Architecture Patterns
- Identify design patterns used (MVC, microservices, event-driven, etc.)
- Note architectural styles and conventions

File Analyses:
{combined_analysis}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=3000,
                temperature=0.3
            )
            
            synthesis = response.choices[0].message.content
            combined_analysis += f"\n\n# SYSTEM-LEVEL SYNTHESIS\n\n{synthesis}"
            
        except Exception as e:
            logger.error(f"Failed to generate system synthesis: {str(e)}")
            combined_analysis += f"\n\n# SYSTEM-LEVEL SYNTHESIS\nError generating synthesis: {str(e)}"
        
        return combined_analysis

class MermaidGenerator:
    """Generates Mermaid diagrams from code understanding"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        
        self.mermaid_prompt = """
You are a technical documentation specialist who creates clear, comprehensive Mermaid diagrams from code analysis. Transform the provided code analysis into visual architectural representations.

Generate the following Mermaid diagrams:

1. **System Overview** - High-level system components and relationships
2. **Component Dependencies** - File-to-file dependencies and imports
3. **Data Flow Diagram** - How data moves through the system

Guidelines:
- Use meaningful node labels (not just filenames)
- Apply appropriate node shapes and colors
- Show both dependencies and data flow
- Use subgraphs to organize related components
- Include proper styling for visual clarity

For each diagram, provide:
1. Diagram title and purpose
2. Mermaid code (properly formatted)
3. Key insights about the architecture

Code Analysis:
{code_understanding}
"""
    
    def generate_mermaid_diagrams(self, code_understanding: str) -> str:
        """Generate Mermaid diagrams from code understanding"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": self.mermaid_prompt.format(
                    code_understanding=code_understanding
                )}],
                max_tokens=4000,
                temperature=0.3
            )
            
            mermaid_content = response.choices[0].message.content
            logger.info("Generated Mermaid diagrams successfully")
            return mermaid_content
            
        except Exception as e:
            logger.error(f"Failed to generate Mermaid diagrams: {str(e)}")
            return f"Error generating Mermaid diagrams: {str(e)}"

class CodeAnalysisSystem:
    """Main system orchestrator"""
    
    def __init__(self, git_config: GitConfig, analysis_config: AnalysisConfig):
        self.git_config = git_config
        self.analysis_config = analysis_config
        self.git_fetcher = GitRepoFetcher(git_config)
        self.code_analyzer = CodeAnalyzer(analysis_config)
        self.mermaid_generator = MermaidGenerator(analysis_config)
    
    def run_analysis(self, output_dir: str = "./output") -> bool:
        """Run the complete analysis pipeline"""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Clone repository
            logger.info("Step 1: Cloning repository...")
            if not self.git_fetcher.clone_repository():
                return False
            
            # Step 2: Get file paths
            logger.info("Step 2: Getting file paths...")
            file_paths = self.git_fetcher.get_file_paths(self.analysis_config)
            if not file_paths:
                logger.warning("No files found to analyze")
                return False
            
            # Step 3: Analyze individual files
            logger.info("Step 3: Analyzing individual files...")
            file_analyses = []
            for i, file_path in enumerate(file_paths, 1):
                logger.info(f"Analyzing file {i}/{len(file_paths)}: {file_path}")
                analysis = self.code_analyzer.analyze_file(file_path)
                file_analyses.append(analysis)
            
            # Step 4: Combine analyses
            logger.info("Step 4: Combining analyses...")
            combined_understanding = self.code_analyzer.combine_analyses(file_analyses)
            
            # Save code understanding
            understanding_file = os.path.join(output_dir, "code_understanding.md")
            with open(understanding_file, 'w', encoding='utf-8') as f:
                f.write(combined_understanding)
            logger.info(f"Code understanding saved to: {understanding_file}")
            
            # Step 5: Generate Mermaid diagrams
            logger.info("Step 5: Generating Mermaid diagrams...")
            mermaid_content = self.mermaid_generator.generate_mermaid_diagrams(combined_understanding)
            
            # Save Mermaid diagrams
            mermaid_file = os.path.join(output_dir, "architecture_diagrams.md")
            with open(mermaid_file, 'w', encoding='utf-8') as f:
                f.write(mermaid_content)
            logger.info(f"Mermaid diagrams saved to: {mermaid_file}")
            
            # Step 6: Save metadata
            metadata = {
                "repository_url": self.git_config.repo_url,
                "files_analyzed": len(file_paths),
                "analysis_timestamp": str(pd.Timestamp.now()),
                "model_used": self.analysis_config.model
            }
            
            metadata_file = os.path.join(output_dir, "analysis_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Analysis completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return False
        
        finally:
            # Cleanup
            self.git_fetcher.cleanup()

def main():
    """Example usage"""
    
    # Configuration
    git_config = GitConfig(
        repo_url="https://github.com/username/repository.git",
        username="your_github_username",
        personal_access_token="your_personal_access_token"
    )
    
    analysis_config = AnalysisConfig(
        openai_api_key="your_openai_api_key",
        model="gpt-4",
        max_file_size=50000
    )
    
    # Run analysis
    system = CodeAnalysisSystem(git_config, analysis_config)
    success = system.run_analysis("./analysis_output")
    
    if success:
        print("Analysis completed successfully!")
        print("Check the ./analysis_output directory for results:")
        print("- code_understanding.md: Detailed code analysis")
        print("- architecture_diagrams.md: Mermaid diagrams")
        print("- analysis_metadata.json: Analysis metadata")
    else:
        print("Analysis failed. Check the logs for details.")

if __name__ == "__main__":
    main()