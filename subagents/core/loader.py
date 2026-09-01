from pathlib import Path
import yaml

from subagents.core.types import AgentDefinition

def load_agent_definition(path: str | Path) -> AgentDefinition:
    
    """
    Load an AgentDefinition from a Markdown file
    containing YAML front matter.
    """
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Agent definition file not found: {path}"
        )
        
    content = path.read_text(encoding="utf-8")
    
    if not content.startswith("---"):
        raise ValueError(
            f"Agent definition file must start with YAML front matter: {path}"
        )
        
    parts = content.split("---", 2)
    
    if len(parts) < 3:
        raise ValueError(
            f"Agent definition '{path}' has invalid YAML front matter."
        )
        
    metadata_text = parts[1]
    system_prompt = parts[2].strip()
    
    metadata = yaml.safe_load(metadata_text)
    
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Agent definition '{path}' contains invalid metadata."
        )
        
    required_fields = [
        "name", 
        "description",
    ]
        
    for field in required_fields:
        if field not in metadata:
            raise ValueError(
                f"Agent definition '{path}' is missing required field: {field}"
            )
            
    return AgentDefinition(
        name=metadata["name"],
        description=metadata["description"],
        tools=metadata.get("tools", []),
        model=metadata.get("model", "local-qwen"),
        max_steps=metadata.get("max_steps", 3),
        system_prompt=system_prompt,
    )
    
def load_agent_directory(
        directory: str | Path,
    ) -> list[AgentDefinition]:
    """
    Load every .md agent definition from a directory.
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(
            f"Agent definition directory not found: {directory}"
        )
    
    if not directory.is_dir():
        raise ValueError(
            f"Agent definition path is not a directory: {directory}"
        )
        
    agents = []
    
    for path in sorted(directory.glob("*.md")):
        agents.append(
            load_agent_definition(path)
        )
        
    return agents