"""Default configuration values for Jotty CLI."""

from .schema import CLIConfig

# Default configuration instance
DEFAULT_CONFIG = CLIConfig.default()

# Default config YAML template
DEFAULT_CONFIG_YAML = """# Jotty CLI Configuration
# ========================
# Location: ~/.jotty/config.yaml

provider:
  # Use Jotty's UnifiedLMProvider/SwarmProviderGateway
  use_unified: true
  fallback_order:
    - claude-cli  # Free, uses local Claude CLI credentials (default)
    - anthropic
    - openrouter
    - openai
    - groq
  default_model: null  # Use provider default

swarm:
  enable_zero_config: true
  max_agents: 10
  enable_learning: true
  enable_memory: true

learning:
  enable_warmup: true
  warmup_episodes: 10
  auto_save: true
  save_interval: 100

ui:
  library: rich
  theme: default
  enable_progress: true
  enable_spinner: true
  max_width: 120

features:
  git_integration: true
  plugin_system: true
  web_search: true
  file_operations: true
  expose_all_skills: true

session:
  auto_save: true
  context_window: 20
  history_file: null  # Default: ~/.jotty/history
  session_dir: null   # Default: ~/.jotty/sessions
"""

# Config paths
DEFAULT_CONFIG_DIR = "~/.jotty"
DEFAULT_CONFIG_FILE = "config.yaml"
DEFAULT_HISTORY_FILE = "history"
DEFAULT_SESSION_DIR = "sessions"
