"""
CLI Configuration Schema
========================

Pydantic-based configuration for Jotty CLI.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProviderConfig:
    """Provider configuration."""
    use_unified: bool = True
    fallback_order: List[str] = field(default_factory=lambda: [
        "claude-cli",  # Free, uses local Claude CLI credentials
        "anthropic", "openrouter", "openai", "groq"
    ])
    default_model: Optional[str] = None


@dataclass
class SwarmConfig:
    """Swarm configuration."""
    enable_zero_config: bool = True
    max_agents: int = 10
    enable_learning: bool = True
    enable_memory: bool = True


@dataclass
class LearningConfig:
    """Learning configuration."""
    enable_warmup: bool = True
    warmup_episodes: int = 10
    auto_save: bool = True
    save_interval: int = 100  # episodes


@dataclass
class UIConfig:
    """UI configuration."""
    library: str = "rich"
    theme: str = "default"
    enable_progress: bool = True
    enable_spinner: bool = True
    max_width: int = 120


@dataclass
class FeaturesConfig:
    """Features configuration."""
    git_integration: bool = True
    plugin_system: bool = True
    web_search: bool = True
    file_operations: bool = True
    expose_all_skills: bool = True


@dataclass
class SessionConfig:
    """Session configuration."""
    auto_save: bool = True
    context_window: int = 20
    history_file: Optional[str] = None
    session_dir: Optional[str] = None


@dataclass
class CLIConfig:
    """
    Main CLI Configuration.

    Loaded from ~/.jotty/config.yaml or custom path.
    """
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    session: SessionConfig = field(default_factory=SessionConfig)

    # Runtime settings
    debug: bool = False
    no_color: bool = False
    working_dir: Optional[str] = None

    @classmethod
    def default(cls) -> "CLIConfig":
        """Create default configuration."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": {
                "use_unified": self.provider.use_unified,
                "fallback_order": self.provider.fallback_order,
                "default_model": self.provider.default_model,
            },
            "swarm": {
                "enable_zero_config": self.swarm.enable_zero_config,
                "max_agents": self.swarm.max_agents,
                "enable_learning": self.swarm.enable_learning,
                "enable_memory": self.swarm.enable_memory,
            },
            "learning": {
                "enable_warmup": self.learning.enable_warmup,
                "warmup_episodes": self.learning.warmup_episodes,
                "auto_save": self.learning.auto_save,
                "save_interval": self.learning.save_interval,
            },
            "ui": {
                "library": self.ui.library,
                "theme": self.ui.theme,
                "enable_progress": self.ui.enable_progress,
                "enable_spinner": self.ui.enable_spinner,
                "max_width": self.ui.max_width,
            },
            "features": {
                "git_integration": self.features.git_integration,
                "plugin_system": self.features.plugin_system,
                "web_search": self.features.web_search,
                "file_operations": self.features.file_operations,
                "expose_all_skills": self.features.expose_all_skills,
            },
            "session": {
                "auto_save": self.session.auto_save,
                "context_window": self.session.context_window,
                "history_file": self.session.history_file,
                "session_dir": self.session.session_dir,
            },
            "debug": self.debug,
            "no_color": self.no_color,
            "working_dir": self.working_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CLIConfig":
        """Create from dictionary."""
        config = cls()

        if "provider" in data:
            p = data["provider"]
            config.provider = ProviderConfig(
                use_unified=p.get("use_unified", True),
                fallback_order=p.get("fallback_order", ["claude-cli", "anthropic", "openrouter", "openai", "groq"]),
                default_model=p.get("default_model"),
            )

        if "swarm" in data:
            s = data["swarm"]
            config.swarm = SwarmConfig(
                enable_zero_config=s.get("enable_zero_config", True),
                max_agents=s.get("max_agents", 10),
                enable_learning=s.get("enable_learning", True),
                enable_memory=s.get("enable_memory", True),
            )

        if "learning" in data:
            l = data["learning"]
            config.learning = LearningConfig(
                enable_warmup=l.get("enable_warmup", True),
                warmup_episodes=l.get("warmup_episodes", 10),
                auto_save=l.get("auto_save", True),
                save_interval=l.get("save_interval", 100),
            )

        if "ui" in data:
            u = data["ui"]
            config.ui = UIConfig(
                library=u.get("library", "rich"),
                theme=u.get("theme", "default"),
                enable_progress=u.get("enable_progress", True),
                enable_spinner=u.get("enable_spinner", True),
                max_width=u.get("max_width", 120),
            )

        if "features" in data:
            f = data["features"]
            config.features = FeaturesConfig(
                git_integration=f.get("git_integration", True),
                plugin_system=f.get("plugin_system", True),
                web_search=f.get("web_search", True),
                file_operations=f.get("file_operations", True),
                expose_all_skills=f.get("expose_all_skills", True),
            )

        if "session" in data:
            ss = data["session"]
            config.session = SessionConfig(
                auto_save=ss.get("auto_save", True),
                context_window=ss.get("context_window", 20),
                history_file=ss.get("history_file"),
                session_dir=ss.get("session_dir"),
            )

        config.debug = data.get("debug", False)
        config.no_color = data.get("no_color", False)
        config.working_dir = data.get("working_dir")

        return config
