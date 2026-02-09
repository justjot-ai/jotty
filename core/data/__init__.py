"""
Data Layer - Data Management & Parameter Resolution
===================================================

Data processing, parameter resolution, and data discovery.
All imports are lazy to avoid pulling in DSPy at module load time.
"""

import importlib as _importlib

_LAZY_IMPORTS: dict[str, str] = {
    # feedback_router
    "AgenticFeedbackRouter": ".feedback_router",
    "AgenticFeedbackSignature": ".feedback_router",
    # parameter_resolver
    "AgenticParameterResolver": ".parameter_resolver",
    "ParameterMatchingSignature": ".parameter_resolver",
    # data_registry
    "DataArtifact": ".data_registry",
    "DataRegistry": ".data_registry",
    "DataRegistryTool": ".data_registry",
    # information_storage
    "InformationTheoreticStorage": ".information_storage",
    "InformationWeightedMemory": ".information_storage",
    "SurpriseEstimator": ".information_storage",
    "SurpriseSignature": ".information_storage",
    # io_manager
    "ActorOutput": ".io_manager",
    "IOManager": ".io_manager",
    "KnowledgeProvenance": ".io_manager",
    "SwarmResult": ".io_manager",
    # data_extractor
    "SmartDataExtractor": ".data_extractor",
    # data_transformer
    "FormatTools": ".data_transformer",
    "SmartDataTransformer": ".data_transformer",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())
