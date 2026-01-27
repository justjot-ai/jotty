"""
Reproducibility Framework

Ensures reproducible results through fixed seeds and standardized protocols.
Based on OAgents reproducibility approach.
"""
import random
import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import numpy (optional)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.debug("numpy not available. NumPy seed setting will be skipped.")

# Try to import torch (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.debug("torch not available. PyTorch seed setting will be skipped.")


@dataclass
class ReproducibilityConfig:
    """
    Configuration for reproducibility guarantees.
    
    Ensures deterministic behavior through fixed seeds.
    """
    random_seed: Optional[int] = None  # Python random seed
    numpy_seed: Optional[int] = None  # NumPy random seed (if available)
    torch_seed: Optional[int] = None  # PyTorch random seed (if available)
    python_hash_seed: Optional[int] = None  # Python hash randomization seed
    enable_deterministic: bool = True  # Enable deterministic operations
    
    def __post_init__(self):
        """Set seeds if provided."""
        if self.random_seed is not None:
            set_reproducible_seeds(
                random_seed=self.random_seed,
                numpy_seed=self.numpy_seed or self.random_seed,
                torch_seed=self.torch_seed or self.random_seed,
                python_hash_seed=self.python_hash_seed or self.random_seed,
                enable_deterministic=self.enable_deterministic
            )


def set_reproducible_seeds(
    random_seed: Optional[int] = None,
    numpy_seed: Optional[int] = None,
    torch_seed: Optional[int] = None,
    python_hash_seed: Optional[int] = None,
    enable_deterministic: bool = True
) -> Dict[str, Any]:
    """
    Set all random seeds for reproducibility.
    
    Args:
        random_seed: Python random seed
        numpy_seed: NumPy random seed
        torch_seed: PyTorch random seed
        python_hash_seed: Python hash randomization seed
        enable_deterministic: Enable deterministic operations (PyTorch)
        
    Returns:
        Dictionary with seed information
        
    Example:
        seeds = set_reproducible_seeds(random_seed=42)
        print(f"Seeds set: {seeds}")
    """
    seeds_set = {}
    
    # Python random
    if random_seed is not None:
        random.seed(random_seed)
        seeds_set['random'] = random_seed
        logger.info(f"Set Python random seed: {random_seed}")
    
    # NumPy random
    if numpy_seed is not None and NUMPY_AVAILABLE:
        np.random.seed(numpy_seed)
        seeds_set['numpy'] = numpy_seed
        logger.info(f"Set NumPy random seed: {numpy_seed}")
    
    # PyTorch random
    if torch_seed is not None and TORCH_AVAILABLE:
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        seeds_set['torch'] = torch_seed
        
        if enable_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            seeds_set['torch_deterministic'] = True
        
        logger.info(f"Set PyTorch random seed: {torch_seed}")
    
    # Python hash randomization
    if python_hash_seed is not None:
        os.environ['PYTHONHASHSEED'] = str(python_hash_seed)
        seeds_set['python_hash'] = python_hash_seed
        logger.info(f"Set PYTHONHASHSEED: {python_hash_seed}")
    
    return seeds_set


def ensure_reproducibility(config: ReproducibilityConfig) -> Dict[str, Any]:
    """
    Ensure reproducibility based on config.
    
    Args:
        config: ReproducibilityConfig instance
        
    Returns:
        Dictionary with seed information
    """
    return set_reproducible_seeds(
        random_seed=config.random_seed,
        numpy_seed=config.numpy_seed,
        torch_seed=config.torch_seed,
        python_hash_seed=config.python_hash_seed,
        enable_deterministic=config.enable_deterministic
    )


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get current reproducibility settings.
    
    Returns:
        Dictionary with current seed information
    """
    info = {
        'random_seed_set': False,
        'numpy_seed_set': NUMPY_AVAILABLE,
        'torch_seed_set': TORCH_AVAILABLE,
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
    }
    
    # Check if seeds are set (can't directly check, but can check environment)
    if os.environ.get('PYTHONHASHSEED'):
        info['python_hash_seed_set'] = True
    
    return info
