"""
Cost Tracking Module

Tracks LLM API costs and provides cost metrics for Jotty framework.
Based on OAgents cost efficiency research.
"""
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (as of 2026-02-10)
# Source: https://docs.anthropic.com/en/docs/about-claude/models
PRICING_TABLE = {
    # Anthropic Claude 4.x
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    # Anthropic Claude 3.5
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 1.0, "output": 5.0},
    "claude-3-5-haiku-latest": {"input": 1.0, "output": 5.0},
    # Anthropic Claude 3 (legacy)
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Aliases
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-haiku": {"input": 1.0, "output": 5.0},

    # OpenAI
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "o1": {"input": 15.0, "output": 60.0},
    "o3-mini": {"input": 1.1, "output": 4.4},

    # Gemini
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},

    # Claude CLI (uses API pricing)
    "claude-cli": {"input": 3.0, "output": 15.0},
}

# Default pricing if model not found
DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""
    timestamp: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    success: bool
    error: Optional[str] = None
    duration: Optional[float] = None  # Duration in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "success": self.success,
            "error": self.error,
            "duration": self.duration,
        }


@dataclass
class CostMetrics:
    """Cost metrics summary."""
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_calls: int
    successful_calls: int
    failed_calls: int
    avg_cost_per_call: float
    avg_tokens_per_call: float
    cost_per_1k_tokens: float
    cost_by_provider: Dict[str, float]
    cost_by_model: Dict[str, float]
    calls_by_provider: Dict[str, int]
    calls_by_model: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "avg_cost_per_call": self.avg_cost_per_call,
            "avg_tokens_per_call": self.avg_tokens_per_call,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "cost_by_provider": self.cost_by_provider,
            "cost_by_model": self.cost_by_model,
            "calls_by_provider": self.calls_by_provider,
            "calls_by_model": self.calls_by_model,
        }


class CostTracker:
    """
    Tracks LLM API costs and provides cost metrics.
    
    Based on OAgents cost efficiency research.
    Provides cost-per-success, efficiency scores, and detailed cost breakdowns.
    
    Usage:
        tracker = CostTracker()
        
        # Record LLM call
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True
        )
        
        # Get metrics
        metrics = tracker.get_metrics()
        print(f"Total cost: ${metrics.total_cost:.4f}")
        
        # Get efficiency metrics
        efficiency = tracker.get_efficiency_metrics(success_count=10)
        print(f"Cost per success: ${efficiency.cost_per_success:.4f}")
    """
    
    def __init__(self, enable_tracking: bool = True) -> None:
        """
        Initialize cost tracker.
        
        Args:
            enable_tracking: Whether to enable cost tracking (opt-in)
        """
        self.enable_tracking = enable_tracking
        self.calls: List[LLMCallRecord] = []
        self.start_time = time.time()
        
        # Pricing table (can be updated)
        self.pricing_table = PRICING_TABLE.copy()
        self.default_pricing = DEFAULT_PRICING.copy()
    
    def update_pricing(self, model: str, input_price: float, output_price: float) -> None:
        """
        Update pricing for a specific model.
        
        Args:
            model: Model name
            input_price: Price per 1M input tokens
            output_price: Price per 1M output tokens
        """
        self.pricing_table[model] = {"input": input_price, "output": output_price}
        logger.info(f"Updated pricing for {model}: ${input_price}/${output_price} per 1M tokens")
    
    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a model call.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        # Get pricing for model
        pricing = self.pricing_table.get(model, self.default_pricing)
        
        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def record_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
        error: Optional[str] = None,
        duration: Optional[float] = None,
        custom_cost: Optional[float] = None
    ) -> LLMCallRecord:
        """
        Record an LLM API call.
        
        Args:
            provider: Provider name (e.g., "anthropic", "openai")
            model: Model name (e.g., "claude-sonnet-4")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            success: Whether call was successful
            error: Error message if failed
            duration: Duration in seconds
            custom_cost: Custom cost override (if provided, uses this instead of calculating)
            
        Returns:
            LLMCallRecord with cost information
        """
        if not self.enable_tracking:
            return LLMCallRecord(
                timestamp=time.time(),
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=0.0,
                success=success,
                error=error,
                duration=duration
            )
        
        # Calculate cost
        if custom_cost is not None:
            cost = custom_cost
        else:
            cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        # Create record
        record = LLMCallRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            success=success,
            error=error,
            duration=duration
        )
        
        # Store record
        self.calls.append(record)
        
        logger.debug(
            f"Recorded LLM call: {provider}/{model} - "
            f"{input_tokens}+{output_tokens} tokens = ${cost:.6f}"
        )
        
        return record
    
    def get_metrics(self) -> CostMetrics:
        """
        Get cost metrics summary.
        
        Returns:
            CostMetrics with aggregated statistics
        """
        if not self.calls:
            return CostMetrics(
                total_cost=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_calls=0,
                successful_calls=0,
                failed_calls=0,
                avg_cost_per_call=0.0,
                avg_tokens_per_call=0.0,
                cost_per_1k_tokens=0.0,
                cost_by_provider={},
                cost_by_model={},
                calls_by_provider={},
                calls_by_model={},
            )
        
        # Aggregate statistics
        total_cost = sum(call.cost for call in self.calls)
        total_input_tokens = sum(call.input_tokens for call in self.calls)
        total_output_tokens = sum(call.output_tokens for call in self.calls)
        total_tokens = total_input_tokens + total_output_tokens
        total_calls = len(self.calls)
        successful_calls = sum(1 for call in self.calls if call.success)
        failed_calls = total_calls - successful_calls
        
        # Cost by provider
        cost_by_provider: Dict[str, float] = {}
        calls_by_provider: Dict[str, int] = {}
        for call in self.calls:
            cost_by_provider[call.provider] = cost_by_provider.get(call.provider, 0.0) + call.cost
            calls_by_provider[call.provider] = calls_by_provider.get(call.provider, 0) + 1
        
        # Cost by model
        cost_by_model: Dict[str, float] = {}
        calls_by_model: Dict[str, int] = {}
        for call in self.calls:
            cost_by_model[call.model] = cost_by_model.get(call.model, 0.0) + call.cost
            calls_by_model[call.model] = calls_by_model.get(call.model, 0) + 1
        
        # Averages
        avg_cost_per_call = total_cost / total_calls if total_calls > 0 else 0.0
        avg_tokens_per_call = total_tokens / total_calls if total_calls > 0 else 0.0
        cost_per_1k_tokens = (total_cost / total_tokens * 1000) if total_tokens > 0 else 0.0
        
        return CostMetrics(
            total_cost=total_cost,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            avg_cost_per_call=avg_cost_per_call,
            avg_tokens_per_call=avg_tokens_per_call,
            cost_per_1k_tokens=cost_per_1k_tokens,
            cost_by_provider=cost_by_provider,
            cost_by_model=cost_by_model,
            calls_by_provider=calls_by_provider,
            calls_by_model=calls_by_model,
        )
    
    def get_efficiency_metrics(self, success_count: int) -> Dict[str, Any]:
        """
        Get efficiency metrics (cost-per-success, etc.).
        
        Args:
            success_count: Number of successful tasks/episodes
            
        Returns:
            Dictionary with efficiency metrics
        """
        metrics = self.get_metrics()
        
        cost_per_success = metrics.total_cost / max(success_count, 1)
        
        # Efficiency score: inverse of cost-per-success (higher is better)
        # Normalized to 0-1 scale (assuming $1 per success is baseline)
        efficiency_score = 1.0 / max(cost_per_success, 0.001)
        
        return {
            "cost_per_success": cost_per_success,
            "efficiency_score": efficiency_score,
            "success_count": success_count,
            "total_cost": metrics.total_cost,
            "success_rate": success_count / max(metrics.total_calls, 1),
        }
    
    def reset(self) -> None:
        """Reset cost tracking (start fresh)."""
        self.calls.clear()
        self.start_time = time.time()
        logger.info("Cost tracker reset")
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save cost tracking data to file.
        
        Args:
            filepath: Path to save file (JSON format)
        """
        data = {
            "start_time": self.start_time,
            "calls": [call.to_dict() for call in self.calls],
            "metrics": self.get_metrics().to_dict(),
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Cost tracking data saved to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load cost tracking data from file.
        
        Args:
            filepath: Path to load file from (JSON format)
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Cost tracking file not found: {filepath}")
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.start_time = data.get("start_time", time.time())
        self.calls = [
            LLMCallRecord(**call_data)
            for call_data in data.get("calls", [])
        ]
        
        logger.info(f"Cost tracking data loaded from {filepath}")
