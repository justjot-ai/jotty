"""
Health Check Module

Provides health and readiness endpoints for:
- Kubernetes liveness/readiness probes
- Load balancer health checks
- Monitoring systems
"""
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import time


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    metadata: Dict[str, Any] = None


class HealthCheck:
    """
    Health check system for Jotty.
    
    Usage:
        health = HealthCheck()
        health.add_check("database", check_database_connection)
        health.add_check("llm_provider", check_llm_available)
        
        status = health.check_all()
        if status['status'] == 'healthy':
            print("System healthy!")
    """

    def __init__(self):
        """Initialize health check system."""
        self.checks: Dict[str, Callable] = {}

    def add_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """
        Add health check.
        
        Args:
            name: Check name
            check_fn: Function returning True if healthy
        """
        self.checks[name] = check_fn

    def run_check(self, name: str) -> HealthCheckResult:
        """
        Run single health check.
        
        Args:
            name: Check name
        
        Returns:
            HealthCheckResult
        """
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check '{name}' not found",
                duration_ms=0
            )

        start_time = time.time()
        try:
            result = self.checks[name]()
            duration = (time.time() - start_time) * 1000

            if result:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="OK",
                    duration_ms=duration
                )
            else:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.DEGRADED,
                    message="Check returned False",
                    duration_ms=duration
                )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration
            )

    def check_all(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Dict with overall status and individual check results
        """
        results = [self.run_check(name) for name in self.checks.keys()]

        # Determine overall status
        if all(r.status == HealthStatus.HEALTHY for r in results):
            overall_status = HealthStatus.HEALTHY
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        return {
            "status": overall_status.value,
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_ms": r.duration_ms
                }
                for r in results
            ],
            "timestamp": time.time()
        }

    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.check_all()["status"] == HealthStatus.HEALTHY.value


# Default health checks
def check_memory_system() -> bool:
    """Check if memory system is available."""
    try:
        from Jotty.core.memory import get_memory_system
        mem = get_memory_system()
        return mem is not None
    except Exception:
        return False


def check_llm_provider() -> bool:
    """Check if LLM provider is configured."""
    import os
    return bool(os.getenv('ANTHROPIC_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('GROQ_API_KEY'))


def check_registry() -> bool:
    """Check if skill registry is available."""
    try:
        from Jotty.core.registry import get_unified_registry
        registry = get_unified_registry()
        return registry is not None
    except Exception:
        return False


# Singleton instance
_health_check: HealthCheck = None


def get_health_check() -> HealthCheck:
    """
    Get singleton health check instance with default checks.
    
    Returns:
        HealthCheck instance
    """
    global _health_check
    
    if _health_check is None:
        _health_check = HealthCheck()
        _health_check.add_check("memory_system", check_memory_system)
        _health_check.add_check("llm_provider", check_llm_provider)
        _health_check.add_check("skill_registry", check_registry)
    
    return _health_check
