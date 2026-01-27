"""
Cost Tracking Example

Demonstrates how to use cost tracking and monitoring in Jotty.
"""
from core.monitoring import CostTracker, MonitoringFramework, EfficiencyMetrics
from core.foundation.data_structures import SwarmConfig

# Example: Basic cost tracking
def example_cost_tracking():
    """Basic cost tracking example."""
    print("=== Cost Tracking Example ===\n")
    
    # Create cost tracker
    tracker = CostTracker(enable_tracking=True)
    
    # Simulate some LLM calls
    print("Recording LLM calls...")
    tracker.record_llm_call(
        provider="anthropic",
        model="claude-sonnet-4",
        input_tokens=1000,
        output_tokens=500,
        success=True
    )
    
    tracker.record_llm_call(
        provider="openai",
        model="gpt-4-turbo",
        input_tokens=2000,
        output_tokens=1000,
        success=True
    )
    
    tracker.record_llm_call(
        provider="anthropic",
        model="claude-sonnet-4",
        input_tokens=500,
        output_tokens=200,
        success=False,
        error="Timeout"
    )
    
    # Get metrics
    metrics = tracker.get_metrics()
    
    print(f"\n=== Cost Metrics ===")
    print(f"Total cost: ${metrics.total_cost:.6f}")
    print(f"Total tokens: {metrics.total_tokens:,}")
    print(f"Total calls: {metrics.total_calls}")
    print(f"Successful calls: {metrics.successful_calls}")
    print(f"Failed calls: {metrics.failed_calls}")
    print(f"Avg cost per call: ${metrics.avg_cost_per_call:.6f}")
    print(f"Cost per 1K tokens: ${metrics.cost_per_1k_tokens:.6f}")
    
    print(f"\n=== Cost by Provider ===")
    for provider, cost in metrics.cost_by_provider.items():
        print(f"  {provider}: ${cost:.6f}")
    
    print(f"\n=== Cost by Model ===")
    for model, cost in metrics.cost_by_model.items():
        print(f"  {model}: ${cost:.6f}")
    
    # Efficiency metrics
    efficiency = tracker.get_efficiency_metrics(success_count=2)
    print(f"\n=== Efficiency Metrics ===")
    print(f"Cost per success: ${efficiency['cost_per_success']:.6f}")
    print(f"Efficiency score: {efficiency['efficiency_score']:.4f}")
    print(f"Success rate: {efficiency['success_rate']:.2%}")


# Example: Monitoring framework
def example_monitoring():
    """Monitoring framework example."""
    print("\n\n=== Monitoring Framework Example ===\n")
    
    # Create monitoring framework
    monitor = MonitoringFramework(enable_monitoring=True)
    
    # Track some executions
    print("Tracking executions...")
    
    exec1 = monitor.start_execution("agent_planner", "task_001")
    # ... simulate work ...
    import time
    time.sleep(0.1)  # Simulate work
    monitor.finish_execution(
        exec1,
        status=monitor.ExecutionStatus.SUCCESS,
        input_tokens=1000,
        output_tokens=500,
        cost=0.015
    )
    
    exec2 = monitor.start_execution("agent_executor", "task_001")
    time.sleep(0.15)
    monitor.finish_execution(
        exec2,
        status=monitor.ExecutionStatus.SUCCESS,
        input_tokens=2000,
        output_tokens=1000,
        cost=0.030
    )
    
    exec3 = monitor.start_execution("agent_reviewer", "task_001")
    time.sleep(0.05)
    monitor.finish_execution(
        exec3,
        status=monitor.ExecutionStatus.FAILURE,
        error="Validation failed",
        input_tokens=500,
        output_tokens=100,
        cost=0.005
    )
    
    # Get performance metrics
    perf_metrics = monitor.get_performance_metrics()
    
    print(f"\n=== Performance Metrics ===")
    print(f"Total executions: {perf_metrics.total_executions}")
    print(f"Successful: {perf_metrics.successful_executions}")
    print(f"Failed: {perf_metrics.failed_executions}")
    print(f"Success rate: {perf_metrics.success_rate:.2%}")
    print(f"Error rate: {perf_metrics.error_rate:.2%}")
    print(f"Total duration: {perf_metrics.total_duration:.2f}s")
    print(f"Avg duration: {perf_metrics.avg_duration:.2f}s")
    print(f"Total cost: ${perf_metrics.total_cost:.6f}")
    print(f"Avg cost: ${perf_metrics.avg_cost:.6f}")
    
    print(f"\n=== Executions by Agent ===")
    for agent, count in perf_metrics.executions_by_agent.items():
        print(f"  {agent}: {count}")
    
    # Error analysis
    error_analysis = monitor.get_error_analysis()
    print(f"\n=== Error Analysis ===")
    print(f"Total errors: {error_analysis['total_errors']}")
    print(f"Error rate: {error_analysis['error_rate']:.2%}")
    
    # Generate full report
    report = monitor.generate_report()
    print(f"\n=== Full Report ===")
    print(f"Uptime: {report['uptime_seconds']:.2f}s")
    print(f"Monitoring enabled: {report['monitoring_enabled']}")


# Example: Integration with SwarmConfig
def example_config_integration():
    """Example of using cost tracking with SwarmConfig."""
    print("\n\n=== Config Integration Example ===\n")
    
    # Create config with cost tracking enabled
    config = SwarmConfig(
        enable_cost_tracking=True,
        cost_budget=10.0,  # $10 budget
        enable_monitoring=True,
        monitoring_output_dir="./monitoring_reports"
    )
    
    print(f"Cost tracking enabled: {config.enable_cost_tracking}")
    print(f"Cost budget: ${config.cost_budget}")
    print(f"Monitoring enabled: {config.enable_monitoring}")
    print(f"Monitoring output dir: {config.monitoring_output_dir}")


if __name__ == "__main__":
    example_cost_tracking()
    example_monitoring()
    example_config_integration()
    
    print("\n\n=== Examples Complete ===")
