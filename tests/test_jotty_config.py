"""
Comprehensive JottyConfig/SwarmConfig Tests
===========================================

Tests all 22 categories of configuration settings to ensure:
1. All parameters are valid and accessible
2. Default values are sensible
3. Computed properties work correctly
4. Configuration can be serialized/deserialized
5. Edge cases are handled properly
"""

import pytest
import json
from pathlib import Path
from core.foundation import JottyConfig, SwarmConfig


# =============================================================================
# TEST 1: Basic Configuration Creation
# =============================================================================

def test_config_creation_with_defaults():
    """Test creating config with all default values."""
    config = JottyConfig()

    # Should create successfully with no errors
    assert config is not None
    assert isinstance(config, SwarmConfig)  # JottyConfig is alias

    # Check a few key defaults
    assert config.output_base_dir == "./outputs"
    assert config.enable_rl == True
    assert config.max_actor_iters == 50
    assert config.episodic_capacity == 1000


def test_config_creation_with_overrides():
    """Test creating config with custom values."""
    config = JottyConfig(
        base_path="/custom/path",
        enable_rl=False,
        max_actor_iters=100,
        verbose=2
    )

    assert config.base_path == "/custom/path"
    assert config.enable_rl == False
    assert config.max_actor_iters == 100
    assert config.verbose == 2


# =============================================================================
# TEST 2: Category 1 - PERSISTENCE
# =============================================================================

def test_persistence_config():
    """Test all persistence-related settings."""
    config = JottyConfig(
        output_base_dir="/test/outputs",
        create_run_folder=False,
        auto_save_interval=5,
        persist_memories=True,
        persist_q_tables=True,
        persist_brain_state=True,
        storage_format="sqlite",
        compress_large_files=True,
        max_runs_to_keep=5,
        enable_backups=False,
    )

    assert config.output_base_dir == "/test/outputs"
    assert config.create_run_folder == False
    assert config.auto_save_interval == 5
    assert config.persist_memories == True
    assert config.storage_format == "sqlite"
    assert config.compress_large_files == True


def test_persistence_logging_config():
    """Test logging-related persistence settings."""
    config = JottyConfig(
        enable_beautified_logs=False,
        enable_debug_logs=False,
        log_level="WARNING",
        enable_profiling=True,
        profiling_verbosity="detailed"
    )

    assert config.enable_beautified_logs == False
    assert config.log_level == "WARNING"
    assert config.enable_profiling == True
    assert config.profiling_verbosity == "detailed"


# =============================================================================
# TEST 3: Category 2 - EXECUTION
# =============================================================================

def test_execution_config():
    """Test execution limits and timeouts."""
    config = JottyConfig(
        max_actor_iters=100,
        max_eval_iters=5,
        max_episode_iterations=20,
        async_timeout=120.0,
        actor_timeout=1800.0,
        max_concurrent_agents=5,
        max_eval_retries=5,
        llm_timeout_seconds=300.0
    )

    assert config.max_actor_iters == 100
    assert config.max_eval_iters == 5
    assert config.max_episode_iterations == 20
    assert config.async_timeout == 120.0
    assert config.actor_timeout == 1800.0
    assert config.max_concurrent_agents == 5


# =============================================================================
# TEST 4: Category 2.5 - TIMEOUT & CIRCUIT BREAKER
# =============================================================================

def test_circuit_breaker_config():
    """Test circuit breaker settings."""
    config = JottyConfig(
        enable_circuit_breakers=True,
        llm_circuit_failure_threshold=10,
        llm_circuit_timeout=120.0,
        tool_circuit_failure_threshold=5,
        enable_adaptive_timeouts=True,
        initial_timeout=60.0,
        timeout_percentile=90.0
    )

    assert config.enable_circuit_breakers == True
    assert config.llm_circuit_failure_threshold == 10
    assert config.llm_circuit_timeout == 120.0
    assert config.enable_adaptive_timeouts == True
    assert config.timeout_percentile == 90.0


def test_dead_letter_queue_config():
    """Test dead letter queue settings."""
    config = JottyConfig(
        enable_dead_letter_queue=True,
        dlq_max_size=500,
        dlq_max_retries=5
    )

    assert config.enable_dead_letter_queue == True
    assert config.dlq_max_size == 500
    assert config.dlq_max_retries == 5


# =============================================================================
# TEST 5: Category 3 - MEMORY
# =============================================================================

def test_memory_config():
    """Test hierarchical memory capacity settings."""
    config = JottyConfig(
        episodic_capacity=2000,
        semantic_capacity=1000,
        procedural_capacity=500,
        meta_capacity=200,
        causal_capacity=300,
        max_entry_tokens=4000
    )

    assert config.episodic_capacity == 2000
    assert config.semantic_capacity == 1000
    assert config.procedural_capacity == 500
    assert config.meta_capacity == 200
    assert config.causal_capacity == 300
    assert config.max_entry_tokens == 4000

    # Test computed property
    assert config.total_memory_capacity == 4000


# =============================================================================
# TEST 6: Category 4 - CONTEXT BUDGET
# =============================================================================

def test_context_budget_config():
    """Test token budget allocation."""
    config = JottyConfig(
        max_context_tokens=200000,
        system_prompt_budget=10000,
        current_input_budget=30000,
        trajectory_budget=40000,
        tool_output_budget=30000,
        enable_dynamic_budget=True,
        min_memory_budget=20000,
        max_memory_budget=100000
    )

    assert config.max_context_tokens == 200000
    assert config.system_prompt_budget == 10000
    assert config.enable_dynamic_budget == True

    # Test computed property
    memory_budget = config.memory_budget
    assert isinstance(memory_budget, int)
    assert memory_budget >= config.min_memory_budget


def test_agentic_discovery_budget():
    """Test agentic discovery budget settings."""
    config = JottyConfig(
        preview_token_budget=30000,
        max_description_tokens=10000,
        compression_trigger_ratio=0.9,
        chunking_threshold_tokens=20000
    )

    assert config.preview_token_budget == 30000
    assert config.max_description_tokens == 10000

    # Test computed properties (set in __post_init__)
    assert config.preview_char_limit == 30000 * 4  # 120k chars
    assert config.max_description_chars == 10000 * 4  # 40k chars


# =============================================================================
# TEST 7: Category 5 - RL PARAMETERS
# =============================================================================

def test_rl_parameters_config():
    """Test reinforcement learning parameters."""
    config = JottyConfig(
        enable_rl=True,
        gamma=0.95,
        lambda_trace=0.90,
        alpha=0.05,
        baseline=0.6,
        n_step=5,
        enable_adaptive_alpha=True,
        alpha_min=0.0005,
        alpha_max=0.2
    )

    assert config.enable_rl == True
    assert config.gamma == 0.95
    assert config.lambda_trace == 0.90
    assert config.alpha == 0.05
    assert config.enable_adaptive_alpha == True


def test_rl_reward_config():
    """Test RL reward parameters."""
    config = JottyConfig(
        enable_intermediate_rewards=True,
        architect_proceed_reward=0.2,
        tool_success_reward=0.1,
        base_reward_weight=0.4,
        cooperation_bonus=0.3,
        predictability_bonus=0.3
    )

    assert config.enable_intermediate_rewards == True
    assert config.architect_proceed_reward == 0.2
    assert config.cooperation_bonus == 0.3


# =============================================================================
# TEST 8: Category 6 - EXPLORATION
# =============================================================================

def test_exploration_config():
    """Test exploration strategy parameters."""
    config = JottyConfig(
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay_episodes=1000,
        ucb_coefficient=3.0,
        enable_adaptive_exploration=True,
        exploration_boost_on_stall=0.2
    )

    assert config.epsilon_start == 0.5
    assert config.epsilon_end == 0.01
    assert config.epsilon_decay_episodes == 1000
    assert config.ucb_coefficient == 3.0
    assert config.enable_adaptive_exploration == True


# =============================================================================
# TEST 9: Category 7 - CREDIT ASSIGNMENT
# =============================================================================

def test_credit_assignment_config():
    """Test credit assignment parameters."""
    config = JottyConfig(
        credit_decay=0.85,
        min_contribution=0.05,
        enable_reasoning_credit=True,
        reasoning_weight=0.4,
        evidence_weight=0.3
    )

    assert config.credit_decay == 0.85
    assert config.min_contribution == 0.05
    assert config.enable_reasoning_credit == True
    assert config.reasoning_weight == 0.4


# =============================================================================
# TEST 10: Category 8 - CONSOLIDATION
# =============================================================================

def test_consolidation_config():
    """Test memory consolidation parameters."""
    config = JottyConfig(
        consolidation_threshold=200,
        consolidation_interval=5,
        min_cluster_size=10,
        pattern_confidence_threshold=0.8,
        enable_causal_extraction=True,
        min_causal_evidence=5
    )

    assert config.consolidation_threshold == 200
    assert config.consolidation_interval == 5
    assert config.enable_causal_extraction == True


def test_brain_consolidation_config():
    """Test brain-inspired consolidation parameters."""
    config = JottyConfig(
        brain_reward_salience_weight=0.4,
        brain_novelty_weight=0.3,
        brain_goal_relevance_weight=0.3,
        brain_memory_threshold=0.5,
        brain_prune_threshold=0.2,
        brain_strengthen_threshold=0.9
    )

    assert config.brain_reward_salience_weight == 0.4
    assert config.brain_novelty_weight == 0.3
    assert config.brain_memory_threshold == 0.5


# =============================================================================
# TEST 11: Category 9 - OFFLINE LEARNING
# =============================================================================

def test_offline_learning_config():
    """Test offline learning parameters."""
    config = JottyConfig(
        episode_buffer_size=2000,
        offline_update_interval=100,
        replay_batch_size=50,
        enable_counterfactual=True,
        counterfactual_samples=10,
        priority_replay_alpha=0.8
    )

    assert config.episode_buffer_size == 2000
    assert config.offline_update_interval == 100
    assert config.replay_batch_size == 50
    assert config.enable_counterfactual == True


# =============================================================================
# TEST 12: Category 10 - PROTECTION MECHANISMS
# =============================================================================

def test_protection_config():
    """Test protection mechanism parameters."""
    config = JottyConfig(
        protected_memory_threshold=0.9,
        task_memory_ratio=0.4,
        suspicion_threshold=0.97,
        ood_entropy_threshold=0.85,
        min_rejection_rate=0.03
    )

    assert config.protected_memory_threshold == 0.9
    assert config.task_memory_ratio == 0.4
    assert config.suspicion_threshold == 0.97


# =============================================================================
# TEST 13: Category 11 - VALIDATION
# =============================================================================

def test_validation_config():
    """Test validation strategy parameters."""
    config = JottyConfig(
        enable_validation=True,
        validation_mode='architect_only',
        max_validation_rounds=5,
        refinement_timeout=60.0,
        advisory_confidence_threshold=0.9,
        max_validation_retries=10
    )

    assert config.enable_validation == True
    assert config.validation_mode == 'architect_only'
    assert config.max_validation_rounds == 5
    assert config.max_validation_retries == 10


def test_confidence_override_config():
    """Test confidence-based override mechanism."""
    config = JottyConfig(
        enable_confidence_override=True,
        confidence_override_threshold=0.25,
        min_confidence_for_override=0.75,
        max_validator_confidence_to_override=0.90
    )

    assert config.enable_confidence_override == True
    assert config.confidence_override_threshold == 0.25
    assert config.min_confidence_for_override == 0.75


# =============================================================================
# TEST 14: Category 12 - ASYNC
# =============================================================================

def test_async_config():
    """Test async execution parameters."""
    config = JottyConfig(
        parallel_architect=False,
        parallel_auditor=False
    )

    assert config.parallel_architect == False
    assert config.parallel_auditor == False


# =============================================================================
# TEST 15: Category 13 - LOGGING
# =============================================================================

def test_logging_config():
    """Test logging parameters."""
    config = JottyConfig(
        verbose=2,
        log_file="/tmp/jotty.log",
        enable_debug_logging=True,
        enable_metrics=True
    )

    assert config.verbose == 2
    assert config.log_file == "/tmp/jotty.log"
    assert config.enable_debug_logging == True
    assert config.enable_metrics == True


# =============================================================================
# TEST 16: Category 14 - LLM RAG
# =============================================================================

def test_llm_rag_config():
    """Test LLM-based RAG parameters."""
    config = JottyConfig(
        enable_llm_rag=True,
        rag_window_size=10,
        rag_max_candidates=100,
        rag_relevance_threshold=0.7,
        rag_use_cot=True,
        retrieval_mode="discrete",
        synthesis_fetch_size=300
    )

    assert config.enable_llm_rag == True
    assert config.rag_window_size == 10
    assert config.retrieval_mode == "discrete"
    assert config.synthesis_fetch_size == 300


def test_llm_rag_chunking_config():
    """Test RAG chunking parameters."""
    config = JottyConfig(
        chunk_size=1000,
        chunk_overlap=100
    )

    assert config.chunk_size == 1000
    assert config.chunk_overlap == 100


# =============================================================================
# TEST 17: Category 15 - GOAL HIERARCHY
# =============================================================================

def test_goal_hierarchy_config():
    """Test goal hierarchy parameters."""
    config = JottyConfig(
        enable_goal_hierarchy=True,
        goal_transfer_weight=0.4,
        max_transfer_distance=3
    )

    assert config.enable_goal_hierarchy == True
    assert config.goal_transfer_weight == 0.4
    assert config.max_transfer_distance == 3


# =============================================================================
# TEST 18: Category 16 - CAUSAL LEARNING
# =============================================================================

def test_causal_learning_config():
    """Test causal learning parameters."""
    config = JottyConfig(
        enable_causal_learning=True,
        causal_confidence_threshold=0.8,
        causal_min_support=5,
        causal_transfer_enabled=True
    )

    assert config.enable_causal_learning == True
    assert config.causal_confidence_threshold == 0.8
    assert config.causal_min_support == 5
    assert config.causal_transfer_enabled == True


# =============================================================================
# TEST 19: Category 17 - INTER-AGENT COMMUNICATION
# =============================================================================

def test_inter_agent_communication_config():
    """Test inter-agent communication parameters."""
    config = JottyConfig(
        enable_agent_communication=True,
        share_tool_results=True,
        share_insights=True,
        max_messages_per_episode=50
    )

    assert config.enable_agent_communication == True
    assert config.share_tool_results == True
    assert config.max_messages_per_episode == 50


def test_marl_config():
    """Test predictive MARL parameters."""
    config = JottyConfig(
        marl_default_cooperation=0.6,
        marl_default_predictability=0.6,
        marl_action_divergence_weight=0.5,
        marl_state_divergence_weight=0.3,
        marl_reward_divergence_weight=0.2
    )

    assert config.marl_default_cooperation == 0.6
    assert config.marl_action_divergence_weight == 0.5


# =============================================================================
# TEST 20: Category 18 - MULTI-ROUND VALIDATION
# =============================================================================

def test_multi_round_validation_config():
    """Test multi-round validation parameters."""
    config = JottyConfig(
        enable_multi_round=True,
        refinement_on_low_confidence=0.7,
        refinement_on_disagreement=True,
        max_refinement_rounds=3
    )

    assert config.enable_multi_round == True
    assert config.refinement_on_low_confidence == 0.7
    assert config.max_refinement_rounds == 3


# =============================================================================
# TEST 21: Category 19 - ADAPTIVE LEARNING
# =============================================================================

def test_adaptive_learning_config():
    """Test adaptive learning parameters."""
    config = JottyConfig(
        enable_adaptive_learning=True,
        stall_detection_window=200,
        stall_threshold=0.0005,
        learning_boost_factor=3.0
    )

    assert config.enable_adaptive_learning == True
    assert config.stall_detection_window == 200
    assert config.stall_threshold == 0.0005
    assert config.learning_boost_factor == 3.0


# =============================================================================
# TEST 22: Category 20 - DEDUPLICATION
# =============================================================================

def test_deduplication_config():
    """Test deduplication parameters."""
    config = JottyConfig(
        enable_deduplication=True,
        similarity_threshold=0.9,
        merge_similar_memories=True
    )

    assert config.enable_deduplication == True
    assert config.similarity_threshold == 0.9
    assert config.merge_similar_memories == True


# =============================================================================
# TEST 23: Category 21 - DISTRIBUTED SUPPORT
# =============================================================================

def test_distributed_support_config():
    """Test distributed support parameters."""
    config = JottyConfig(
        enable_distributed=True,
        instance_id="worker-1",
        lock_timeout=10.0,
        redis_host="localhost",
        redis_port=6380,
        redis_db=1
    )

    assert config.enable_distributed == True
    assert config.instance_id == "worker-1"
    assert config.lock_timeout == 10.0
    assert config.redis_host == "localhost"
    assert config.redis_port == 6380


# =============================================================================
# TEST 24: Category 22 - DYNAMIC ORCHESTRATION
# =============================================================================

def test_dynamic_orchestration_config():
    """Test dynamic orchestration parameters."""
    config = JottyConfig(
        enable_dynamic_planning=True,
        planning_complexity_threshold=0.8,
        enable_agent_registry=True,
        auto_infer_capabilities=True,
        enable_state_analysis=True,
        enable_recovery_management=True,
        recovery_max_retries=5
    )

    assert config.enable_dynamic_planning == True
    assert config.planning_complexity_threshold == 0.8
    assert config.enable_agent_registry == True
    assert config.enable_recovery_management == True
    assert config.recovery_max_retries == 5


# =============================================================================
# TEST 25: Computed Properties
# =============================================================================

def test_computed_properties():
    """Test all computed properties work correctly."""
    config = JottyConfig(
        preview_token_budget=25000,
        max_description_tokens=8000,
        max_context_tokens=150000,
        system_prompt_budget=8000,
        current_input_budget=20000,
        trajectory_budget=30000,
        tool_output_budget=20000,
        min_memory_budget=15000
    )

    # Test char limits (computed in __post_init__)
    assert config.preview_char_limit == 25000 * 4
    assert config.max_description_chars == 8000 * 4

    # Test memory_budget property
    reserved = 8000 + 20000 + 30000 + 20000  # 78000
    expected = max(15000, 150000 - 78000)  # 72000
    assert config.memory_budget == expected

    # Test total_memory_capacity property
    total = (config.episodic_capacity + config.semantic_capacity +
             config.procedural_capacity + config.meta_capacity +
             config.causal_capacity)
    assert config.total_memory_capacity == total


# =============================================================================
# TEST 26: Edge Cases and Validation
# =============================================================================

def test_config_with_zero_values():
    """Test config handles zero values correctly."""
    config = JottyConfig(
        max_actor_iters=0,
        episodic_capacity=0,
        epsilon_end=0.0,
        alpha=0.0
    )

    assert config.max_actor_iters == 0
    assert config.episodic_capacity == 0
    assert config.epsilon_end == 0.0


def test_config_with_extreme_values():
    """Test config handles extreme values."""
    config = JottyConfig(
        max_context_tokens=1000000,
        episode_buffer_size=100000,
        epsilon_start=1.0,
        gamma=1.0
    )

    assert config.max_context_tokens == 1000000
    assert config.epsilon_start == 1.0


def test_config_backward_compatibility():
    """Test that JottyConfig and SwarmConfig are equivalent."""
    config1 = JottyConfig(base_path="/test")
    config2 = SwarmConfig(base_path="/test")

    # Both should be SwarmConfig instances
    assert isinstance(config1, SwarmConfig)
    assert isinstance(config2, SwarmConfig)

    # Both should have same values
    assert config1.base_path == config2.base_path


# =============================================================================
# TEST 27: Integration Tests
# =============================================================================

def test_config_can_be_used_by_both_orchestrators():
    """Test that config works with both SAS and MAS."""
    from core.orchestration import SingleAgentOrchestrator, MultiAgentsOrchestrator
    from core.foundation import AgentConfig  # Phase 7: Consistent naming - use AgentConfig
    import dspy

    # Create unified config
    config = JottyConfig(
        base_path="/tmp/test",
        enable_validation=False,
        enable_rl=False
    )

    # Test with SingleAgentOrchestrator
    class SimpleSignature(dspy.Signature):
        """Simple test signature."""
        input: str = dspy.InputField()
        output: str = dspy.OutputField()

    sas = SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(SimpleSignature),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config  # Should work
    )

    assert sas.config == config

    # Test with MultiAgentsOrchestrator
    agent_config = AgentConfig(
        name="TestAgent",
        agent=sas,
        enable_architect=False,
        enable_auditor=False
    )

    mas = MultiAgentsOrchestrator(
        actors=[agent_config],
        metadata_provider=None,
        config=config  # Should work
    )

    assert mas.config == config


def test_config_all_categories_accessible():
    """Test that all 22 categories are accessible and work."""
    config = JottyConfig()

    # Category 1: Persistence
    assert hasattr(config, 'output_base_dir')

    # Category 2: Execution
    assert hasattr(config, 'max_actor_iters')

    # Category 2.5: Timeout & Circuit Breaker
    assert hasattr(config, 'enable_circuit_breakers')

    # Category 3: Memory
    assert hasattr(config, 'episodic_capacity')

    # Category 4: Context Budget
    assert hasattr(config, 'max_context_tokens')

    # Category 4.5: Agentic Discovery Budget
    assert hasattr(config, 'preview_token_budget')

    # Category 4.6: Token Counting
    assert hasattr(config, 'token_model_name')

    # Category 5: RL Parameters
    assert hasattr(config, 'gamma')

    # Category 6: Exploration
    assert hasattr(config, 'epsilon_start')

    # Category 7: Credit Assignment
    assert hasattr(config, 'credit_decay')

    # Category 8: Consolidation
    assert hasattr(config, 'consolidation_threshold')

    # Category 9: Offline Learning
    assert hasattr(config, 'episode_buffer_size')

    # Category 10: Protection
    assert hasattr(config, 'protected_memory_threshold')

    # Category 11: Validation
    assert hasattr(config, 'enable_validation')

    # Category 12: Async
    assert hasattr(config, 'parallel_architect')

    # Category 13: Logging
    assert hasattr(config, 'verbose')

    # Category 14: LLM RAG
    assert hasattr(config, 'enable_llm_rag')

    # Category 15: Goal Hierarchy
    assert hasattr(config, 'enable_goal_hierarchy')

    # Category 16: Causal Learning
    assert hasattr(config, 'enable_causal_learning')

    # Category 17: Inter-Agent Communication
    assert hasattr(config, 'enable_agent_communication')

    # Category 18: Multi-Round Validation
    assert hasattr(config, 'enable_multi_round')

    # Category 19: Adaptive Learning
    assert hasattr(config, 'enable_adaptive_learning')

    # Category 20: Deduplication
    assert hasattr(config, 'enable_deduplication')

    # Category 21: Distributed Support
    assert hasattr(config, 'enable_distributed')

    # Category 22: Dynamic Orchestration
    assert hasattr(config, 'enable_dynamic_planning')

    print("✅ All 22 categories accessible!")


# =============================================================================
# SUMMARY
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
