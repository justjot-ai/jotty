"""
Base Expert Agent Framework

 DEPRECATED (Phase 8): Use SingleAgentOrchestrator with enable_gold_standard_learning=True instead.

Expert agents are specialized agents that use OptimizationPipeline to ensure
they always produce correct outputs. They can be pre-trained and validated.

Migration Guide:
    Old (deprecated):
        from Jotty.core.experts import ExpertAgent, ExpertAgentConfig
        config = ExpertAgentConfig(name="Expert", domain="mermaid")
        expert = ExpertAgent(config)

    New (recommended):
        from Jotty.core.experts.expert_templates import create_mermaid_expert
        expert = create_mermaid_expert(config=SwarmConfig())

    Or custom:
        from Jotty.core.orchestration import SingleAgentOrchestrator
        expert = SingleAgentOrchestrator(
            agent=my_agent,
            enable_gold_standard_learning=True,
            gold_standards=[...],
            domain="my_domain"
        )

This class is kept for backward compatibility only.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field

from ..orchestration import (
    create_optimization_pipeline,
    OptimizationPipeline,
    OptimizationConfig
)
from ..foundation.agent_config import AgentConfig
from ..memory.cortex import SwarmMemory
from ..foundation.data_structures import MemoryLevel

logger = logging.getLogger(__name__)


@dataclass
class ExpertAgentConfig:
    """Configuration for an Expert Agent."""
    name: str
    domain: str  # e.g., "mermaid", "pipeline", "plantuml"
    description: str
    
    # Training configuration
    training_gold_standards: Optional[List[Dict[str, Any]]] = None
    max_training_iterations: int = 5
    required_training_pass_count: int = 1
    
    # Validation configuration
    validation_cases: Optional[List[Dict[str, Any]]] = None
    min_validation_score: float = 1.0
    
    # Optimization pipeline config
    enable_teacher_model: bool = True
    enable_kb_updates: bool = False
    save_improvements: bool = True
    
    # Storage
    expert_data_dir: Optional[str] = None
    use_memory_storage: bool = True  # Use Jotty memory system instead of files
    use_memory_synthesis: bool = False  # Use memory synthesis for consolidated improvements
    
    # Custom evaluation function
    evaluation_function: Optional[Callable] = None
    
    # Agent configuration
    agent_module: Optional[Any] = None  # The underlying DSPy module/agent
    teacher_module: Optional[Any] = None  # Optional custom teacher
    
    def __post_init__(self) -> None:
        """Initialize defaults."""
        if self.expert_data_dir is None:
            self.expert_data_dir = f"./expert_data/{self.domain}"


class ExpertAgent:
    """
    Base class for Expert Agents.

     DEPRECATED (Phase 8): Use SingleAgentOrchestrator with enable_gold_standard_learning=True
    or use expert_templates.py factory functions instead.

    Expert agents use OptimizationPipeline to ensure they always produce
    correct outputs. They can be pre-trained and validated.
    """

    def __init__(self, config: ExpertAgentConfig, memory: Optional[SwarmMemory] = None) -> None:
        import warnings
        warnings.warn(
            "ExpertAgent is deprecated. Use SingleAgentOrchestrator with "
            "enable_gold_standard_learning=True or expert_templates factory functions instead. "
            "See docs/PHASE_8_EXPERT_INTEGRATION_PROPOSAL.md for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )

        self.config = config
        self.data_dir = Path(config.expert_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory system integration
        self.memory = memory  # Use provided memory or create new one
        self.use_memory_storage = config.use_memory_storage and memory is not None
        
        # Create memory system if not provided but enabled
        if config.use_memory_storage and not memory:
            try:
                from ..memory.cortex import SwarmMemory
                from ..foundation.data_structures import SwarmConfig
                from ..memory.memory_persistence import enable_memory_persistence
                
                memory_config = SwarmConfig()
                self.memory = SwarmMemory(
                    agent_name=f"expert_{config.domain}",
                    config=memory_config
                )
                
                # Enable persistence
                persistence_dir = self.data_dir / "memory"
                self.memory_persistence = enable_memory_persistence(
                    memory=self.memory,
                    persistence_dir=persistence_dir
                )
                
                self.use_memory_storage = True
                logger.info(f"Created SwarmMemory for expert {config.name} with persistence at {persistence_dir}")
            except Exception as e:
                logger.warning(f"Could not create memory system: {e}")
                self.use_memory_storage = False
                self.memory_persistence = None
        else:
            self.memory_persistence = None
        
        self.trained = False
        self.validation_passed = False
        self.improvements_file = self.data_dir / "improvements.json"
        self.training_results_file = self.data_dir / "training_results.json"
        self.validation_results_file = self.data_dir / "validation_results.json"
        
        # Load existing improvements (from memory or file)
        # Use synthesis if configured
        self.improvements = self._load_improvements(use_synthesis=config.use_memory_synthesis)
        
        # Internal optimization pipeline (created during training)
        self._optimization_pipeline: Optional[OptimizationPipeline] = None
    
    def _load_improvements(self, use_synthesis: bool = False) -> List[Dict[str, Any]]:
        """
        Load previously learned improvements from memory or file.
        
        Args:
            use_synthesis: If True, use memory synthesis to get consolidated improvements
                          instead of raw improvements. This provides summarized patterns.
        
        Returns:
            List of improvement dictionaries
        """
        improvements = []
        
        # Try memory system first
        if self.use_memory_storage and self.memory:
            try:
                if use_synthesis:
                    # Use synthesis to get consolidated improvements
                    from .memory_integration import retrieve_synthesized_improvements
                    synthesized = retrieve_synthesized_improvements(
                        memory=self.memory,
                        expert_name=self.config.name,
                        domain=self.config.domain
                    )
                    
                    if synthesized:
                        # Convert synthesized text to improvement format
                        improvements.append({
                            "timestamp": datetime.now().isoformat(),
                            "task": "Consolidated patterns",
                            "learned_pattern": synthesized,
                            "source": "memory_synthesis",
                            "memory_level": "semantic",
                            "is_synthesized": True
                        })
                        logger.info(f"Loaded synthesized improvements from memory (length: {len(synthesized)} chars)")
                else:
                    # Retrieve raw improvements from memory (PROCEDURAL or META level)
                    memory_entries = self.memory.retrieve(
                        query=f"expert agent improvements {self.config.domain} {self.config.name}",
                        goal=f"expert_{self.config.domain}_improvements",
                        budget_tokens=10000,
                        levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC]
                    )
                    
                    # Convert memory entries to improvement format
                    for entry in memory_entries:
                        try:
                            # Try to parse as JSON if stored as JSON
                            improvement_data = json.loads(entry.content)
                            if isinstance(improvement_data, dict):
                                improvements.append(improvement_data)
                            elif isinstance(improvement_data, list):
                                improvements.extend(improvement_data)
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, create improvement from memory content
                            improvements.append({
                                "timestamp": entry.last_accessed.isoformat() if entry.last_accessed else datetime.now().isoformat(),
                                "task": entry.context.get('task', 'Unknown'),
                                "learned_pattern": entry.content,
                                "source": "memory",
                                "memory_level": entry.level.value,
                                "memory_key": entry.key
                            })
                    
                    logger.info(f"Loaded {len(improvements)} improvements from memory system")
            except Exception as e:
                logger.warning(f"Failed to load improvements from memory: {e}")
        
        # Fallback to file if memory not available or empty
        if not improvements and self.improvements_file.exists():
            try:
                with open(self.improvements_file, 'r') as f:
                    file_improvements = json.load(f)
                    improvements.extend(file_improvements)
                    logger.info(f"Loaded {len(file_improvements)} improvements from file")
            except Exception as e:
                logger.warning(f"Failed to load improvements from file: {e}")
        
        return improvements
    
    async def train(
        self,
        gold_standards: Optional[List[Dict[str, Any]]] = None,
        force_retrain: bool = False,
        enable_pre_training: bool = True,
        training_mode: str = "both"  # "iterative", "pattern_extraction", "both"
    ) -> Dict[str, Any]:
        """
        Train the expert agent on gold standards.
        
        Args:
            gold_standards: List of training cases with 'task', 'context', 'gold_standard'
                Can be used for both pre-training (pattern extraction) and iterative learning.
            force_retrain: If True, retrain even if already trained
            enable_pre_training: If True, extract patterns from gold_standards before iterative learning
            training_mode: Training mode - "iterative" (learn from mistakes), 
                          "pattern_extraction" (extract patterns only), or "both"
        
        Returns:
            Training results dictionary
        
        Use Cases for gold_standards:
        1. Iterative Learning: Learn from mistakes via optimization pipeline (default)
        2. Pattern Extraction: Extract patterns before iterative learning (pre-training)
        3. Validation: Verify expert performance after training
        4. Few-Shot Learning: Use as examples in context for generation
        5. Template Learning: Learn common structures/patterns
        6. Domain Adaptation: Adapt to new domains using examples
        """
        if self.trained and not force_retrain:
            logger.info(f"Expert {self.config.name} already trained. Use force_retrain=True to retrain.")
            return self._load_training_results()
        
        gold_standards = gold_standards or self.config.training_gold_standards
        if not gold_standards:
            raise ValueError("No gold standards provided for training")
        
        # Phase 1: Pre-Training (Pattern Extraction from gold_standards)
        if enable_pre_training and training_mode in ["pattern_extraction", "both"]:
            logger.info(f"Phase 1: Pre-training on {len(gold_standards)} gold standards")
            pre_training_results = await self._pre_train_from_gold_standards(gold_standards)
            logger.info(f"Pre-training complete: {pre_training_results.get('patterns_learned', 0)} patterns learned")
        
        # Phase 2: Fine-Tuning from Mistakes (Iterative Learning)
        if training_mode in ["iterative", "both"]:
            logger.info(f"Phase 2: Fine-tuning expert agent: {self.config.name}")
            
            # Create agents for optimization pipeline
            agents = self._create_agents()
            
            # Create optimization pipeline
            pipeline = create_optimization_pipeline(
                agents=agents,
                max_iterations=self.config.max_training_iterations,
                required_pass_count=self.config.required_training_pass_count,
                enable_teacher_model=self.config.enable_teacher_model,
                enable_kb_updates=self.config.enable_kb_updates,
                save_improvements=self.config.save_improvements,
                output_path=str(self.data_dir)
            )
            
            # Pass expert's memory to pipeline if available
            # This ensures improvements are stored to expert's memory directly
            if self.use_memory_storage and self.memory:
                # Store memory reference for pipeline to use
                pipeline.expert_memory = self.memory
                pipeline.expert_name = self.config.name
                pipeline.expert_domain = self.config.domain
            
            if self.config.evaluation_function:
                pipeline.config.evaluation_function = self.config.evaluation_function
            
            self._optimization_pipeline = pipeline
            
            # Train on each gold standard
            training_results = {
                "expert_name": self.config.name,
                "domain": self.config.domain,
                "training_cases": [],
                "overall_success": True,
                "total_cases": len(gold_standards),
                "passed_cases": 0
            }
            
            for i, case in enumerate(gold_standards, 1):
                logger.info(f"Training case {i}/{len(gold_standards)}: {case.get('task', 'Unknown')}")
                
                result = await pipeline.optimize(
                    task=case.get('task', ''),
                    context=case.get('context', {}),
                    gold_standard=case.get('gold_standard', '')
                )
                
                case_result = {
                    "case_number": i,
                    "task": case.get('task', ''),
                    "success": result.get('optimization_complete', False),
                    "iterations": result.get('total_iterations', 0),
                    "final_score": result.get('final_result', {}).get('evaluation_score', 0.0)
                }
                
                training_results["training_cases"].append(case_result)
                
                if case_result["success"]:
                    training_results["passed_cases"] += 1
                else:
                    training_results["overall_success"] = False
            
            # Save training results
            with open(self.training_results_file, 'w') as f:
                json.dump(training_results, f, indent=2)
            
            # Mark as trained if at least some cases passed OR if improvements were learned
            has_passed_cases = training_results["passed_cases"] > 0
            has_improvements = len(self.improvements) > 0
        else:
            # Pattern extraction only - no iterative learning
            training_results = {
                "expert_name": self.config.name,
                "domain": self.config.domain,
                "training_cases": [],
                "overall_success": True,
                "total_cases": len(gold_standards),
                "passed_cases": 0,
                "pattern_extraction_only": True
            }
            
            # Save training results
            with open(self.training_results_file, 'w') as f:
                json.dump(training_results, f, indent=2)
            
            # Mark as trained if patterns were learned
            has_passed_cases = False
            has_improvements = len(self.improvements) > 0
        
        self.trained = has_passed_cases or has_improvements
        
        if self.trained:
            logger.info(f"Expert {self.config.name} marked as trained (passed: {has_passed_cases}, improvements: {has_improvements})")
        
        # Reload improvements from memory after training
        self.improvements = self._load_improvements(use_synthesis=self.config.use_memory_synthesis)
        
        # Run consolidation if we have enough improvements
        if self.use_memory_storage and self.memory and len(self.improvements) >= 3:
            try:
                from .memory_integration import consolidate_improvements
                logger.info(f"Running consolidation for {len(self.improvements)} improvements...")
                consolidation_result = consolidate_improvements(
                    memory=self.memory,
                    expert_name=self.config.name,
                    domain=self.config.domain
                )
                if consolidation_result.get('consolidated', 0) > 0:
                    logger.info(f"Consolidated {consolidation_result['consolidated']} patterns to SEMANTIC level")
                    # Reload improvements to include consolidated ones
                    self.improvements = self._load_improvements(use_synthesis=False)
            except Exception as e:
                logger.warning(f"Consolidation failed: {e}")
        
        logger.info(f"Training complete: {training_results['passed_cases']}/{training_results['total_cases']} cases passed")
        
        # Save memory to disk if persistence enabled
        if self.memory_persistence:
            try:
                self.memory_persistence.save()
                logger.info(f"Saved memory to disk: {self.memory_persistence.persistence_dir}")
            except Exception as e:
                logger.warning(f"Failed to save memory: {e}")
        
        return training_results
    
    async def validate(
        self,
        validation_cases: Optional[List[Dict[str, Any]]] = None,
        skip_if_not_trained: bool = False
    ) -> Dict[str, Any]:
        """
        Validate the expert agent on test cases.
        
        Args:
            validation_cases: List of validation cases with 'task', 'context', 'gold_standard'
            skip_if_not_trained: If True, skip validation if not trained (instead of raising error)
        
        Returns:
            Validation results dictionary
        """
        if not self.trained:
            if skip_if_not_trained:
                return {"validated": False, "reason": "Expert not trained yet"}
            raise RuntimeError(f"Expert {self.config.name} must be trained before validation")
        
        validation_cases = validation_cases or self.config.validation_cases
        if not validation_cases:
            logger.warning("No validation cases provided, skipping validation")
            return {"validated": False, "reason": "No validation cases"}
        
        logger.info(f"Validating expert agent: {self.config.name}")
        
        validation_results = {
            "expert_name": self.config.name,
            "validation_cases": [],
            "overall_pass": True,
            "total_cases": len(validation_cases),
            "passed_cases": 0,
            "average_score": 0.0
        }
        
        scores = []
        
        for i, case in enumerate(validation_cases, 1):
            logger.info(f"Validation case {i}/{len(validation_cases)}: {case.get('task', 'Unknown')}")
            
            output = await self.generate(
                task=case.get('task', ''),
                context=case.get('context', {})
            )
            
            # Evaluate output
            if self.config.evaluation_function:
                eval_result = await self.config.evaluation_function(
                    output=output,
                    gold_standard=case.get('gold_standard', ''),
                    task=case.get('task', ''),
                    context=case.get('context', {})
                )
                score = eval_result.get('score', 0.0)
            else:
                # Simple string comparison
                score = 1.0 if str(output).strip() == str(case.get('gold_standard', '')).strip() else 0.0
            
            scores.append(score)
            
            case_result = {
                "case_number": i,
                "task": case.get('task', ''),
                "score": score,
                "passed": score >= self.config.min_validation_score
            }
            
            validation_results["validation_cases"].append(case_result)
            
            if case_result["passed"]:
                validation_results["passed_cases"] += 1
            else:
                validation_results["overall_pass"] = False
        
        validation_results["average_score"] = sum(scores) / len(scores) if scores else 0.0
        
        # Save validation results
        with open(self.validation_results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        self.validation_passed = validation_results["overall_pass"]
        
        logger.info(f"Validation complete: {validation_results['passed_cases']}/{validation_results['total_cases']} cases passed")
        
        return validation_results
    
    async def _pre_train_from_gold_standards(
        self,
        gold_standards: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Pre-train on gold standards by extracting patterns.
        
        Converts gold_standards format to examples format and extracts patterns.
        
        Args:
            gold_standards: List of training cases with 'task', 'context', 'gold_standard'
        
        Returns:
            Pre-training results dictionary
        """
        # Convert gold_standards to examples format
        examples = []
        for case in gold_standards:
            gold_standard = case.get('gold_standard', '')
            task = case.get('task', '')
            context = case.get('context', {})
            description = context.get('description', task)
            
            # Extract diagram type from context or infer from gold_standard
            diagram_type = context.get('diagram_type', 'unknown')
            if diagram_type == 'unknown':
                # Try to infer from gold_standard
                gold_lower = str(gold_standard).lower()
                if 'gitgraph' in gold_lower or 'git graph' in gold_lower:
                    diagram_type = 'gitGraph'
                elif 'sequence' in gold_lower:
                    diagram_type = 'sequenceDiagram'
                elif 'state' in gold_lower:
                    diagram_type = 'stateDiagram-v2'
                elif 'gantt' in gold_lower:
                    diagram_type = 'gantt'
                elif 'er' in gold_lower or 'entity' in gold_lower:
                    diagram_type = 'erDiagram'
                elif 'journey' in gold_lower:
                    diagram_type = 'journey'
                elif 'class' in gold_lower:
                    diagram_type = 'classDiagram'
                elif 'graph' in gold_lower or 'flowchart' in gold_lower:
                    diagram_type = 'flowchart'
            
            examples.append({
                "code": str(gold_standard),
                "description": description,
                "type": diagram_type,
                "source": "gold_standard",
                "task": task
            })
        
        return await self._pre_train(examples)
    
    async def _pre_train(
        self,
        examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Pre-train on curated examples before learning from mistakes.
        
        Extracts patterns from examples and stores as initial improvements.
        
        Args:
            examples: List of training examples with 'code', 'description', 'type'
        
        Returns:
            Pre-training results dictionary
        """
        logger.info(f"Pre-training on {len(examples)} examples")
        
        # Validate examples using domain validator
        validated_examples = []
        if self.config.evaluation_function:
            # Use evaluation function to validate
            for example in examples:
                code = example.get('code', '')
                description = example.get('description', '')
                diagram_type = example.get('type', 'unknown')
                
                # Validate using domain validator if available
                try:
                    from .domain_validators import get_validator
                    validator = get_validator(self.config.domain)
                    
                    if validator:
                        is_valid, error, metadata = validator.validate(
                            output=code,
                            expected_type=diagram_type,
                            context={"task": description}
                        )
                        
                        if is_valid:
                            validated_examples.append(example)
                        else:
                            logger.debug(f"Skipping invalid example: {error}")
                except Exception as e:
                    logger.warning(f"Error validating example: {e}")
                    # Include anyway if validation fails
                    validated_examples.append(example)
        else:
            validated_examples = examples
        
        # Extract patterns from examples
        patterns_learned = 0
        for example in validated_examples:
            code = example.get('code', '')
            description = example.get('description', '')
            diagram_type = example.get('type', 'unknown')
            
            # Extract pattern (simplified - could be enhanced with LLM)
            pattern = self._extract_pattern_from_example(code, description, diagram_type)
            
            if pattern:
                improvement = {
                    "iteration": 0,  # Pre-training iteration
                    "timestamp": datetime.now().isoformat(),
                    "task": description,
                    "learned_pattern": pattern,
                    "improvement_type": "pre_training",
                    "source": example.get('source', 'curated'),
                    "example_code": code[:200]  # Store snippet for reference
                }
                
                # Store improvement
                self.improvements.append(improvement)
                
                # Store to memory if available
                if self.use_memory_storage and self.memory:
                    try:
                        from .memory_integration import store_improvement_to_memory
                        store_improvement_to_memory(
                            memory=self.memory,
                            improvement=improvement,
                            expert_name=self.config.name,
                            domain=self.config.domain
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store pre-training improvement to memory: {e}")
                
                patterns_learned += 1
        
        logger.info(f"Pre-training extracted {patterns_learned} patterns from {len(validated_examples)} examples")
        
        return {
            "patterns_learned": patterns_learned,
            "examples_processed": len(examples),
            "examples_validated": len(validated_examples)
        }
    
    def _extract_pattern_from_example(
        self,
        code: str,
        description: str,
        diagram_type: str
    ) -> Optional[str]:
        """
        Extract a learning pattern from an example.
        
        This is a simplified extraction - could be enhanced with LLM analysis.
        
        Args:
            code: Example code
            description: Description of example
            diagram_type: Type of diagram
        
        Returns:
            Pattern string or None
        """
        # Simple pattern extraction based on domain
        if self.config.domain == "plantuml":
            # Check for common patterns
            if '@startuml' in code and '@enduml' in code:
                return f"When generating {diagram_type} diagrams, always include @startuml/@enduml tags"
            
            # Extract key elements
            if 'sequence' in code.lower():
                return f"For sequence diagrams, use participant declarations and proper message syntax"
        
        elif self.config.domain == "mermaid":
            # Check for diagram type
            first_line = code.split('\n')[0].strip().lower()
            if 'gitgraph' in first_line:
                return f"When task mentions git flow or branching, use gitGraph syntax, not graph"
            
            if 'sequence' in first_line:
                return f"For sequence diagrams, use sequenceDiagram syntax with participants"
        
        # Generic pattern
        return f"When generating {diagram_type} diagrams, follow the structure and syntax patterns from examples"
    
    async def generate(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Generate output using the expert agent.
        
        Training is OPTIONAL - any LLM can generate without training.
        Training is for OPTIMIZATION (better quality, correctness, learned patterns).
        
        Args:
            task: Task description
            context: Optional context dictionary
        
        Returns:
            Generated output
        """
        # Training is optional - expert can generate without training
        # Training is for optimization (better quality, correctness, learned patterns)
        if not self.trained:
            logger.info(
                f"Expert {self.config.name} not trained - using base agent. "
                f"Training is optional (for optimization via OptimizationPipeline)"
            )
        
        # Use the agent (trained or untrained)
        agents = self._create_agents()
        
        # Get the main agent (not teacher)
        main_agent_config = agents[0]
        agent = main_agent_config.agent
        
        # Apply learned improvements to DSPy module
        if self.improvements:
            try:
                from .dspy_improvements import apply_improvements_to_dspy_module
                apply_improvements_to_dspy_module(agent, self.improvements)
            except ImportError:
                logger.warning("Could not import dspy_improvements, using string context")
        
        # Prioritize improvements using credit assignment if available
        improvements_to_use = self.improvements
        if improvements_to_use and hasattr(self, '_optimization_pipeline') and self._optimization_pipeline:
            if hasattr(self._optimization_pipeline, 'credit_assignment') and self._optimization_pipeline.credit_assignment:
                try:
                    improvements_to_use = self._optimization_pipeline.credit_assignment.prioritize_improvements(
                        improvements=self.improvements,
                        max_improvements=10,  # Use top 10 prioritized
                        min_credit_threshold=0.1
                    )
                    logger.debug(f"Using {len(improvements_to_use)} prioritized improvements (from {len(self.improvements)} total)")
                except Exception as e:
                    logger.warning(f"Error prioritizing improvements: {e}, using all improvements")
                    improvements_to_use = self.improvements
        
        # Apply domain-specific validation if available
        if improvements_to_use:
            improvements_to_use = self._apply_domain_specific_improvements(improvements_to_use, context)
        
        # Also pass improvements as context string for DSPy input field
        improvements_str = ""
        if improvements_to_use:
            # Format improvements as string for DSPy input
            improvements_str = "\n".join([
                f"- {imp.get('learned_pattern', '')}" 
                for imp in improvements_to_use[-10:]  # Use top prioritized improvements
            ])
        
        # Prepare context for DSPy agent
        context = context or {}
        
        # Call agent using base class helper (handles DSPy and regular)
        result = self._call_dspy_agent(
            agent,
            task=task,
            learned_improvements=improvements_str,
            **context
        )
        
        # Extract output using base class helper
        result_output = self._extract_dspy_output(result)
        
        # Apply domain-specific post-processing
        result_output = self._apply_domain_specific_post_processing(result_output, context)
        
        return result_output
    
    def _apply_domain_specific_improvements(
        self,
        improvements: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply domain-specific filtering/enhancement to improvements.
        
        Args:
            improvements: List of improvements
            context: Context dictionary
        
        Returns:
            Filtered/enhanced improvements
        """
        # Filter improvements by domain relevance
        domain = self.config.domain.lower()
        filtered = []
        
        for imp in improvements:
            pattern = imp.get('learned_pattern', '').lower()
            task = imp.get('task', '').lower()
            
            # Keep if pattern mentions domain or task is relevant
            if domain in pattern or domain in task or not pattern:
                filtered.append(imp)
        
        return filtered if filtered else improvements
    
    def _apply_domain_specific_post_processing(
        self,
        output: Any,
        context: Dict[str, Any]
    ) -> Any:
        """
        Apply domain-specific post-processing to output.
        
        Args:
            output: Generated output
            context: Context dictionary
        
        Returns:
            Post-processed output
        """
        # Domain-specific post-processing can be overridden by subclasses
        return output
    
    def _create_agents(self) -> List[AgentConfig]:
        """Create agent configurations for optimization pipeline."""
        agents = []
        
        # Main agent
        if self.config.agent_module:
            # Pass improvements if agent_module accepts them
            try:
                agent = self.config.agent_module(improvements=self.improvements)
            except TypeError:
                # Fallback if agent_module doesn't accept improvements
                agent = self.config.agent_module()
            agents.append(AgentConfig(
                name=f"{self.config.domain}_generator",
                agent=agent,
                outputs=["output"]
            ))
        else:
            # Use default agent (to be overridden by subclasses)
            # Pass improvements to default agent creation
            agent = self._create_default_agent(improvements=self.improvements)
            agents.append(AgentConfig(
                name=f"{self.config.domain}_generator",
                agent=agent,
                outputs=["output"]
            ))
        
        # Teacher agent
        if self.config.enable_teacher_model:
            if self.config.teacher_module:
                teacher_agent = self.config.teacher_module()
            else:
                teacher_agent = self._create_default_teacher()
            
            agents.append(AgentConfig(
                name="teacher",
                agent=teacher_agent,
                metadata={"is_teacher": True}
            ))
        
        return agents
    
    def _create_default_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create default agent (to be overridden by subclasses).
        
        Args:
            improvements: Optional list of improvements to use when creating agent
        """
        raise NotImplementedError("Subclasses must implement _create_default_agent")
    
    def _create_default_teacher(self) -> Any:
        """Create default teacher agent (supports both DSPy and regular)."""
        try:
            import dspy
            # Use DSPy for teacher if available
            class TeacherSignature(dspy.Signature):
                """Provide the correct output based on gold standard."""
                task: str = dspy.InputField(desc="Task description")
                gold_standard: str = dspy.InputField(desc="The correct output to return")
                student_output: str = dspy.InputField(desc="What the student generated")
                
                output: str = dspy.OutputField(desc="The correct output (must match gold_standard)")
            
            return dspy.Predict(TeacherSignature)
        except ImportError:
            # Fallback to regular teacher
            class DefaultTeacher:
                def forward(self, **kwargs: Any) -> Any:
                    result = type('Result', (), {})()
                    result._store = {"output": kwargs.get('gold_standard', '')}
                    return result
            
            return DefaultTeacher()
    
    def _is_dspy_module(self, agent: Any) -> bool:
        """Check if agent is a DSPy module."""
        try:
            import dspy
            return isinstance(agent, dspy.Module)
        except ImportError:
            return False
    
    def _call_dspy_agent(self, agent: Any, **kwargs: Any) -> Any:
        """Call a DSPy agent correctly (handles both DSPy modules and regular callables)."""
        try:
            import dspy
            if isinstance(agent, dspy.Module):
                # DSPy module - call directly (not .forward())
                return agent(**kwargs)
        except ImportError:
            pass
        
        # Regular callable - try forward() or direct call
        if hasattr(agent, 'forward'):
            return agent.forward(**kwargs)
        else:
            return agent(**kwargs)
    
    def _extract_dspy_output(self, result: Any) -> Any:
        """Extract output from DSPy Prediction or regular result."""
        try:
            import dspy
            # DSPy Prediction has 'output' attribute
            if isinstance(result, dspy.Prediction) or (hasattr(result, 'output') and not isinstance(result, dict)):
                return result.output
        except ImportError:
            pass
        
        # Handle _store pattern
        if hasattr(result, '_store') and isinstance(result._store, dict):
            return result._store.get('output', result)
        
        # Handle dict with output key
        if isinstance(result, dict):
            return result.get('output', result)
        
        return result
    
    def _load_training_results(self) -> Dict[str, Any]:
        """Load training results from file."""
        if self.training_results_file.exists():
            try:
                with open(self.training_results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load training results: {e}")
        return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the expert agent."""
        return {
            "name": self.config.name,
            "domain": self.config.domain,
            "trained": self.trained,
            "validation_passed": self.validation_passed,
            "improvements_count": len(self.improvements),
            "data_dir": str(self.data_dir)
        }
