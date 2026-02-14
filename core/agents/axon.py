"""
SmartAgentSlack: Intelligent Agent Communication Layer

CRITICAL: This is NOT just a message bus. It's a SMART layer that:
1. Knows what format each agent wants (from signatures)
2. Auto-transforms data (using internal Transformer)
3. Auto-chunks huge files (using internal Chunker)
4. Auto-compresses context (using internal Compressor)
5. Manages message history
6. Handles ALL communication nuances

User Insight: "It's a SMART Slack that knows what format and type 
the other agent is comfortable with"

Author: A-Team (Turing, Nash, Sutton, Chomsky, Modern AI Engineers)
Date: December 27, 2025
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import dspy
import json
import sys
import os

from ..utils.tokenizer import SmartTokenizer

logger = logging.getLogger(__name__)


@dataclass
class AgentCapabilities:
    """
    What an agent can handle (extracted from signature automatically).
    
    Agent Slack reads the agent's DSPy signature to understand:
    - What formats it prefers
    - What size limits it has
    - What context budget it has
    
    ZERO manual configuration needed!
    """
    agent_name: str
    preferred_format: str
    acceptable_formats: List[str]
    max_input_size: int  # bytes
    max_context_tokens: int
    
    def can_accept_format(self, format: str) -> bool:
        """Check if agent can accept this format."""
        return format in self.acceptable_formats
    
    def needs_compression(self, current_tokens: int) -> bool:
        """Check if context needs compression (70% threshold)."""
        return current_tokens > (self.max_context_tokens * 0.7)


@dataclass
class Message:
    """Message passed between agents."""
    from_agent: str
    to_agent: str
    data: Any
    format: str
    size_bytes: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"Message({self.from_agent} → {self.to_agent}, {self.format}, {self.size_bytes}B)"


class MessageBus:
    """Simple pub/sub message bus."""
    
    def __init__(self):
        self.subscribers: Dict[str, Callable] = {}
        self.message_count = 0
        logger.info(" [MESSAGE BUS] Initialized")
        
    def subscribe(self, agent_name: str, callback: Callable) -> None:
        """Subscribe agent to messages."""
        self.subscribers[agent_name] = callback
        logger.debug(f" [MESSAGE BUS] Subscribed: {agent_name}")
        
    def publish(self, message: Message):
        """Publish message to target agent."""
        self.message_count += 1
        target = message.to_agent
        
        if target not in self.subscribers:
            logger.error(f" [MESSAGE BUS] No subscriber: {target}")
            return False
            
        try:
            callback = self.subscribers[target]
            callback(message)
            logger.debug(f" [MESSAGE BUS] Delivered #{self.message_count}: {message}")
            return True
        except Exception as e:
            logger.error(f" [MESSAGE BUS] Delivery failed: {e}")
            return False


class FormatRegistry:
    """Tracks format preferences for all agents."""
    
    def __init__(self):
        self.registry: Dict[str, AgentCapabilities] = {}
        logger.info(" [FORMAT REGISTRY] Initialized")
        
    def register(self, agent_name: str, capabilities: AgentCapabilities) -> None:
        """Register agent's capabilities."""
        self.registry[agent_name] = capabilities
        logger.debug(f" [FORMAT REGISTRY] Registered: {agent_name}")
        
    def get_capabilities(self, agent_name: str) -> Optional[AgentCapabilities]:
        """Get agent's capabilities."""
        return self.registry.get(agent_name)


class SmartAgentSlack:
    """
    INTELLIGENT agent communication layer with embedded helper agents.
    
    This is the KEY component for inter-agent communication. It:
    1. Auto-detects format needs from agent signatures
    2. Has Transformer agent internally (for format conversion)
    3. Has Chunker agent internally (for large file handling)
    4. Has Compressor agent internally (for context management)
    5. Manages message history and context budgets
    6. Handles ALL communication nuances automatically
    7. 🆕 Tracks cooperation events for credit assignment
    8. 🆕 Enables Nash Equilibrium communication decisions
    
    Agents just send/receive - Agent Slack handles the rest!
    """
    
    def __init__(self, config: Optional[Dict] = None, enable_cooperation: bool = True):
        """
        Initialize SmartAgentSlack with embedded helper agents.
        
        Args:
            config: Optional config dict with LM settings for helper agents
            enable_cooperation: Enable cooperation tracking (default: True)
        """
        # Core infrastructure
        self.message_bus = MessageBus()
        self.format_registry = FormatRegistry()
        
        # Message history (for context management & compression)
        self.message_history: List[Message] = []
        self.context_budget_tracker: Dict[str, int] = {}  # agent_name -> current_tokens
        
        # Agent capabilities registry
        self.agent_capabilities: Dict[str, AgentCapabilities] = {}
        
        # CRITICAL: Agent Slack HAS helper agents internally!
        # These are initialized lazily when first needed
        self._transformer = None
        self._chunker = None
        self._compressor = None
        self._config = config or {}
        
        # 🆕 COOPERATION TRACKING
        self.enable_cooperation = enable_cooperation
        self.cooperation_events: List[Dict] = []  # Track all cooperation events
        self.help_matrix: Dict[Tuple[str, str], int] = {}  # (from, to) -> count
        
        logger.info(" [SMART AGENT SLACK] Initialized")
        logger.info(" Core: MessageBus + FormatRegistry ready")
        logger.info(" Helpers: Transformer, Chunker, Compressor (lazy-init)")
        if enable_cooperation:
            logger.info(" Cooperation: Tracking enabled")
        
    @property
    def transformer(self):
        """Lazy-init Transformer agent."""
        if self._transformer is None:
            logger.info(" [SMART AGENT SLACK] Lazy-initializing Transformer...")
            from Jotty.core.smart_data_transformer import SmartDataTransformer
            # Get LM from config or use global DSPy LM
            lm = self._config.get('lm') if self._config else None
            if lm is None:
                try:
                    import dspy
                    lm = dspy.settings.lm
                except (AttributeError, TypeError) as e:
                    logger.debug(f"Could not get LM from dspy.settings: {e}")
                    pass
            self._transformer = SmartDataTransformer(lm=lm)
            logger.info(" [HELPER] Transformer initialized")
        return self._transformer
    
    @property
    def chunker(self):
        """Lazy-init Chunker agent."""
        if self._chunker is None:
            logger.info(" [SMART AGENT SLACK] Lazy-initializing Chunker...")
            from Jotty.core.context.chunker import ContextChunker
            # Get LM from config or use global DSPy LM
            lm = self._config.get('lm') if self._config else None
            if lm is None:
                try:
                    import dspy
                    lm = dspy.settings.lm
                except (AttributeError, TypeError) as e:
                    logger.debug(f"Could not get LM from dspy.settings: {e}")
                    pass
            self._chunker = ContextChunker(lm=lm)
            logger.info(" [HELPER] Chunker initialized")
        return self._chunker
    
    @property
    def compressor(self):
        """Lazy-init Compressor agent."""
        if self._compressor is None:
            logger.info(" [SMART AGENT SLACK] Lazy-initializing Compressor...")
            from Jotty.core.agentic_compressor import AgenticCompressor
            # Get LM from config or use global DSPy LM
            lm = self._config.get('lm') if self._config else None
            if lm is None:
                try:
                    import dspy
                    lm = dspy.settings.lm
                except (AttributeError, TypeError) as e:
                    logger.debug(f"Could not get LM from dspy.settings: {e}")
                    pass
            self._compressor = AgenticCompressor(lm=lm)
            logger.info(" [HELPER] Compressor initialized")
        return self._compressor
        
    def register_agent(
        self,
        agent_name: str,
        signature: Optional[dspy.Signature] = None,
        max_context: int = 16000,
        callback: Optional[Callable] = None,
        capabilities: Optional[AgentCapabilities] = None
    ):
        """
        Register agent and AUTOMATICALLY extract capabilities from signature.
        
        This is the KEY: Agent Slack READS the signature to understand:
        - What formats agent accepts
        - What size limits agent has
        - What data types agent expects
        - ALL automatically from signature!
        
        Args:
            agent_name: Name of the agent
            signature: DSPy signature (will extract capabilities from it)
            max_context: Max context window in tokens
            callback: Callback for receiving messages
            capabilities: Explicitly provided capabilities (if signature not available)
        """
        logger.info(f" [AGENT SLACK] Registering '{agent_name}'")
        
        # Extract or use provided capabilities
        if capabilities:
            caps = capabilities
        elif signature:
            caps = self._extract_capabilities_from_signature(
                agent_name=agent_name,
                signature=signature,
                max_context=max_context
            )
        else:
            # Default capabilities
            caps = AgentCapabilities(
                agent_name=agent_name,
                preferred_format="dict",
                acceptable_formats=["dict", "str"],
                max_input_size=max_context * 4,  # Rough: 1 token ≈ 4 bytes
                max_context_tokens=max_context
            )
        
        self.agent_capabilities[agent_name] = caps
        self.context_budget_tracker[agent_name] = 0
        self.format_registry.register(agent_name, caps)
        
        if callback:
            self.message_bus.subscribe(agent_name, callback)
        
        # Log what we learned about this agent
        logger.info(f" Capabilities:")
        logger.info(f"    • Preferred format: {caps.preferred_format}")
        logger.info(f"    • Accepts formats: {caps.acceptable_formats}")
        logger.info(f"    • Max input size: {caps.max_input_size} bytes")
        logger.info(f"    • Max context: {caps.max_context_tokens} tokens")
        logger.info(f" Registered successfully")
        
    def send(
        self,
        from_agent: str,
        to_agent: str,
        data: Any,
        field_name: str = "data",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        SMART SEND: Handles ALL communication nuances automatically.
        
        WORKFLOW:
        1. Check target's format preference (from signature)
        2. Check if transformation needed → invoke internal Transformer
        3. Check if chunking needed → invoke internal Chunker
        4. Check if target's context full → invoke internal Compressor for target
        5. Deliver message
        6. Update context tracking
        
        ALL AUTOMATIC - agents don't need to worry about ANY of this!
        
        Args:
            from_agent: Sender agent name
            to_agent: Receiver agent name
            data: Data to send
            field_name: Name of the field (for metadata)
            metadata: Optional metadata dict
            
        Returns:
            True if message delivered successfully
        """
        logger.info(f" [SMART AGENT SLACK] {from_agent} → {to_agent}")
        
        # Get target's capabilities (what they need)
        target_caps = self.agent_capabilities.get(to_agent)
        if not target_caps:
            logger.error(f" Unknown agent: {to_agent} (not registered)")
            logger.error(f"   Known agents: {list(self.agent_capabilities.keys())}")
            return False
            
        # STEP 1: Check if target's context is getting full
        target_context = self.context_budget_tracker.get(to_agent, 0)
        target_max = target_caps.max_context_tokens
        
        if target_caps.needs_compression(target_context):
            logger.info(f" {to_agent}'s context at {target_context}/{target_max} ({int(target_context/target_max*100)}%)")
            if self.compressor:
                logger.info(f" [AUTO] Invoking internal Compressor for {to_agent}")
                self._compress_agent_context(to_agent)
            else:
                logger.warning(f" [AUTO] Compressor not available (skipping)")
            
        # STEP 2: Detect current data format
        current_format = self._detect_format(data)
        current_size = self._estimate_size(data)
        logger.info(f" Data: format={current_format}, size={current_size} bytes")
        
        # STEP 3: Check if format transformation needed
        if not target_caps.can_accept_format(current_format):
            target_format = target_caps.preferred_format
            logger.info(f" [AUTO] Format mismatch: '{current_format}' not in {target_caps.acceptable_formats}")
            logger.info(f" [AUTO] Invoking internal Transformer: {current_format} → {target_format}")
            
            try:
                transformed_data = self.transformer.transform(
                    data=data,
                    target_format=target_format,
                    source_agent=from_agent,
                    target_agent=to_agent
                )
                
                data = transformed_data
                current_format = target_format
                current_size = self._estimate_size(data)
                logger.info(f" Transformed: new size={current_size} bytes, format={current_format}")
            except Exception as e:
                logger.error(f" Transformation failed: {e}")
                logger.info(f" ℹ Proceeding with original format (may cause issues)")
            
        # STEP 4: Check if chunking needed (data too large for target)
        if current_size > target_caps.max_input_size:
            logger.info(f" [AUTO] Data too large: {current_size} > {target_caps.max_input_size}")
            if self.chunker:
                logger.info(f" [AUTO] Invoking internal Chunker")
                
                try:
                    chunked_file = self.chunker.chunk_and_consolidate(
                        data=data,
                        goal=f"Prepare data for {to_agent}",
                        target_agent_context=target_caps,
                        max_chunk_size=target_caps.max_input_size
                    )
                    
                    data = chunked_file
                    current_size = self._estimate_size(data)
                    logger.info(f" Chunked: new size={current_size} bytes")
                except Exception as e:
                    logger.error(f" Chunking failed: {e}")
                    logger.info(f" ℹ Proceeding with original data (may exceed limits)")
            else:
                logger.warning(f" [AUTO] Chunker not available (data may exceed limits)")
            
        # STEP 5: Deliver message
        message = Message(
            from_agent=from_agent,
            to_agent=to_agent,
            data=data,
            format=current_format,
            size_bytes=current_size,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        success = self.message_bus.publish(message)
        
        if success:
            # STEP 6: Update context tracking
            data_tokens = self._estimate_tokens(data)
            self.context_budget_tracker[to_agent] = self.context_budget_tracker.get(to_agent, 0) + data_tokens
            self.message_history.append(message)
            
            # 🆕 COOPERATION TRACKING
            if self.enable_cooperation:
                self._record_cooperation_event(
                    event_type='share',
                    from_agent=from_agent,
                    to_agent=to_agent,
                    description=f"Shared {field_name} ({current_size} bytes)",
                    impact=1.0  # Positive impact (helping)
                )
            
            logger.info(f" Message delivered successfully")
            logger.info(f" {to_agent} context now: {self.context_budget_tracker[to_agent]}/{target_max} tokens")
        else:
            logger.error(f" Message delivery failed")
        
        return success
        
    def receive(self, agent_name: str) -> List[Message]:
        """
        Get all messages for an agent.
        
        Args:
            agent_name: Agent requesting messages
            
        Returns:
            List of messages for this agent
        """
        messages = [m for m in self.message_history if m.to_agent == agent_name]
        logger.debug(f" [AGENT SLACK] {agent_name} received {len(messages)} messages")
        return messages

    # =========================================================================
    # MsgHub-inspired: Broadcast to all registered agents
    # =========================================================================

    def broadcast(
        self,
        from_agent: str,
        data: Any,
        field_name: str = "broadcast",
        metadata: Optional[Dict] = None,
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """
        Broadcast data to ALL registered agents (AgentScope MsgHub pattern).

        When an agent produces output that all other agents should see
        (e.g. swarm-wide announcements, shared context updates),
        broadcast instead of N individual send() calls.

        DRY: Reuses existing send() for each target with all its
        smart format detection, transformation, chunking, and compression.

        Args:
            from_agent: Agent sending the broadcast
            data: Data to broadcast
            field_name: Name of the field (for metadata)
            metadata: Optional metadata dict
            exclude: Agent names to exclude from broadcast

        Returns:
            Dict of agent_name -> delivery success
        """
        exclude_set = set(exclude or [])
        exclude_set.add(from_agent)  # Never broadcast to self

        targets = [
            name for name in self.agent_capabilities
            if name not in exclude_set
        ]

        results = {}
        for target in targets:
            results[target] = self.send(
                from_agent=from_agent,
                to_agent=target,
                data=data,
                field_name=field_name,
                metadata=metadata,
            )

        delivered = sum(1 for v in results.values() if v)
        logger.info(
            f" [AGENT SLACK] {from_agent} broadcast to "
            f"{delivered}/{len(targets)} agents"
        )
        return results
        
    def _compress_agent_context(self, agent_name: str):
        """
        Compress an agent's accumulated context using internal Compressor.
        
        This is called automatically when agent hits 70% context budget.
        """
        # Get agent's message history
        agent_messages = [m for m in self.message_history if m.to_agent == agent_name]
        
        if not agent_messages:
            logger.debug(f" ℹ No messages to compress for {agent_name}")
            return
            
        # Compress using internal Compressor
        original_tokens = self.context_budget_tracker[agent_name]
        
        try:
            compressed_context = self.compressor.compress(
                messages=agent_messages,
                agent_name=agent_name,
                target_ratio=0.5  # Compress to 50%
            )
            
            # Update tracking
            new_tokens = int(original_tokens * 0.5)
            self.context_budget_tracker[agent_name] = new_tokens
            
            logger.info(f" Compressed {agent_name} context: {original_tokens} → {new_tokens} tokens")
        except Exception as e:
            logger.error(f" Compression failed: {e}")
        
    def _extract_capabilities_from_signature(
        self,
        agent_name: str,
        signature: dspy.Signature,
        max_context: int
    ) -> AgentCapabilities:
        """
        AUTOMATICALLY extract agent's capabilities from its DSPy signature.
        
        This is the MAGIC: We READ the signature to understand everything
        the agent needs, so Agent Slack can handle it automatically!
        
        Args:
            agent_name: Name of the agent
            signature: DSPy signature to analyze
            max_context: Max context window in tokens
            
        Returns:
            AgentCapabilities extracted from signature
        """
        # Default values
        preferred_format = "dict"
        acceptable_formats = ["dict", "str", "list"]
        max_input_size = max_context * 4  # Rough estimate: 1 token ≈ 4 bytes
        
        # Try to extract format preferences from signature
        # NOTE: This is a best-effort extraction. DSPy signatures may not have explicit format hints.
        # We look for common patterns in field descriptions or annotations.
        
        try:
            # Check if signature has input_fields
            if hasattr(signature, 'input_fields'):
                for field_name, field in signature.input_fields.items():
                    # Check for format hints in description
                    if hasattr(field, 'desc') and field.desc:
                        desc_lower = field.desc.lower()
                        if 'json' in desc_lower:
                            preferred_format = 'json'
                            acceptable_formats = ['json', 'dict']
                        elif 'csv' in desc_lower:
                            preferred_format = 'csv'
                            acceptable_formats = ['csv', 'str']
                        elif 'text' in desc_lower or 'string' in desc_lower:
                            preferred_format = 'str'
                            acceptable_formats = ['str', 'text']
                    
                    # Check for type annotations
                    if hasattr(field, 'annotation'):
                        annotation = field.annotation
                        if annotation == dict or str(annotation) == 'Dict':
                            preferred_format = 'dict'
                            acceptable_formats = ['dict', 'json']
                        elif annotation == str:
                            preferred_format = 'str'
                            acceptable_formats = ['str', 'text']
                        elif annotation == list or str(annotation) == 'List':
                            preferred_format = 'list'
                            acceptable_formats = ['list', 'json']
        except Exception as e:
            logger.warning(f" Could not extract format from signature: {e}")
            logger.warning(f" ℹ Using default capabilities")
        
        return AgentCapabilities(
            agent_name=agent_name,
            preferred_format=preferred_format,
            acceptable_formats=acceptable_formats,
            max_input_size=max_input_size,
            max_context_tokens=max_context
        )
    
    def _detect_format(self, data: Any) -> str:
        """Detect format of data."""
        if isinstance(data, dict):
            return 'dict'
        elif isinstance(data, list):
            return 'list'
        elif isinstance(data, str):
            # STRICT POLICY: avoid content-based heuristics (no regex/keyword matching).
            # Treat all strings as 'str' unless the sender explicitly provides format metadata.
            return 'str'
        elif isinstance(data, bytes):
            return 'bytes'
        else:
            return 'unknown'
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        try:
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data))
            else:
                return sys.getsizeof(data)
        except (TypeError, ValueError) as e:
            logger.debug(f"Size estimation failed: {e}")
            return 1000  # Default estimate
    
    def _estimate_tokens(self, data: Any) -> int:
        """Estimate token count using SmartTokenizer."""
        if isinstance(data, str):
            return SmartTokenizer.get_instance().count_tokens(data)
        # For non-string data, convert to string first
        size_bytes = self._estimate_size(data)
        # Use size-based estimation for non-string types
        return size_bytes // 4
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about Agent Slack activity."""
        return {
            'total_messages': len(self.message_history),
            'registered_agents': len(self.agent_capabilities),
            'context_budgets': dict(self.context_budget_tracker),
            'agent_capabilities': {
                name: {
                    'preferred_format': caps.preferred_format,
                    'max_context': caps.max_context_tokens
                }
                for name, caps in self.agent_capabilities.items()
            }
        }
    
    def reset_context(self, agent_name: Optional[str] = None) -> None:
        """Reset context budget for agent(s)."""
        if agent_name:
            self.context_budget_tracker[agent_name] = 0
            logger.info(f" [AGENT SLACK] Reset context for {agent_name}")
        else:
            self.context_budget_tracker = {name: 0 for name in self.agent_capabilities.keys()}
            logger.info(f" [AGENT SLACK] Reset context for all agents")
    
    # =========================================================================
    # 🆕 COOPERATION METHODS
    # =========================================================================
    
    def _record_cooperation_event(
        self,
        event_type: str,
        from_agent: str,
        to_agent: str,
        description: str,
        impact: float
    ):
        """
        Record a cooperation event.
        
        Args:
            event_type: 'share', 'help', 'unblock', 'request'
            from_agent: Agent providing help
            to_agent: Agent receiving help
            description: What happened
            impact: Impact score (0.0-1.0)
        """
        event = {
            'type': event_type,
            'from': from_agent,
            'to': to_agent,
            'description': description,
            'impact': impact,
            'timestamp': time.time()
        }
        
        self.cooperation_events.append(event)
        
        # Update help matrix
        key = (from_agent, to_agent)
        self.help_matrix[key] = self.help_matrix.get(key, 0) + 1
        
        logger.debug(f" [COOPERATION] {from_agent} → {to_agent}: {event_type}")
    
    def get_cooperation_stats(self) -> Dict[str, Any]:
        """
        Get cooperation statistics.
        
        Returns:
            {
                'total_events': int,
                'help_matrix': {(from, to): count},
                'most_helpful': agent_name,
                'most_helped': agent_name,
                'recent_events': [events]
            }
        """
        if not self.enable_cooperation:
            return {'enabled': False}
        
        # Find most helpful agent (gives most help)
        help_given = {}
        for (from_agent, to_agent), count in self.help_matrix.items():
            help_given[from_agent] = help_given.get(from_agent, 0) + count
        
        most_helpful = max(help_given.items(), key=lambda x: x[1])[0] if help_given else None
        
        # Find most helped agent (receives most help)
        help_received = {}
        for (from_agent, to_agent), count in self.help_matrix.items():
            help_received[to_agent] = help_received.get(to_agent, 0) + count
        
        most_helped = max(help_received.items(), key=lambda x: x[1])[0] if help_received else None
        
        return {
            'enabled': True,
            'total_events': len(self.cooperation_events),
            'help_matrix': dict(self.help_matrix),
            'help_given': help_given,
            'help_received': help_received,
            'most_helpful': most_helpful,
            'most_helped': most_helped,
            'recent_events': self.cooperation_events[-10:]  # Last 10
        }
    
    def should_communicate(
        self,
        from_agent: str,
        to_agent: str,
        information_value: float,
        cost_estimate: float = 0.1,
        receiver_confidence_before: float = 0.5,
        context_budget_remaining: float = 1.0
    ) -> bool:
        """
        Nash Equilibrium: Should I communicate this information?
        
         A-TEAM ENHANCEMENTS (per GRF MARL paper):
        - Learn value from (receiver confidence lift, predicted reward lift)
        - Learn cost from context budget + latency history
        - Log decisions for learning
        
        Communicate if: Value_of_information > Cost_of_communication
        
        Args:
            from_agent: Sender
            to_agent: Receiver
            information_value: Estimated value to receiver (0.0-1.0)
            cost_estimate: Cost of communication (default: 0.1)
            receiver_confidence_before: Receiver's confidence before receiving info
            context_budget_remaining: How much context budget receiver has (0.0-1.0)
        
        Returns:
            True if should communicate
        """
        # LEARNED VALUE: Incorporate historical cooperation success
        pair_key = (from_agent, to_agent)
        historical_success = self._get_cooperation_success_rate(from_agent, to_agent)
        
        # Value = base_value * historical_success * (1 - receiver_confidence)
        # Higher value if receiver needs it (low confidence) and history shows benefit
        confidence_gap = 1.0 - receiver_confidence_before
        learned_value = information_value * (0.5 + 0.5 * historical_success) * (0.5 + 0.5 * confidence_gap)
        
        # LEARNED COST: Incorporate context budget and latency
        # Higher cost if receiver has low context budget (might get compressed/lost)
        budget_penalty = 0.2 * (1.0 - context_budget_remaining)  # 0-0.2 extra cost
        latency_history = self._get_average_latency(from_agent, to_agent)
        latency_penalty = min(0.1, latency_history / 10.0)  # 0-0.1 extra cost
        
        learned_cost = cost_estimate + budget_penalty + latency_penalty
        
        # Nash equilibrium: communicate if net positive
        net_value = learned_value - learned_cost
        
        should_send = net_value > 0
        
        # LOG FOR LEARNING
        self._log_communication_decision(
            from_agent=from_agent,
            to_agent=to_agent,
            raw_value=information_value,
            learned_value=learned_value,
            raw_cost=cost_estimate,
            learned_cost=learned_cost,
            decision=should_send,
            historical_success=historical_success
        )
        
        if should_send:
            logger.debug(f" [NASH] {from_agent} → {to_agent}: SEND (learned_value={learned_value:.2f}, learned_cost={learned_cost:.2f}, history_success={historical_success:.2f})")
        else:
            logger.debug(f" [NASH] {from_agent} → {to_agent}: SKIP (learned_value={learned_value:.2f}, learned_cost={learned_cost:.2f}, history_success={historical_success:.2f})")
        
        return should_send
    
    def _get_cooperation_success_rate(self, from_agent: str, to_agent: str) -> float:
        """Get historical success rate for this agent pair."""
        pair_key = (from_agent, to_agent)
        if pair_key in self.help_matrix:
            help_count = self.help_matrix[pair_key]
            # Simple heuristic: more help = more success (cap at 1.0)
            return min(1.0, help_count / 5.0)
        return 0.5  # Default: neutral
    
    def _get_average_latency(self, from_agent: str, to_agent: str) -> float:
        """Get average communication latency for this pair (in seconds)."""
        # Check recent comms for this pair
        pair_comms = [
            c for c in self.communication_log[-20:]
            if c.get('from') == from_agent and c.get('to') == to_agent
        ]
        if pair_comms:
            latencies = [c.get('latency', 0.1) for c in pair_comms]
            return sum(latencies) / len(latencies)
        return 0.1  # Default
    
    def _log_communication_decision(self, **kwargs):
        """Log communication decision for learning."""
        self.communication_log.append({
            'timestamp': time.time(),
            'type': 'nash_decision',
            **kwargs
        })

