"""
Scratchpad Persistence - Save/Load SharedScratchpad to Disk

Enables:
- Saving scratchpad state between sessions
- Resuming multi-agent workflows
- Audit trail of agent communication
- Time-travel debugging (replay messages)

Storage Format: JSON Lines (.jsonl)
- One message per line
- Easy to append
- Easy to stream/replay
- Human-readable
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from core.foundation.types.agent_types import SharedScratchpad, AgentMessage, CommunicationType

logger = logging.getLogger(__name__)


class ScratchpadPersistence:
    """
    Persist SharedScratchpad to disk.

    Features:
    - Auto-save on every message (append-only)
    - Load from disk to resume
    - Export full history as JSON
    - Replay messages chronologically

    File Format:
        scratchpad_YYYYMMDD_HHMMSS.jsonl

    Each line:
        {"timestamp": "...", "sender": "...", "receiver": "...", "type": "...", "content": {...}}
    """

    def __init__(self, workspace_dir: str = 'workspace/scratchpads') -> None:
        """
        Initialize scratchpad persistence.

        Args:
            workspace_dir: Directory to store scratchpad files
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f" ScratchpadPersistence initialized: {self.workspace_dir}")

    def create_session(self, session_name: Optional[str] = None) -> Path:
        """
        Create new scratchpad session file.

        Args:
            session_name: Optional session name (defaults to timestamp)

        Returns:
            Path to session file
        """
        if session_name is None:
            session_name = datetime.now().strftime("scratchpad_%Y%m%d_%H%M%S")

        session_file = self.workspace_dir / f"{session_name}.jsonl"

        # Create empty file
        session_file.touch()

        logger.info(f" Created scratchpad session: {session_file}")

        return session_file

    def save_message(self, session_file: Path, message: AgentMessage) -> None:
        """
        Append message to scratchpad file.

        Args:
            session_file: Path to session file
            message: AgentMessage to save
        """
        # Convert message to dict
        message_dict = {
            'timestamp': message.timestamp.isoformat(),
            'sender': message.sender,
            'receiver': message.receiver,
            'message_type': message.message_type.value,
            'content': message.content,
            'tool_name': message.tool_name,
            'tool_args': message.tool_args,
            'tool_result': str(message.tool_result)[:500] if message.tool_result else None,  # Truncate large results
            'insight': message.insight,
            'confidence': message.confidence
        }

        # Append as JSON line
        with session_file.open('a') as f:
            f.write(json.dumps(message_dict) + '\n')

    def save_scratchpad(self, session_file: Path, scratchpad: SharedScratchpad) -> None:
        """
        Save entire scratchpad state to file.

        Args:
            session_file: Path to session file
            scratchpad: SharedScratchpad to save
        """
        # Save all messages
        for message in scratchpad.messages:
            self.save_message(session_file, message)

        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'type': 'metadata',
            'total_messages': len(scratchpad.messages),
            'total_insights': len(scratchpad.shared_insights),
            'tool_cache_size': len(scratchpad.tool_cache),
            'shared_insights': scratchpad.shared_insights
        }

        with session_file.open('a') as f:
            f.write(json.dumps(metadata) + '\n')

        logger.info(f" Saved scratchpad: {len(scratchpad.messages)} messages to {session_file}")

    def load_scratchpad(self, session_file: Path) -> SharedScratchpad:
        """
        Load scratchpad from file.

        Args:
            session_file: Path to session file

        Returns:
            Loaded SharedScratchpad
        """
        scratchpad = SharedScratchpad()

        if not session_file.exists():
            logger.warning(f" Scratchpad file not found: {session_file}")
            return scratchpad

        # Read all lines
        with session_file.open('r') as f:
            for line in f:
                data = json.loads(line)

                # Skip metadata lines
                if data.get('type') == 'metadata':
                    if 'shared_insights' in data:
                        scratchpad.shared_insights = data['shared_insights']
                    continue

                # Reconstruct AgentMessage
                message = AgentMessage(
                    sender=data['sender'],
                    receiver=data['receiver'],
                    message_type=CommunicationType(data['message_type']),
                    content=data['content'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    tool_name=data.get('tool_name'),
                    tool_args=data.get('tool_args'),
                    tool_result=data.get('tool_result'),
                    insight=data.get('insight'),
                    confidence=data.get('confidence')
                )

                scratchpad.add_message(message)

        logger.info(f" Loaded scratchpad: {len(scratchpad.messages)} messages from {session_file}")

        return scratchpad

    def list_sessions(self) -> List[Path]:
        """
        List all scratchpad session files.

        Returns:
            List of session file paths
        """
        return sorted(self.workspace_dir.glob("*.jsonl"), reverse=True)

    def export_session(self, session_file: Path, output_format: str = 'json') -> str:
        """
        Export session to readable format.

        Args:
            session_file: Path to session file
            output_format: 'json' or 'markdown'

        Returns:
            Formatted output string
        """
        scratchpad = self.load_scratchpad(session_file)

        if output_format == 'json':
            return json.dumps({
                'messages': [
                    {
                        'timestamp': msg.timestamp.isoformat(),
                        'sender': msg.sender,
                        'receiver': msg.receiver,
                        'type': msg.message_type.value,
                        'content': msg.content,
                        'insight': msg.insight
                    }
                    for msg in scratchpad.messages
                ],
                'shared_insights': scratchpad.shared_insights,
                'total_messages': len(scratchpad.messages)
            }, indent=2)

        elif output_format == 'markdown':
            lines = [f"# Scratchpad Session: {session_file.stem}\n"]
            lines.append(f"**Total Messages**: {len(scratchpad.messages)}\n")
            lines.append(f"**Shared Insights**: {len(scratchpad.shared_insights)}\n")
            lines.append("\n---\n\n## Messages\n")

            for i, msg in enumerate(scratchpad.messages, 1):
                lines.append(f"### Message {i}\n")
                lines.append(f"**From**: {msg.sender} → **To**: {msg.receiver}\n")
                lines.append(f"**Type**: {msg.message_type.value}\n")
                lines.append(f"**Time**: {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                if msg.insight:
                    lines.append(f"**Insight**: {msg.insight}\n")
                lines.append(f"**Content**:\n```json\n{json.dumps(msg.content, indent=2)}\n```\n\n")

            lines.append("## Shared Insights\n")
            for insight in scratchpad.shared_insights:
                lines.append(f"- {insight}\n")

            return ''.join(lines)

        else:
            raise ValueError(f"Unknown format: {output_format}")

    def replay_session(self, session_file: Path) -> List[AgentMessage]:
        """
        Replay messages chronologically.

        Args:
            session_file: Path to session file

        Returns:
            List of messages in chronological order
        """
        scratchpad = self.load_scratchpad(session_file)

        # Already sorted by timestamp (append-only)
        return scratchpad.messages

    def get_messages_by_agent(self, session_file: Path, agent_name: str) -> List[AgentMessage]:
        """
        Get all messages from specific agent.

        Args:
            session_file: Path to session file
            agent_name: Agent name to filter by

        Returns:
            List of messages from agent
        """
        scratchpad = self.load_scratchpad(session_file)

        return [msg for msg in scratchpad.messages if msg.sender == agent_name]

    def get_conversation(self, session_file: Path, agent_a: str, agent_b: str) -> List[AgentMessage]:
        """
        Get conversation between two agents.

        Args:
            session_file: Path to session file
            agent_a: First agent name
            agent_b: Second agent name

        Returns:
            List of messages between agents
        """
        scratchpad = self.load_scratchpad(session_file)

        return [
            msg for msg in scratchpad.messages
            if (msg.sender == agent_a and msg.receiver in [agent_b, "*"]) or
               (msg.sender == agent_b and msg.receiver in [agent_a, "*"])
        ]

    def __repr__(self) -> str:
        """String representation."""
        return f"ScratchpadPersistence(workspace={self.workspace_dir})"
