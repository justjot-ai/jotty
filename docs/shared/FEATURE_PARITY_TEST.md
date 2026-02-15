# Feature Parity Testing Guide

**Purpose:** Verify that the new shared components implementation maintains 100% feature parity with the old custom implementation.

**Date:** February 15, 2026
**Status:** Testing Required

---

## Quick Test Commands

### Test TUI with Shared Components
```bash
# NEW implementation (shared components)
python -m apps.cli.app_migrated

# OLD implementation (custom rendering)
python -m apps.cli.app
```

### Test Telegram with Shared Components
```bash
# NEW implementation requires updating bot.py first
# See "Telegram Migration" section below
```

---

## Feature Parity Checklist

### ✅ Core Features (Must Work Identically)

#### 1. Message Display
- [ ] **User messages** display with correct styling (blue border in terminal, proper format in Telegram)
- [ ] **Assistant messages** display with markdown rendering
- [ ] **System messages** display with yellow/info styling
- [ ] **Timestamps** shown for all messages
- [ ] **Role labels** (User/Assistant/System) visible

#### 2. Markdown Rendering
- [ ] **Headers** (H1-H6) render correctly
- [ ] **Bold** text (`**text**`)
- [ ] **Italic** text (`*text*`)
- [ ] **Code inline** (`` `code` ``)
- [ ] **Code blocks** with syntax highlighting (```python, ```javascript, etc.)
- [ ] **Lists** (ordered and unordered)
- [ ] **Tables** render correctly
- [ ] **Links** are clickable/visible
- [ ] **Blockquotes** display with proper indentation

#### 3. Streaming
- [ ] **Text streaming** works (incremental display)
- [ ] **Chunks** append to message without flicker
- [ ] **Streaming indicator** shows while active
- [ ] **Final message** displays when stream completes

#### 4. Event Types (24 Total)

**Lifecycle Events:**
- [ ] START event displays correctly
- [ ] COMPLETE event shows completion status
- [ ] ERROR event displays error with traceback

**Processing Events:**
- [ ] THINKING event shows "Thinking..." indicator
- [ ] PLANNING event shows "Planning..." indicator

**Skill Events:**
- [ ] SKILL_START shows skill name and icon
- [ ] SKILL_PROGRESS shows progress bar/percentage
- [ ] SKILL_COMPLETE shows completion status

**Output Events:**
- [ ] STREAM event updates message incrementally
- [ ] DELTA event appends chunks correctly

**Agent Events:**
- [ ] AGENT_START shows agent name and icon
- [ ] AGENT_COMPLETE shows completion

**Memory Events:**
- [ ] MEMORY_RECALL shows retrieved memories
- [ ] MEMORY_STORE confirms storage

**Validation & Learning Events:**
- [ ] VALIDATION_START shows validation beginning
- [ ] VALIDATION_COMPLETE shows validation results
- [ ] LEARNING_UPDATE shows learning progress

**Voice Events:**
- [ ] VOICE_STT_START shows transcription starting
- [ ] VOICE_STT_COMPLETE shows transcribed text
- [ ] VOICE_TTS_START shows synthesis starting
- [ ] VOICE_TTS_CHUNK plays audio chunk
- [ ] VOICE_TTS_COMPLETE confirms synthesis done

**Swarm Events:**
- [ ] SWARM_AGENT_START shows swarm agent starting
- [ ] SWARM_AGENT_COMPLETE shows swarm agent completion
- [ ] SWARM_COORDINATION shows coordination status

#### 5. State Management (13 States)
- [ ] IDLE state (initial state)
- [ ] THINKING state (with indicator)
- [ ] PLANNING state (with indicator)
- [ ] EXECUTING_SKILL state (shows skill name)
- [ ] EXECUTING_AGENT state (shows agent name)
- [ ] COORDINATING_SWARM state (shows swarm status)
- [ ] STREAMING state (shows streaming indicator)
- [ ] TRANSCRIBING state (voice input)
- [ ] SYNTHESIZING state (voice output)
- [ ] WAITING_INPUT state (waiting for user)
- [ ] VALIDATING state (validation in progress)
- [ ] LEARNING state (learning update)
- [ ] ERROR state (error display)

#### 6. CLI Commands (36 Total)

**Basic Commands:**
- [ ] `/help` - Show help
- [ ] `/clear` - Clear screen
- [ ] `/quit` or `/exit` - Exit CLI
- [ ] `/history` - Show command history

**Agent Commands:**
- [ ] `/agent <name>` - Run specific agent
- [ ] `/agent list` - List available agents
- [ ] `/agent status` - Show agent status

**Skill Commands:**
- [ ] `/skill <name>` - Execute skill
- [ ] `/skill list` - List all skills
- [ ] `/skill search <query>` - Search skills
- [ ] `/skill info <name>` - Skill details

**Swarm Commands:**
- [ ] `/swarm <agents>` - Run swarm coordination
- [ ] `/swarm status` - Show swarm status
- [ ] `/swarm list` - List available swarms

**Workflow Commands:**
- [ ] `/workflow <name>` - Execute workflow
- [ ] `/workflow list` - List workflows
- [ ] `/workflow create` - Create new workflow

**Memory Commands:**
- [ ] `/memory` - Show memory status
- [ ] `/memory search <query>` - Search memories
- [ ] `/memory clear` - Clear memory
- [ ] `/memory export` - Export memories

**Learning Commands:**
- [ ] `/learning status` - Show learning stats
- [ ] `/learning rewards` - Show rewards
- [ ] `/learning policy` - Show policy

**Session Commands:**
- [ ] `/session save` - Save session
- [ ] `/session load` - Load session
- [ ] `/session export` - Export session
- [ ] `/session list` - List sessions

**Model Commands:**
- [ ] `/model` - Show current model
- [ ] `/model switch <name>` - Switch model
- [ ] `/model list` - List available models

**Voice Commands:**
- [ ] `/voice` - Toggle voice mode
- [ ] `/tts <text>` - Text-to-speech
- [ ] `/stt` - Speech-to-text

**Debug Commands:**
- [ ] `/debug` - Toggle debug mode
- [ ] `/status` - Show system status
- [ ] `/tokens` - Show token usage
- [ ] `/cost` - Show cost breakdown

**Config Commands:**
- [ ] `/config` - Show configuration
- [ ] `/config set <key> <value>` - Set config value

#### 7. Input Handling
- [ ] **Text input** works with prompt
- [ ] **Multiline input** (Shift+Enter or special mode)
- [ ] **Voice input** (if microphone available)
- [ ] **Autocomplete** (for slash commands)
- [ ] **History navigation** (Up/Down arrows)
- [ ] **Ctrl+C** interrupt handling
- [ ] **Ctrl+D** exit handling
- [ ] **Vi mode** keybindings (if enabled)
- [ ] **Emacs mode** keybindings (if enabled)

#### 8. Progress Indicators
- [ ] **Spinner** shows during processing
- [ ] **Progress bar** shows for long operations
- [ ] **Percentage** displays for measurable progress
- [ ] **Step indicators** (e.g., "Step 2/5")

#### 9. Error Handling
- [ ] **Error messages** display clearly
- [ ] **Traceback** shown (if debug mode)
- [ ] **Recoverable errors** allow continuation
- [ ] **Non-recoverable errors** exit gracefully
- [ ] **Network errors** display helpful message

#### 10. Special Features

**Terminal (TUI):**
- [ ] **Rich panels** with borders
- [ ] **Syntax highlighting** (40+ languages)
- [ ] **Tables** render with proper alignment
- [ ] **Live updates** (for progress)
- [ ] **Color themes** work correctly
- [ ] **No-color mode** works (--no-color flag)

**Telegram:**
- [ ] **MarkdownV2 formatting** renders correctly
- [ ] **Special characters** escaped properly
- [ ] **Message splitting** (4096 char limit)
- [ ] **Buttons** work (if used)
- [ ] **Inline queries** work (if used)
- [ ] **File attachments** display correctly
- [ ] **Images** render inline
- [ ] **Voice messages** work

**WhatsApp:**
- [ ] **Markdown formatting** renders correctly
- [ ] **Message splitting** (65536 char limit)
- [ ] **Media attachments** work
- [ ] **Voice messages** work

---

## Testing Procedure

### Phase 1: Basic Functionality (15 minutes)

1. **Start TUI (New Implementation)**
   ```bash
   python -m apps.cli.app_migrated
   ```

2. **Test Basic Chat**
   - Input: "Hello, how are you?"
   - Expected: Assistant responds with greeting
   - Verify: Message displays with correct formatting

3. **Test Markdown**
   - Input: "Show me a code example in Python"
   - Expected: Code block with syntax highlighting
   - Verify: Colors and line numbers visible

4. **Test Streaming**
   - Input: "Write a long story"
   - Expected: Text appears incrementally
   - Verify: No flicker, smooth display

5. **Test Commands**
   - Input: `/help`
   - Expected: Help text displays
   - Input: `/status`
   - Expected: System status shows
   - Input: `/clear`
   - Expected: Screen clears

### Phase 2: Event Handling (20 minutes)

6. **Test Skill Execution**
   - Input: "/skill web-search"
   - Expected: SKILL_START → SKILL_PROGRESS → SKILL_COMPLETE events
   - Verify: Progress bar displays, completion message shows

7. **Test Agent Execution**
   - Input: "/agent researcher"
   - Expected: AGENT_START → THINKING → PLANNING → AGENT_COMPLETE
   - Verify: State transitions visible, agent name displayed

8. **Test Swarm Coordination**
   - Input: "/swarm researcher,coder,tester"
   - Expected: SWARM_AGENT_START for each agent, SWARM_COORDINATION
   - Verify: Multiple agent statuses displayed

9. **Test Memory Operations**
   - Input: "/memory search AI"
   - Expected: MEMORY_RECALL event, results displayed
   - Verify: Memories retrieved and formatted correctly

10. **Test Learning Updates**
    - Input: "Complete a task successfully"
    - Expected: LEARNING_UPDATE event
    - Verify: Learning message displays

### Phase 3: Error Handling (10 minutes)

11. **Test Network Error**
    - Disable internet, send message
    - Expected: ERROR event with helpful message
    - Verify: Error panel displays, traceback visible (debug mode)

12. **Test Invalid Command**
    - Input: "/invalid-command"
    - Expected: Error message "Unknown command"
    - Verify: Error displays without crash

13. **Test Interrupt**
    - Send long task, press Ctrl+C
    - Expected: Graceful cancellation
    - Verify: Returns to prompt, no crash

### Phase 4: Advanced Features (15 minutes)

14. **Test Voice (if available)**
    - Input: "/voice"
    - Expected: Voice mode activates
    - Speak into microphone
    - Expected: VOICE_STT_START → VOICE_STT_COMPLETE → transcribed text
    - Verify: Transcription accurate

15. **Test Multiline Input**
    - Press Shift+Enter or enable multiline mode
    - Type multiple lines
    - Expected: All lines accepted
    - Verify: Formatting preserved

16. **Test History**
    - Press Up arrow
    - Expected: Previous command appears
    - Press Down arrow
    - Expected: Next command appears
    - Verify: History navigation works

17. **Test Autocomplete**
    - Type "/ski" and press Tab
    - Expected: Autocomplete to "/skill"
    - Verify: Suggestions appear

### Phase 5: Telegram Testing (30 minutes)

**Prerequisites:**
1. Set `TELEGRAM_TOKEN` in .env
2. Update `apps/telegram/bot.py` to use shared components (see migration section below)
3. Start bot: `python -m apps.telegram.bot`

18. **Test Basic Telegram Chat**
    - Send: "Hello"
    - Expected: Bot responds with greeting
    - Verify: MarkdownV2 formatting correct

19. **Test Telegram Commands**
    - Send: "/help"
    - Expected: Help message with commands
    - Send: "/status"
    - Expected: Bot status

20. **Test Long Messages**
    - Send: "Write a very long article about AI"
    - Expected: Multiple messages (auto-split at 4096 chars)
    - Verify: No truncation, proper formatting

21. **Test Special Characters**
    - Send: "Show me code with _, *, [], (), etc."
    - Expected: Characters properly escaped
    - Verify: No Telegram parse errors

---

## Telegram Migration Instructions

To test Telegram with shared components, update `apps/telegram/bot.py`:

```python
# Add at top of file
from apps.shared import ChatInterface
from apps.shared.renderers import TelegramMessageRenderer, TelegramStatusRenderer
from apps.shared.events import EventProcessor
from apps.shared.models import Message

# In TelegramBotHandler.__init__, add:
async def send_telegram_message(text: str):
    """Send message to Telegram."""
    await update.message.reply_text(text, parse_mode="MarkdownV2")

self.chat_interface = ChatInterface(
    message_renderer=TelegramMessageRenderer(send_telegram_message),
    status_renderer=TelegramStatusRenderer(send_telegram_message),
    input_handler=None,  # Not needed for bot
)
self.event_processor = EventProcessor(self.chat_interface)

# In message handler:
async for event in executor.chat_stream(text):
    await self.event_processor.process_event(event)
```

---

## Known Limitations (To Be Fixed)

1. **Voice Mode** - Voice input/output may need platform-specific adapters
2. **File Attachments** - Attachment rendering might differ across platforms
3. **Inline Buttons** - Telegram inline keyboards not yet implemented in shared renderer
4. **Live Updates** - Rich live displays may not work identically in all terminals

---

## Success Criteria

**Passing Grade:** 90% of features working identically (allowable differences for platform constraints)

**Feature Parity Met When:**
- ✅ All 24 event types display correctly
- ✅ All 13 states transition properly
- ✅ All 36 CLI commands work
- ✅ Markdown renders identically
- ✅ Streaming works smoothly
- ✅ Error handling is identical
- ✅ No crashes or missing features

**Acceptable Differences:**
- Terminal colors vs Telegram emoji (platform-specific)
- Progress bar width (depends on terminal width)
- Voice features (if hardware not available)

**Unacceptable Differences:**
- Missing commands
- Non-working event types
- Broken markdown rendering
- Crashes or errors
- Missing state transitions

---

## Reporting Issues

If you find features that don't work:

1. **Document the issue:**
   - Feature name
   - Expected behavior (from old implementation)
   - Actual behavior (from new implementation)
   - Steps to reproduce

2. **Create a test case:**
   ```python
   def test_missing_feature():
       """Test that X feature works correctly."""
       # Setup
       chat = ChatInterface(...)
       # Execute
       result = chat.method(...)
       # Verify
       assert result == expected
   ```

3. **Priority:**
   - P0 (Critical): Core chat functionality broken
   - P1 (High): Commands don't work, crashes
   - P2 (Medium): Visual differences, minor bugs
   - P3 (Low): Nice-to-have features, polish

---

## Next Steps After Testing

1. **If all tests pass:**
   - Migrate production `apps/cli/app.py` to use shared components
   - Migrate production `apps/telegram/bot.py` to use shared components
   - Migrate `apps/web` to use `web.tsx` renderer
   - Migrate WhatsApp, Slack, Discord

2. **If issues found:**
   - Fix shared components to match old behavior
   - Update renderers to support missing features
   - Add tests to prevent regressions
   - Re-test until 90%+ parity achieved

3. **Documentation:**
   - Update `MIGRATION_GUIDE.md` with lessons learned
   - Document any breaking changes
   - Create upgrade guide for custom renderers

---

**Built with ❤️ for 100% feature parity across all platforms**
