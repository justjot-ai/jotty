# Workflow Examples

Complete examples demonstrating Jotty's three intent-based workflows with multi-format output and multi-channel delivery.

---

## 📚 Available Examples

### 1. **AutoWorkflow: AI Trading System** (`auto_workflow_trading_system.py`)

**Scenario:** Production-ready AI trading system for crypto

**Pipeline:** 9 stages (22% auto, 33% customized, 11% replaced, 33% custom)

**Demonstrates:**
- Auto-generate base pipeline (6 stages)
- Customize specific stages (code, docs, deployment)
- Replace stage with custom swarms (tests → ML testing)
- Add custom stages (backtesting, compliance, monitoring)

**Deliverables:**
- Requirements & architecture
- LSTM neural network code
- Backtesting framework
- ML test suite (overfitting, data leakage, drift detection)
- Regulatory compliance (SEC, FINRA)
- Monitoring & alerts (Prometheus, Grafana)
- Documentation & deployment configs

**Run:**
```bash
python examples/workflows/auto_workflow_trading_system.py
```

**Expected Output:**
- 9-stage production trading system
- Cost: ~$0.015
- Time: ~3 minutes

---

### 2. **ResearchWorkflow: AI Safety Paper** (`research_workflow_ai_safety.py`)

**Scenario:** Comprehensive AI safety research report (publication-ready)

**Pipeline:** 9 stages (44% auto, 22% customized, 11% replaced, 33% custom)

**Demonstrates:**
- Auto-generate academic research pipeline
- Customize analysis & documentation stages
- Replace literature review with specialized AI safety researchers
- Add custom stages (expert perspectives, technical deep dive, recommendations)

**Deliverables:**
- Literature review (30+ sources from Anthropic, OpenAI, DeepMind)
- Technical analysis (RLHF, Constitutional AI, interpretability)
- Expert perspectives synthesis
- Technical deep dive (algorithms, implementations)
- Synthesis & recommendations
- Visualization & bibliography
- Academic paper format (arXiv-ready)

**Run:**
```bash
python examples/workflows/research_workflow_ai_safety.py
```

**Expected Output:**
- Publication-ready AI safety paper
- Cost: ~$0.015
- Time: ~3 minutes

---

### 3. **LearningWorkflow: Olympiad Mathematics** (`learning_workflow_olympiad.py`)

**Scenario:** Olympiad-level number theory course for competition preparation

**Pipeline:** 15 stages (60% auto, 13% customized, 7% replaced, 20% custom)

**Demonstrates:**
- Auto-generate complete learning curriculum
- Customize patterns stage with olympiad techniques
- Replace problems with specialized olympiad problem crafters
- Add custom stages (competition strategies, mental math tricks, practice schedule)

**Deliverables:**
- Complete curriculum & learning plan
- Fundamental concepts with intuition
- Olympiad-specific patterns & techniques
- 25 competition-level problems (⭐ to ⭐⭐⭐⭐⭐)
- Multi-approach detailed solutions
- Common mistakes analysis
- Competition strategies & time management
- Mental math shortcuts
- 6-month preparation schedule
- Interactive HTML + Professional PDF
- Assessment (quizzes, tests, challenge problems)

**Run:**
```bash
python examples/workflows/learning_workflow_olympiad.py
```

**Expected Output:**
- Complete olympiad preparation course
- Cost: ~$0.020
- Time: ~4 minutes

---

### 4. **Channel Delivery Test** (`test_channel_delivery.py`)

**Scenario:** Test multi-channel delivery (Telegram, WhatsApp)

**Demonstrates:**
- Send text messages to Telegram
- Send files to Telegram
- Send to WhatsApp (if configured)
- Batch send to multiple channels

**Prerequisites:**
```bash
# Required for Telegram
export TELEGRAM_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Optional for WhatsApp
export WHATSAPP_TO="14155238886"  # With country code
```

**Run:**
```bash
python examples/workflows/test_channel_delivery.py
```

**Expected Output:**
- ✅ Telegram message sent
- ✅ Telegram file sent
- ✅ WhatsApp delivery (if configured)
- ✅ Batch send complete

---

### 5. **End-to-End Test** (`test_research_formats_channels.py`)

**Scenario:** Complete pipeline: Research → Formats → Channels

**Demonstrates:**
- Generate research content (ResearchWorkflow)
- Create multiple formats (PDF, EPUB, HTML)
- Send to multiple channels (Telegram, WhatsApp)

**Steps:**
1. Generate research on "Benefits of Daily Meditation"
2. Extract documentation stage
3. Generate PDF, EPUB, HTML formats
4. Send PDF to Telegram
5. Send PDF to WhatsApp (if configured)
6. Batch send to all channels

**Prerequisites:**
```bash
# Required
export ANTHROPIC_API_KEY="your_key"

# For format generation (optional)
brew install pandoc  # or apt/yum install pandoc

# For Telegram
export TELEGRAM_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# For WhatsApp (optional)
export WHATSAPP_TO="14155238886"
```

**Run:**
```bash
python examples/workflows/test_research_formats_channels.py
```

**Expected Output:**
- ✅ Research content generated (4 stages)
- ✅ Markdown saved
- ✅ PDF/EPUB/HTML generated (if Pandoc installed)
- ✅ Telegram delivery complete
- ✅ WhatsApp delivery (if configured)

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to Jotty
cd /var/www/sites/personal/stock_market/Jotty

# Set API key
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Optional: Telegram
export TELEGRAM_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Optional: WhatsApp
export WHATSAPP_TO="14155238886"

# Optional: Install Pandoc for format generation
# brew install pandoc  # macOS
# apt install pandoc   # Ubuntu
# yum install pandoc   # CentOS
```

### 2. Run Examples

```bash
# Start with simplest: Channel delivery test
python examples/workflows/test_channel_delivery.py

# Then try research workflow
python examples/workflows/research_workflow_ai_safety.py

# Try learning workflow
python examples/workflows/learning_workflow_olympiad.py

# Try software development
python examples/workflows/auto_workflow_trading_system.py

# Finally: End-to-end pipeline
python examples/workflows/test_research_formats_channels.py
```

---

## 📊 Comparison

| Example | Stages | Auto | Custom | Replaced | Added | Cost | Time |
|---------|--------|------|--------|----------|-------|------|------|
| Trading System | 9 | 22% | 33% | 11% | 33% | $0.015 | 3 min |
| AI Safety Paper | 9 | 44% | 22% | 11% | 33% | $0.015 | 3 min |
| Olympiad Course | 15 | 60% | 13% | 7% | 20% | $0.020 | 4 min |

---

## 🎯 What to Learn from Each Example

### Trading System
- **Learn:** Complex multi-stage customization
- **Pattern:** Start simple, add domain expertise (backtesting, compliance)
- **Use When:** Building production systems with specialized requirements

### AI Safety Paper
- **Learn:** Academic research workflow
- **Pattern:** Literature review → Analysis → Synthesis
- **Use When:** Creating publication-quality research papers

### Olympiad Course
- **Learn:** Educational content generation
- **Pattern:** Curriculum → Concepts → Practice → Assessment
- **Use When:** Creating comprehensive learning materials

### Channel Delivery
- **Learn:** Multi-channel communication
- **Pattern:** Generate once → Deliver everywhere
- **Use When:** Need to distribute content across multiple channels

### End-to-End
- **Learn:** Complete pipeline integration
- **Pattern:** Content → Formats → Channels
- **Use When:** Building production automation workflows

---

## 🔧 Customization Tips

### 1. Adjust Depth/Level

```python
# Research: Quick vs Comprehensive
research("topic", depth="quick")        # 5 sources, surface-level
research("topic", depth="comprehensive") # 25 sources, deep analysis

# Learning: Quick lesson vs Marathon course
learn("math", "topic", "student", depth="quick")    # 15-30 min
learn("math", "topic", "student", depth="marathon") # Full day
```

### 2. Modify Deliverables

```python
# Research: Choose specific deliverables
workflow = ResearchWorkflow.from_intent(
    topic="Your Topic",
    deliverables=["overview", "analysis", "recommendations"]  # Skip synthesis
)

# Learning: Focus on specific stages
workflow = LearningWorkflow.from_intent(
    topic="Your Topic",
    deliverables=["concepts", "problems", "solutions"]  # Skip assessment
)
```

### 3. Add Domain Expertise

```python
# Add custom swarms for domain-specific knowledge
workflow.add_custom_stage(
    "domain_expert_analysis",
    swarms=SwarmAdapter.quick_swarms([
        ("Domain Expert", "Analyze from domain perspective...")
    ])
)
```

---

## 📝 Output Locations

All examples save outputs to:
- **Markdown:** `~/jotty/outputs/*.md`
- **PDF:** `~/jotty/outputs/*.pdf`
- **EPUB:** `~/jotty/outputs/*.epub`
- **HTML:** `~/jotty/outputs/*.html`
- **Telegram:** Sent to configured chat
- **WhatsApp:** Sent to configured recipient

---

## 🐛 Troubleshooting

### "pandoc not found"
```bash
# Install Pandoc
brew install pandoc  # macOS
sudo apt install pandoc  # Ubuntu
sudo yum install pandoc  # CentOS
```

### "telegram-sender skill not available"
```bash
# Set Telegram credentials
export TELEGRAM_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### "WHATSAPP_TO parameter required"
```bash
# Set WhatsApp recipient
export WHATSAPP_TO="14155238886"  # With country code
```

### "Anthropic API error"
```bash
# Check API key
export ANTHROPIC_API_KEY="your_key"

# Verify it's loaded
echo $ANTHROPIC_API_KEY
```

---

## 📚 Further Reading

- **Complete Guide:** `docs/WORKFLOWS_COMPLETE_GUIDE.md`
- **Architecture:** `docs/JOTTY_ARCHITECTURE.md`
- **Pipeline Guide:** `docs/MULTI_STAGE_PIPELINE_GUIDE.md`

---

## 🎊 Summary

**5 Complete Examples:**
- ✅ AI Trading System (9 stages)
- ✅ AI Safety Paper (9 stages)
- ✅ Olympiad Course (15 stages)
- ✅ Channel Delivery (Telegram + WhatsApp)
- ✅ End-to-End Pipeline (Content → Formats → Channels)

**All Validated:**
- ✅ Content generation works
- ✅ Customization works
- ✅ Format generation works (with Pandoc)
- ✅ Channel delivery works (Telegram tested ✅)

**Ready to Use:**
- Copy examples as templates
- Modify for your use cases
- Build production workflows

Happy automating! 🚀
