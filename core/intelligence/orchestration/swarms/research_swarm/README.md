# Research Swarm

Deep research on stocks, companies, or general topics with professional PDF reports.

## 🎯 Purpose

Generates comprehensive research reports:
- Stock analysis with price targets and ratings
- Technical indicators and charts
- Peer comparison and sentiment analysis
- Web search with up-to-date information
- Professional PDF output

## 🚀 Quick Start

```python
from Jotty.core.swarms.research_swarm import ResearchSwarm

swarm = ResearchSwarm()

# Stock research
result = await swarm.execute(ticker="AAPL")

# Topic research
result = await swarm.research_topic("AI Trends 2026")
```

## 📋 Configuration

```python
from Jotty.core.swarms.research_swarm.types import ResearchConfig

config = ResearchConfig(
    send_telegram=True,
    include_charts=True,
    include_peers=True,
    include_sentiment=True,
    target_pages=12,
    max_web_results=25,
    exchange="NSE",  # or "NASDAQ", "NYSE", etc.
)

swarm = ResearchSwarm(config)
```

## 💼 Stock Analysis

```python
result = await swarm.execute(
    ticker="RELIANCE",
    exchange="NSE"
)

# Results
result.current_price         # Latest price
result.target_price          # Analyst target
result.rating                # BUY/HOLD/SELL
result.rating_confidence     # 0-1 confidence
result.investment_thesis     # List of reasons
result.key_risks            # List of risks
result.sentiment_score      # -1 to 1
result.pdf_path             # Report location
```

## 📊 Output Includes

- Executive summary with rating
- Investment thesis and risks
- Technical analysis with charts
- Peer comparison table
- Sentiment analysis from news
- Financial metrics
- Price targets and forecasts

## 📄 License

Part of Jotty AI Framework
