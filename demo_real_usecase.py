#!/usr/bin/env python3
"""
REAL USE CASE: Stock Analysis Pipeline with World-Class Swarm
==============================================================

This demo shows the swarm solving a REAL problem:
- Fetch stock data
- Analyze trends
- Generate buy/sell signals
- Validate recommendations

Watch how the swarm features help in practice!
"""

import sys
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Colors for output
class C:
    H = '\033[95m'  # Header
    B = '\033[94m'  # Blue
    C = '\033[96m'  # Cyan
    G = '\033[92m'  # Green
    Y = '\033[93m'  # Yellow
    R = '\033[91m'  # Red
    BOLD = '\033[1m'
    E = '\033[0m'   # End

def log(msg: str, color: str = ""):
    ts = time.strftime("%H:%M:%S")
    print(f"{color}[{ts}] {msg}{C.E}")
    sys.stdout.flush()

def section(title: str):
    print(f"\n{C.BOLD}{C.H}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{C.E}\n")
    sys.stdout.flush()

# =============================================================================
# SIMULATED STOCK DATA (Real-ish data for demo)
# =============================================================================

STOCKS = {
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology'},
    'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automotive'},
    'JPM': {'name': 'JPMorgan Chase', 'sector': 'Finance'},
    'NVDA': {'name': 'NVIDIA Corp.', 'sector': 'Technology'},
}

def generate_stock_data(symbol: str, days: int = 30) -> List[Dict]:
    """Generate realistic-looking stock data."""
    base_prices = {'AAPL': 175, 'GOOGL': 140, 'TSLA': 250, 'JPM': 150, 'NVDA': 450}
    base = base_prices.get(symbol, 100)

    data = []
    price = base
    for i in range(days):
        date = (datetime.now() - timedelta(days=days-i)).strftime('%Y-%m-%d')
        change = random.gauss(0, 0.02)  # 2% daily volatility
        price = price * (1 + change)
        volume = random.randint(10_000_000, 50_000_000)

        data.append({
            'date': date,
            'open': round(price * (1 - random.random() * 0.01), 2),
            'high': round(price * (1 + random.random() * 0.02), 2),
            'low': round(price * (1 - random.random() * 0.02), 2),
            'close': round(price, 2),
            'volume': volume
        })
    return data

# =============================================================================
# AGENT IMPLEMENTATIONS (Simulated but realistic)
# =============================================================================

@dataclass
class AgentResult:
    success: bool
    data: Any
    error: str = ""
    execution_time: float = 0.0
    claimed_success: bool = True  # For Byzantine testing

class DataFetchAgent:
    """Fetches stock data from 'API'."""
    name = "DataFetchAgent"

    def execute(self, task: Dict) -> AgentResult:
        symbol = task.get('symbol', 'AAPL')
        log(f"  📡 {self.name}: Fetching data for {symbol}...", C.B)
        time.sleep(0.3)  # Simulate API call

        # 90% success rate
        if random.random() > 0.1:
            data = generate_stock_data(symbol, 30)
            log(f"  ✅ {self.name}: Got {len(data)} days of data for {symbol}", C.G)
            return AgentResult(success=True, data={'symbol': symbol, 'prices': data})
        else:
            log(f"  ❌ {self.name}: API timeout for {symbol}", C.R)
            return AgentResult(success=False, data=None, error="API timeout")

class TechnicalAnalysisAgent:
    """Calculates technical indicators."""
    name = "TechnicalAnalysisAgent"

    def execute(self, task: Dict) -> AgentResult:
        prices = task.get('prices', [])
        symbol = task.get('symbol', '???')

        log(f"  📊 {self.name}: Analyzing {symbol} technicals...", C.B)
        time.sleep(0.4)

        if not prices or len(prices) < 10:
            return AgentResult(success=False, data=None, error="Insufficient data")

        # Calculate real indicators
        closes = [p['close'] for p in prices]

        # SMA 10 and SMA 20
        sma10 = sum(closes[-10:]) / 10
        sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else sma10

        # RSI (simplified)
        gains = [max(0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
        losses = [max(0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0.01
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0.01
        rs = avg_gain / max(avg_loss, 0.01)
        rsi = 100 - (100 / (1 + rs))

        # Trend
        trend = "BULLISH" if sma10 > sma20 else "BEARISH"

        analysis = {
            'symbol': symbol,
            'current_price': closes[-1],
            'sma10': round(sma10, 2),
            'sma20': round(sma20, 2),
            'rsi': round(rsi, 1),
            'trend': trend,
            'volatility': round(sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes))) / len(closes) * 100, 2)
        }

        log(f"  ✅ {self.name}: {symbol} - RSI={analysis['rsi']}, Trend={trend}", C.G)
        return AgentResult(success=True, data=analysis)

class SentimentAgent:
    """Analyzes market sentiment (simulated)."""
    name = "SentimentAgent"

    def execute(self, task: Dict) -> AgentResult:
        symbol = task.get('symbol', '???')

        log(f"  🗞️  {self.name}: Checking sentiment for {symbol}...", C.B)
        time.sleep(0.3)

        # Simulate sentiment analysis
        sentiment_score = random.gauss(0.1, 0.3)  # Slightly bullish on average
        sentiment_score = max(-1, min(1, sentiment_score))

        news_items = random.randint(5, 20)

        sentiment = {
            'symbol': symbol,
            'score': round(sentiment_score, 2),
            'label': 'POSITIVE' if sentiment_score > 0.1 else ('NEGATIVE' if sentiment_score < -0.1 else 'NEUTRAL'),
            'news_count': news_items,
            'social_mentions': random.randint(100, 5000)
        }

        log(f"  ✅ {self.name}: {symbol} sentiment = {sentiment['label']} ({sentiment['score']})", C.G)
        return AgentResult(success=True, data=sentiment)

class SignalGeneratorAgent:
    """Generates trading signals based on analysis."""
    name = "SignalGeneratorAgent"

    def execute(self, task: Dict) -> AgentResult:
        technical = task.get('technical', {})
        sentiment = task.get('sentiment', {})
        symbol = task.get('symbol', '???')

        log(f"  🎯 {self.name}: Generating signal for {symbol}...", C.B)
        time.sleep(0.2)

        # Scoring system
        score = 0
        reasons = []

        # Technical factors
        rsi = technical.get('rsi', 50)
        if rsi < 30:
            score += 2
            reasons.append(f"RSI oversold ({rsi})")
        elif rsi > 70:
            score -= 2
            reasons.append(f"RSI overbought ({rsi})")

        trend = technical.get('trend', 'NEUTRAL')
        if trend == 'BULLISH':
            score += 1
            reasons.append("Bullish trend (SMA10 > SMA20)")
        elif trend == 'BEARISH':
            score -= 1
            reasons.append("Bearish trend (SMA10 < SMA20)")

        # Sentiment factors
        sent_score = sentiment.get('score', 0)
        if sent_score > 0.2:
            score += 1
            reasons.append(f"Positive sentiment ({sent_score})")
        elif sent_score < -0.2:
            score -= 1
            reasons.append(f"Negative sentiment ({sent_score})")

        # Generate signal
        if score >= 2:
            signal = 'STRONG_BUY'
        elif score == 1:
            signal = 'BUY'
        elif score == -1:
            signal = 'SELL'
        elif score <= -2:
            signal = 'STRONG_SELL'
        else:
            signal = 'HOLD'

        result = {
            'symbol': symbol,
            'signal': signal,
            'score': score,
            'confidence': min(100, 50 + abs(score) * 15),
            'reasons': reasons,
            'current_price': technical.get('current_price', 0),
            'target_price': round(technical.get('current_price', 100) * (1 + score * 0.03), 2)
        }

        log(f"  ✅ {self.name}: {symbol} → {signal} (confidence: {result['confidence']}%)", C.G)
        return AgentResult(success=True, data=result)

class ValidationAgent:
    """Validates recommendations against risk rules."""
    name = "ValidationAgent"
    is_unreliable = False  # For Byzantine testing

    def execute(self, task: Dict) -> AgentResult:
        signal_data = task.get('signal', {})
        symbol = signal_data.get('symbol', '???')

        log(f"  🔍 {self.name}: Validating {symbol} recommendation...", C.B)
        time.sleep(0.2)

        issues = []

        # Risk checks
        confidence = signal_data.get('confidence', 0)
        if confidence < 60:
            issues.append(f"Low confidence ({confidence}%)")

        signal = signal_data.get('signal', 'HOLD')
        if signal in ['STRONG_BUY', 'STRONG_SELL'] and confidence < 70:
            issues.append("Strong signal with insufficient confidence")

        # Simulate occasional Byzantine behavior
        actual_valid = len(issues) == 0

        if self.is_unreliable and random.random() > 0.6:
            # Lie about validation
            claimed_valid = not actual_valid
            log(f"  ⚠️  {self.name}: [LYING] Claiming {'valid' if claimed_valid else 'invalid'}", C.Y)
        else:
            claimed_valid = actual_valid

        if claimed_valid:
            log(f"  ✅ {self.name}: {symbol} recommendation VALIDATED", C.G)
        else:
            log(f"  ⚠️  {self.name}: {symbol} has issues: {issues}", C.Y)

        return AgentResult(
            success=True,
            data={'valid': actual_valid, 'issues': issues, 'symbol': symbol},
            claimed_success=claimed_valid
        )

# =============================================================================
# SWARM COORDINATOR (Uses all world-class features)
# =============================================================================

class StockAnalysisSwarm:
    """Coordinates agents to analyze stocks using world-class swarm features."""

    def __init__(self):
        # Import swarm intelligence
        from core.orchestration.v2.swarm_intelligence import SwarmIntelligence
        from core.foundation.robust_parsing import AdaptiveWeightGroup

        self.si = SwarmIntelligence()

        # Adaptive weights for agent selection
        self.agent_weights = AdaptiveWeightGroup({
            'historical_success': 0.4,
            'stigmergy_signal': 0.3,
            'trust_score': 0.3
        })

        # Initialize agents
        self.agents = {
            'data_fetch': DataFetchAgent(),
            'technical': TechnicalAnalysisAgent(),
            'sentiment': SentimentAgent(),
            'signal': SignalGeneratorAgent(),
            'validation': ValidationAgent(),
        }

        # Register with swarm intelligence
        for name in self.agents:
            self.si.register_agent(name)

        # Track results for learning
        self.pipeline_runs = 0
        self.successful_runs = 0

    def analyze_stock(self, symbol: str) -> Dict:
        """Run full analysis pipeline for a stock."""
        self.pipeline_runs += 1

        log(f"🚀 Starting analysis pipeline for {symbol}", C.BOLD + C.C)
        pipeline_start = time.time()

        results = {'symbol': symbol, 'stages': {}, 'success': False}

        # Stage 1: Fetch Data
        log("\n📌 STAGE 1: Data Fetching", C.BOLD)

        # Check stigmergy for best data fetcher
        rec = self.si.get_stigmergy_recommendation('data_fetch')
        if rec:
            log(f"  💡 Stigmergy recommends: {rec} for data fetching", C.C)

        fetch_result = self.agents['data_fetch'].execute({'symbol': symbol})
        self._record_result('data_fetch', 'data_fetch', fetch_result)

        if not fetch_result.success:
            results['error'] = f"Data fetch failed: {fetch_result.error}"
            return results

        results['stages']['data_fetch'] = fetch_result.data

        # Stage 2: Technical Analysis (parallel with Sentiment)
        log("\n📌 STAGE 2: Analysis (Technical + Sentiment)", C.BOLD)

        tech_result = self.agents['technical'].execute({
            'symbol': symbol,
            'prices': fetch_result.data['prices']
        })
        self._record_result('technical', 'analysis', tech_result)

        sent_result = self.agents['sentiment'].execute({'symbol': symbol})
        self._record_result('sentiment', 'analysis', sent_result)

        if not tech_result.success:
            results['error'] = f"Technical analysis failed: {tech_result.error}"
            return results

        results['stages']['technical'] = tech_result.data
        results['stages']['sentiment'] = sent_result.data

        # Stage 3: Signal Generation
        log("\n📌 STAGE 3: Signal Generation", C.BOLD)

        signal_result = self.agents['signal'].execute({
            'symbol': symbol,
            'technical': tech_result.data,
            'sentiment': sent_result.data
        })
        self._record_result('signal', 'signal_generation', signal_result)

        results['stages']['signal'] = signal_result.data

        # Stage 4: Validation with Byzantine Check
        log("\n📌 STAGE 4: Validation (with Byzantine Verification)", C.BOLD)

        val_result = self.agents['validation'].execute({
            'signal': signal_result.data
        })

        # Byzantine verification
        actual_valid = val_result.data.get('valid', False)
        claimed_valid = val_result.claimed_success

        is_consistent = self.si.byzantine.verify_claim(
            agent='validation',
            claimed_success=claimed_valid,
            actual_result={'success': actual_valid},
            task_type='validation'
        )

        if not is_consistent:
            log(f"  🚨 BYZANTINE ALERT: Validator claim inconsistent!", C.R)
            trust = self.si.agent_profiles['validation'].trust_score
            log(f"  📉 Validator trust now: {trust:.2f}", C.Y)

        results['stages']['validation'] = val_result.data
        results['validation_consistent'] = is_consistent

        # Final result
        pipeline_time = time.time() - pipeline_start
        results['success'] = True
        results['execution_time'] = round(pipeline_time, 2)

        # Record benchmark
        self.si.benchmarks.record_multi_agent_run(
            task_type='stock_analysis',
            execution_time=pipeline_time,
            agents_count=len(self.agents),
            success=True
        )

        self.successful_runs += 1

        # Update adaptive weights based on success
        self.agent_weights.update_from_feedback('stigmergy_signal', 0.1, reward=1.0)

        return results

    def _record_result(self, agent_name: str, task_type: str, result: AgentResult):
        """Record result in swarm intelligence."""
        self.si.record_task_result(
            agent_name=agent_name,
            task_type=task_type,
            success=result.success,
            execution_time=result.execution_time or 0.3,
            is_multi_agent=True,
            agents_count=len(self.agents)
        )

    def get_swarm_report(self) -> str:
        """Generate report on swarm performance."""
        metrics = self.si.benchmarks.compute_metrics(self.si.agent_profiles)

        lines = [
            f"\n{C.BOLD}📊 SWARM PERFORMANCE REPORT{C.E}",
            f"{'='*50}",
            f"Pipelines run: {self.pipeline_runs}",
            f"Successful: {self.successful_runs} ({self.successful_runs/max(1,self.pipeline_runs)*100:.0f}%)",
            f"",
            f"{C.C}Adaptive Weights:{C.E}",
            f"  {self.agent_weights}",
            f"",
            f"{C.C}Agent Trust Scores:{C.E}",
        ]

        for name, profile in self.si.agent_profiles.items():
            lines.append(f"  {name}: {profile.trust_score:.2f} (tasks: {profile.total_tasks})")

        lines.append(f"\n{C.C}Stigmergy Signals:{C.E} {len(self.si.stigmergy.signals)}")

        # Show routing recommendations
        lines.append(f"\n{C.C}Routing Recommendations:{C.E}")
        for task_type in ['data_fetch', 'analysis', 'validation']:
            rec = self.si.get_stigmergy_recommendation(task_type)
            if rec:
                lines.append(f"  {task_type} → {rec}")

        return "\n".join(lines)

# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    print(f"{C.BOLD}{C.H}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║     REAL USE CASE: Stock Analysis with World-Class Swarm          ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(C.E)

    swarm = StockAnalysisSwarm()

    # Analyze multiple stocks
    symbols = ['AAPL', 'NVDA', 'TSLA', 'GOOGL', 'JPM']
    all_results = []

    section("RUNNING STOCK ANALYSIS PIPELINE")

    for i, symbol in enumerate(symbols):
        log(f"\n{'='*60}", C.H)
        log(f"ANALYZING {symbol} ({STOCKS[symbol]['name']}) - {i+1}/{len(symbols)}", C.BOLD + C.H)
        log(f"{'='*60}", C.H)

        # Make validator unreliable for one stock to show Byzantine detection
        if symbol == 'TSLA':
            swarm.agents['validation'].is_unreliable = True
            log("⚠️  [TEST] Making validator unreliable for this stock", C.Y)
        else:
            swarm.agents['validation'].is_unreliable = False

        result = swarm.analyze_stock(symbol)
        all_results.append(result)

        # Show result summary
        if result['success']:
            signal_data = result['stages'].get('signal', {})
            tech_data = result['stages'].get('technical', {})

            log(f"\n📋 RESULT FOR {symbol}:", C.BOLD + C.G)
            log(f"   Price: ${tech_data.get('current_price', 0):.2f}", C.G)
            log(f"   Signal: {signal_data.get('signal', 'N/A')}", C.G)
            log(f"   Confidence: {signal_data.get('confidence', 0)}%", C.G)
            log(f"   Target: ${signal_data.get('target_price', 0):.2f}", C.G)
            log(f"   Reasons: {', '.join(signal_data.get('reasons', []))}", C.C)
        else:
            log(f"\n❌ FAILED: {result.get('error', 'Unknown error')}", C.R)

        time.sleep(0.5)

    # Final report
    section("FINAL SWARM REPORT")
    print(swarm.get_swarm_report())

    # Show all recommendations
    section("INVESTMENT RECOMMENDATIONS")

    print(f"{C.BOLD}{'Symbol':<8} {'Signal':<12} {'Confidence':<12} {'Price':<10} {'Target':<10}{C.E}")
    print("-" * 55)

    for result in all_results:
        if result['success']:
            sig = result['stages']['signal']
            symbol = sig['symbol']
            signal = sig['signal']
            conf = sig['confidence']
            price = sig['current_price']
            target = sig['target_price']

            color = C.G if 'BUY' in signal else (C.R if 'SELL' in signal else C.Y)
            print(f"{color}{symbol:<8} {signal:<12} {conf}%{'':<9} ${price:<9.2f} ${target:<10.2f}{C.E}")

    # Show world-class features in action
    section("WORLD-CLASS FEATURES DEMONSTRATED")

    features = [
        ("✅ ADAPTIVE WEIGHTS", f"Agent selection weights: {swarm.agent_weights}"),
        ("✅ STIGMERGY", f"Deposited {len(swarm.si.stigmergy.signals)} pheromone signals for routing"),
        ("✅ BYZANTINE", f"Validator trust after checks: {swarm.si.agent_profiles['validation'].trust_score:.2f}"),
        ("✅ BENCHMARKS", f"Tracked {swarm.pipeline_runs} pipeline runs"),
        ("✅ SPECIALIZATION", "Agents tracked by task type performance"),
    ]

    for name, desc in features:
        log(f"{name}: {desc}", C.G)

    print(f"\n{C.BOLD}{C.G}🏆 Real stock analysis completed using world-class swarm!{C.E}\n")

    return 0

if __name__ == '__main__':
    sys.exit(main())
