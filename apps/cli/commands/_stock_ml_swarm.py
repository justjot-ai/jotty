"""
Stock ML - Swarm Learning Methods
====================================

Swarm-based learning, stock profiling, config pattern analysis,
and Q-learning for stock ML optimization.
"""

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

from .base import CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class StockMLSwarmMixin:
    """Mixin providing swarm learning methods for StockMLCommand.

    These methods implement swarm-based optimization and learning
    for stock ML configurations.
    """

    async def _run_swarm_learning(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """
        Run SwarmML auto-learning pipeline.

        Analyzes sweep results and feature importance patterns to:
        1. Learn which configs work best for which stock characteristics
        2. Identify underperforming stocks and suggest improvements
        3. Build transferable knowledge across stocks
        """
        from datetime import datetime

        import numpy as np

        cli.renderer.header("SwarmML Auto-Learning")
        cli.renderer.info("")

        # Load existing sweep results
        if not self.SWEEP_RESULTS_FILE.exists():
            cli.renderer.error("No sweep results found. Run: /stock-ml --sweep --stocks top10")
            return CommandResult.fail("No sweep results")

        with open(self.SWEEP_RESULTS_FILE) as f:
            sweep_results = json.load(f)

        if not sweep_results:
            cli.renderer.error("Empty sweep results")
            return CommandResult.fail("No results")

        cli.renderer.info(f"Analyzing {len(sweep_results)} sweep results...")

        # Load or initialize swarm learning state
        swarm_state = self._load_swarm_ml_state()

        # Initialize Q-Learner for real RL
        q_learner = StockMLQLearner()
        cli.renderer.info(
            f"Q-Learner: {q_learner.get_stats()['entries']} entries, ε={q_learner.epsilon:.3f}"
        )

        # ============ Phase 1: Learn Stock Characteristics ============
        cli.renderer.info("")
        cli.renderer.info("Phase 1: Learning Stock Characteristics")

        stock_profiles = self._analyze_stock_profiles(sweep_results)
        swarm_state["stock_profiles"] = stock_profiles

        cli.renderer.info(f"  Analyzed {len(stock_profiles)} stock profiles")

        # ============ Phase 2: Learn Config Patterns ============
        cli.renderer.info("")
        cli.renderer.info("Phase 2: Learning Config Patterns")

        config_patterns = self._analyze_config_patterns(sweep_results)
        swarm_state["config_patterns"] = config_patterns

        cli.renderer.info(f"  Found {len(config_patterns['best_by_sector'])} sector patterns")
        cli.renderer.info(f"  Target rankings: {list(config_patterns['target_rankings'].keys())}")

        # ============ Phase 2b: Q-Learning Updates ============
        cli.renderer.info("")
        cli.renderer.info("Phase 2b: Q-Learning from Results")

        updates_count = 0
        for r in sweep_results:
            symbol = r.get("symbol", "UNKNOWN")
            target = r.get("target", "next_5d_up")
            timeframe = r.get("timeframe", "day")
            auc = r.get("auc", 0.5)

            # Get state from profile
            profile = stock_profiles.get(symbol, {})
            if not profile:
                continue

            state = q_learner.get_state(profile)
            action = q_learner.get_action_key(target, timeframe)
            reward = q_learner.compute_reward({"auc": auc})

            # Q-learning update
            q_learner.update(state, action, reward)
            updates_count += 1

        cli.renderer.info(f"  Q-learning updates: {updates_count}")
        cli.renderer.info(f"  Q-table size: {q_learner.get_stats()['entries']}")
        cli.renderer.info(f"  Avg Q-value: {q_learner.get_stats()['avg_q']:.4f}")

        # Store Q-learner reference in state
        swarm_state["q_learner_stats"] = q_learner.get_stats()

        # ============ Phase 3: Identify Underperformers ============
        cli.renderer.info("")
        cli.renderer.info("Phase 3: Identifying Underperformers")

        underperformers = self._identify_underperformers(sweep_results, stock_profiles)
        swarm_state["underperformers"] = underperformers

        cli.renderer.info(f"  Found {len(underperformers)} underperforming configs")

        # ============ Phase 4: Generate Recommendations ============
        cli.renderer.info("")
        cli.renderer.info("Phase 4: Generating Recommendations")

        recommendations = self._generate_recommendations(
            sweep_results, stock_profiles, config_patterns, underperformers
        )
        swarm_state["recommendations"] = recommendations
        swarm_state["last_updated"] = datetime.now().isoformat()

        # Save learning state
        self._save_swarm_ml_state(swarm_state)
        cli.renderer.info("")
        cli.renderer.success(f"Learnings saved to {self.SWARM_ML_STATE_FILE}")

        # ============ Display Results ============
        cli.renderer.info("")
        cli.renderer.header("SwarmML Insights")

        # Top patterns
        cli.renderer.info("")
        cli.renderer.info("Top Performing Patterns:")
        for pattern in config_patterns.get("top_patterns", [])[:5]:
            cli.renderer.info(
                f"  {pattern['pattern']}: AUC {pattern['avg_auc']:.4f} ({pattern['count']} samples)"
            )

        # Sector insights
        cli.renderer.info("")
        cli.renderer.info("Best Target by Sector:")
        for sector, best in list(config_patterns["best_by_sector"].items())[:8]:
            cli.renderer.info(f"  {sector:<15}: {best['target']:<12} (AUC {best['auc']:.4f})")

        # Recommendations
        cli.renderer.info("")
        cli.renderer.info("Top Recommendations:")
        for rec in recommendations[:5]:
            cli.renderer.info(f"  {rec['symbol']}: {rec['recommendation']}")
            if "suggested_config" in rec:
                cli.renderer.info(f"    -> Try: {rec['suggested_config']}")

        # Cross-stock learnings
        cli.renderer.info("")
        cli.renderer.info("Cross-Stock Learnings:")
        cli.renderer.info(
            f"  Best overall timeframe: {config_patterns.get('best_timeframe', 'day')}"
        )
        cli.renderer.info(
            f"  Best overall target: {config_patterns.get('best_target', 'next_1d_up')}"
        )
        cli.renderer.info(
            f"  Optimal feature correlation: {config_patterns.get('feature_insight', 'momentum features dominate')}"
        )

        # Q-Learning Insights
        cli.renderer.info("")
        cli.renderer.header("Q-Learning Strategy Recommendations")

        # Show Q-recommendations by sector
        for sector in ["banking", "it", "fmcg", "auto", "other"]:
            # Create mock profile for sector query
            mock_profile = {
                "sector": sector,
                "predictability": "medium",
                "auc_std": 0.05,
                "target_sensitivity": {},
            }
            state = q_learner.get_state(mock_profile)
            recs = q_learner.get_recommendations(state, top_k=2)

            if recs and recs[0]["confidence"] > 0:
                cli.renderer.info(f"  {sector.upper()}:")
                for rec in recs[:2]:
                    conf_str = (
                        f"({rec['confidence']*100:.0f}% confident)"
                        if rec["confidence"] > 0.3
                        else "(exploring)"
                    )
                    cli.renderer.info(
                        f"    {rec['target']} + {rec['timeframe']}: Q={rec['q_value']:.3f} {conf_str}"
                    )

        # Transfer learning context
        cli.renderer.info("")
        cli.renderer.info("Transfer Learning:")
        context = q_learner.get_transfer_learning_context("banking|medium|high|medium")
        if context:
            for line in context.split("\n")[:4]:
                cli.renderer.info(f"  {line}")
        else:
            cli.renderer.info("  (Not enough data yet - run more sweeps)")

        return CommandResult.ok(
            data={
                "stock_profiles": stock_profiles,
                "config_patterns": config_patterns,
                "underperformers": underperformers,
                "recommendations": recommendations,
            }
        )

    def _analyze_stock_profiles(self, sweep_results: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze each stock's performance characteristics.

        Groups stocks by their optimal configs and performance patterns.
        """
        import numpy as np

        profiles = {}

        # Group results by stock
        by_stock = {}
        for r in sweep_results:
            symbol = r.get("symbol", "UNKNOWN")
            if symbol not in by_stock:
                by_stock[symbol] = []
            by_stock[symbol].append(r)

        for symbol, results in by_stock.items():
            if not results:
                continue

            aucs = [r.get("auc", 0) for r in results]
            accuracies = [r.get("accuracy", 0) for r in results]

            # Find best config for this stock
            best_result = max(results, key=lambda x: x.get("auc", 0))

            # Analyze sensitivity to different parameters
            by_target = {}
            by_timeframe = {}

            for r in results:
                target = r.get("target", "unknown")
                timeframe = r.get("timeframe", "day")

                if target not in by_target:
                    by_target[target] = []
                by_target[target].append(r.get("auc", 0))

                if timeframe not in by_timeframe:
                    by_timeframe[timeframe] = []
                by_timeframe[timeframe].append(r.get("auc", 0))

            # Compute averages
            target_avg = {t: np.mean(aucs) for t, aucs in by_target.items()}
            timeframe_avg = {t: np.mean(aucs) for t, aucs in by_timeframe.items()}

            # Determine stock characteristics
            best_target = (
                max(target_avg.items(), key=lambda x: x[1])[0] if target_avg else "next_1d_up"
            )
            best_timeframe = (
                max(timeframe_avg.items(), key=lambda x: x[1])[0] if timeframe_avg else "day"
            )

            # Classify stock predictability
            max_auc = max(aucs) if aucs else 0
            if max_auc >= 0.70:
                predictability = "high"
            elif max_auc >= 0.60:
                predictability = "medium"
            else:
                predictability = "low"

            # Determine sector (heuristic based on stock name)
            sector = self._infer_sector(symbol)

            profiles[symbol] = {
                "best_auc": max(aucs) if aucs else 0,
                "avg_auc": np.mean(aucs) if aucs else 0,
                "auc_std": np.std(aucs) if aucs else 0,
                "best_target": best_target,
                "best_timeframe": best_timeframe,
                "predictability": predictability,
                "sector": sector,
                "n_configs_tested": len(results),
                "target_sensitivity": target_avg,
                "timeframe_sensitivity": timeframe_avg,
                "best_config": {
                    "target": best_result.get("target"),
                    "timeframe": best_result.get("timeframe"),
                    "years": best_result.get("years"),
                    "auc": best_result.get("auc"),
                },
            }

        return profiles

    def _infer_sector(self, symbol: str) -> str:
        """Infer sector from stock symbol."""
        # Check against known sets
        for sector, stocks in self.STOCK_SETS.items():
            if symbol in stocks:
                if "bank" in sector.lower() or sector in ["banks"]:
                    return "banking"
                if "it" in sector.lower():
                    return "it"
                if "pharma" in sector.lower():
                    return "pharma"
                if "auto" in sector.lower():
                    return "auto"
                if "fmcg" in sector.lower():
                    return "fmcg"
                if "metal" in sector.lower():
                    return "metal"
                if "energy" in sector.lower():
                    return "energy"
                if "infra" in sector.lower() or "realty" in sector.lower():
                    return "infra"

        # Keyword matching
        symbol_upper = symbol.upper()
        if any(x in symbol_upper for x in ["BANK", "FIN"]):
            return "banking"
        if any(x in symbol_upper for x in ["TECH", "INFO", "SOFT", "WIPRO", "INFY", "TCS"]):
            return "it"
        if any(x in symbol_upper for x in ["PHARMA", "SUN", "CIPLA", "LUPIN"]):
            return "pharma"
        if any(x in symbol_upper for x in ["AUTO", "MOTOR", "MARUTI", "TATA"]):
            return "auto"

        return "other"

    def _analyze_config_patterns(self, sweep_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze patterns across all configs to find what works best.
        """
        import numpy as np

        patterns = {
            "target_rankings": {},
            "timeframe_rankings": {},
            "best_by_sector": {},
            "top_patterns": [],
            "best_timeframe": "day",
            "best_target": "next_1d_up",
            "feature_insight": "",
        }

        # Group by target
        by_target = {}
        for r in sweep_results:
            target = r.get("target", "unknown")
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(r.get("auc", 0))

        patterns["target_rankings"] = {
            t: {"avg_auc": np.mean(aucs), "count": len(aucs)} for t, aucs in by_target.items()
        }

        if patterns["target_rankings"]:
            patterns["best_target"] = max(
                patterns["target_rankings"].items(), key=lambda x: x[1]["avg_auc"]
            )[0]

        # Group by timeframe
        by_timeframe = {}
        for r in sweep_results:
            tf = r.get("timeframe", "day")
            if tf not in by_timeframe:
                by_timeframe[tf] = []
            by_timeframe[tf].append(r.get("auc", 0))

        patterns["timeframe_rankings"] = {
            t: {"avg_auc": np.mean(aucs), "count": len(aucs)} for t, aucs in by_timeframe.items()
        }

        if patterns["timeframe_rankings"]:
            patterns["best_timeframe"] = max(
                patterns["timeframe_rankings"].items(), key=lambda x: x[1]["avg_auc"]
            )[0]

        # Group by sector (using inferred sectors)
        by_sector = {}
        for r in sweep_results:
            sector = self._infer_sector(r.get("symbol", ""))
            if sector not in by_sector:
                by_sector[sector] = []
            by_sector[sector].append(r)

        for sector, results in by_sector.items():
            if not results:
                continue
            best = max(results, key=lambda x: x.get("auc", 0))
            patterns["best_by_sector"][sector] = {
                "target": best.get("target"),
                "timeframe": best.get("timeframe"),
                "auc": best.get("auc", 0),
                "symbol": best.get("symbol"),
            }

        # Find top patterns (target + timeframe combos)
        combo_results = {}
        for r in sweep_results:
            combo = f"{r.get('target', 'unknown')}_{r.get('timeframe', 'day')}"
            if combo not in combo_results:
                combo_results[combo] = []
            combo_results[combo].append(r.get("auc", 0))

        patterns["top_patterns"] = sorted(
            [
                {"pattern": k, "avg_auc": np.mean(v), "count": len(v)}
                for k, v in combo_results.items()
            ],
            key=lambda x: -x["avg_auc"],
        )

        # Feature insight (based on patterns)
        best_tf = patterns["best_timeframe"]
        best_tgt = patterns["best_target"]
        if "60minute" in best_tf or "hour" in best_tf:
            patterns["feature_insight"] = (
                "Intraday patterns are predictive; focus on short-term momentum"
            )
        elif "1d" in best_tgt:
            patterns["feature_insight"] = (
                "Short-term momentum features dominate; RSI, MACD effective"
            )
        elif "20d" in best_tgt or "30d" in best_tgt:
            patterns["feature_insight"] = "Trend-following features important; moving averages, ADX"
        else:
            patterns["feature_insight"] = "Mixed signals; consider ensemble of timeframes"

        return patterns

    def _identify_underperformers(
        self, sweep_results: List[Dict], stock_profiles: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Identify underperforming stocks and configs.
        """
        import numpy as np

        underperformers = []

        # Calculate overall baseline
        all_aucs = [r.get("auc", 0) for r in sweep_results]
        baseline_auc = np.median(all_aucs) if all_aucs else 0.5
        threshold = baseline_auc - 0.05  # Underperform if 5% below median

        # Find stocks consistently below threshold
        for symbol, profile in stock_profiles.items():
            if profile["best_auc"] < threshold:
                underperformers.append(
                    {
                        "symbol": symbol,
                        "best_auc": profile["best_auc"],
                        "gap": baseline_auc - profile["best_auc"],
                        "sector": profile["sector"],
                        "reason": self._diagnose_underperformance(profile),
                    }
                )

        # Sort by gap (worst first)
        underperformers.sort(key=lambda x: -x["gap"])

        return underperformers

    def _generate_recommendations(
        self,
        sweep_results: List[Dict],
        stock_profiles: Dict[str, Dict],
        config_patterns: Dict[str, Any],
        underperformers: List[Dict],
    ) -> List[Dict]:
        """
        Generate actionable recommendations based on learnings.
        """
        recommendations = []

        # 1. Recommendations for underperformers
        for up in underperformers[:10]:
            symbol = up["symbol"]
            profile = stock_profiles.get(symbol, {})
            sector = profile.get("sector", "other")

            # Find what works for similar stocks in same sector
            sector_best = config_patterns["best_by_sector"].get(sector, {})

            rec = {
                "symbol": symbol,
                "current_best_auc": up["best_auc"],
                "recommendation": up["reason"],
                "priority": "high" if up["gap"] > 0.1 else "medium",
            }

            # Suggest alternative config based on sector patterns
            if sector_best and sector_best.get("auc", 0) > up["best_auc"]:
                rec["suggested_config"] = (
                    f"target={sector_best['target']}, " f"timeframe={sector_best['timeframe']}"
                )

            recommendations.append(rec)

        # 2. General recommendations
        best_tf = config_patterns.get("best_timeframe", "day")
        best_tgt = config_patterns.get("best_target", "next_1d_up")

        if best_tf == "60minute":
            recommendations.append(
                {
                    "symbol": "ALL",
                    "recommendation": f"60-minute data consistently outperforms daily; migrate all stocks to {best_tf}",
                    "priority": "high",
                }
            )

        if best_tgt == "next_1d_up":
            recommendations.append(
                {
                    "symbol": "ALL",
                    "recommendation": "Short-term (1-day) predictions most reliable; avoid long-horizon targets",
                    "priority": "medium",
                }
            )

        # 3. Sector-specific recommendations
        for sector, best in config_patterns.get("best_by_sector", {}).items():
            if best.get("auc", 0) >= 0.65:
                recommendations.append(
                    {
                        "symbol": f"{sector.upper()} SECTOR",
                        "recommendation": f"Focus on {best['target']} with {best['timeframe']} data (AUC {best['auc']:.4f})",
                        "priority": "medium",
                    }
                )

        return recommendations

    def _show_swarm_insights(self, cli: "JottyCLI") -> CommandResult:
        """Display current swarm ML insights."""
        if not self.SWARM_ML_STATE_FILE.exists():
            cli.renderer.info("No swarm learnings yet. Run: /stock-ml --swarm-learn")
            return CommandResult.ok(data={})

        state = self._load_swarm_ml_state()

        cli.renderer.header("SwarmML Insights Dashboard")
        cli.renderer.info(f"Last updated: {state.get('last_updated', 'Never')}")
        cli.renderer.info("")

        # Stock profiles summary
        profiles = state.get("stock_profiles", {})
        if profiles:
            cli.renderer.info("Stock Predictability Tiers")

            high = [s for s, p in profiles.items() if p.get("predictability") == "high"]
            medium = [s for s, p in profiles.items() if p.get("predictability") == "medium"]
            low = [s for s, p in profiles.items() if p.get("predictability") == "low"]

            cli.renderer.info(f"  High ({len(high)}):   {', '.join(high[:10])}")
            cli.renderer.info(f"  Medium ({len(medium)}): {', '.join(medium[:10])}")
            cli.renderer.info(f"  Low ({len(low)}):    {', '.join(low[:10])}")

        # Config patterns
        patterns = state.get("config_patterns", {})
        if patterns:
            cli.renderer.info("")
            cli.renderer.info("Optimal Configurations")
            cli.renderer.info(f"  Best timeframe: {patterns.get('best_timeframe', 'N/A')}")
            cli.renderer.info(f"  Best target: {patterns.get('best_target', 'N/A')}")
            cli.renderer.info(f"  Insight: {patterns.get('feature_insight', 'N/A')}")

            cli.renderer.info("")
            cli.renderer.info("Target Rankings:")
            for target, stats in sorted(
                patterns.get("target_rankings", {}).items(), key=lambda x: -x[1].get("avg_auc", 0)
            )[:5]:
                cli.renderer.info(
                    f"    {target:<15}: AUC {stats['avg_auc']:.4f} ({stats['count']} tests)"
                )

        # Recommendations
        recs = state.get("recommendations", [])
        if recs:
            cli.renderer.info("")
            cli.renderer.info("Active Recommendations")
            for rec in recs[:8]:
                priority = rec.get("priority", "low")
                marker = "!" if priority == "high" else "-"
                cli.renderer.info(f"  {marker} {rec['symbol']}: {rec['recommendation']}")

        return CommandResult.ok(data=state)

    async def _run_swarm_refinement(
        self, symbol: str, args: ParsedArgs, cli: "JottyCLI"
    ) -> CommandResult:
        """
        Run swarm-guided refinement for a specific stock.

        Uses learnings to try alternative configs automatically.
        """
        cli.renderer.header(f"SwarmML Refinement: {symbol}")

        # Load learnings
        state = self._load_swarm_ml_state()
        profiles = state.get("stock_profiles", {})
        patterns = state.get("config_patterns", {})

        profile = profiles.get(symbol, {})
        if not profile:
            cli.renderer.info(f"No profile for {symbol}. Running initial sweep...")
            # Run quick sweep for this stock
            args.flags["stocks"] = symbol
            args.flags["sweep-targets"] = "next_1d_up,next_5d_up,next_10d_up"
            args.flags["sweep-timeframes"] = "day,60minute"
            return await self._run_sweep(args, cli)

        # Get recommendations for this stock's sector
        sector = profile.get("sector", "other")
        sector_best = patterns.get("best_by_sector", {}).get(sector, {})

        cli.renderer.info(f"Current best: AUC {profile['best_auc']:.4f}")
        cli.renderer.info(f"  Target: {profile['best_config'].get('target')}")
        cli.renderer.info(f"  Timeframe: {profile['best_config'].get('timeframe')}")
        cli.renderer.info("")

        # Try sector-recommended config if different
        if sector_best:
            suggested_target = sector_best.get("target")
            suggested_tf = sector_best.get("timeframe")

            if suggested_target != profile["best_config"].get("target") or suggested_tf != profile[
                "best_config"
            ].get("timeframe"):

                cli.renderer.info(f"Trying sector-recommended config:")
                cli.renderer.info(f"  Target: {suggested_target}")
                cli.renderer.info(f"  Timeframe: {suggested_tf}")
                cli.renderer.info("")

                # Load data and test
                years = int(args.flags.get("years", "3"))
                df = await self._load_stock_data(symbol, suggested_tf, years, cli)

                if df is not None and len(df) >= 100:
                    target_config = self._parse_target(suggested_target)
                    X, y, feature_names = self._create_features_and_target(df.copy(), target_config)

                    if X is not None and len(X) >= 100:
                        result = await self._quick_train(X, y, feature_names, target_config, cli)

                        new_auc = result.get("auc", 0)
                        improvement = new_auc - profile["best_auc"]

                        cli.renderer.info(f"Result: AUC {new_auc:.4f}")

                        if improvement > 0:
                            cli.renderer.info(f"IMPROVEMENT: +{improvement:.4f}")
                            cli.renderer.info("Updating profile...")

                            # Update profile
                            profile["best_auc"] = new_auc
                            profile["best_config"] = {
                                "target": suggested_target,
                                "timeframe": suggested_tf,
                                "years": years,
                                "auc": new_auc,
                            }
                            profiles[symbol] = profile
                            state["stock_profiles"] = profiles
                            self._save_swarm_ml_state(state)
                        else:
                            cli.renderer.info(f"No improvement ({improvement:.4f})")

        return CommandResult.ok(data={"symbol": symbol, "profile": profile})


# =============================================================================
# REAL Q-LEARNING INTEGRATION FOR STOCK ML
# =============================================================================


class StockMLQLearner:
    """
    Q-Learning for Stock ML Strategy Selection.

    Uses Jotty's Q-learning infrastructure to learn:
    - Which configs (target, timeframe) work for which stock characteristics
    - How to adapt strategies based on market conditions
    - Transfer learning across similar stocks

    State = (sector, volatility_regime, trend_strength, predictability_tier)
    Action = (target, timeframe, feature_set)
    Reward = AUC + Sharpe_bonus - drawdown_penalty
    """

    # Q-table persistence path
    Q_TABLE_PATH = Path.home() / ".jotty" / "stock_ml_qtable.json"


class StockMLQLearner:
    """
    Q-Learning for Stock ML Strategy Selection.

    Uses Jotty's Q-learning infrastructure to learn:
    - Which configs (target, timeframe) work for which stock characteristics
    - How to adapt strategies based on market conditions
    - Transfer learning across similar stocks

    State = (sector, volatility_regime, trend_strength, predictability_tier)
    Action = (target, timeframe, feature_set)
    Reward = AUC + Sharpe_bonus - drawdown_penalty
    """

    # Q-table persistence path
    Q_TABLE_PATH = Path.home() / ".jotty" / "stock_ml_qtable.json"

    def __init__(self) -> None:
        self.Q = {}  # State-Action -> (value, count, last_updated)
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor (lower for stock ML - focus on immediate)
        self.epsilon = 0.15  # Exploration rate

        # Load existing Q-table
        self._load_q_table()

    def _load_q_table(self) -> Any:
        """Load Q-table from disk."""
        if self.Q_TABLE_PATH.exists():
            try:
                with open(self.Q_TABLE_PATH) as f:
                    data = json.load(f)
                    self.Q = data.get("Q", {})
                    self.alpha = data.get("alpha", 0.1)
                    self.epsilon = data.get("epsilon", 0.15)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                # Q-table file not found or invalid, start fresh
                self.Q = {}

    def _save_q_table(self) -> Any:
        """Save Q-table to disk."""
        self.Q_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.Q_TABLE_PATH, "w") as f:
            json.dump(
                {
                    "Q": self.Q,
                    "alpha": self.alpha,
                    "epsilon": self.epsilon,
                },
                f,
                indent=2,
            )

    def get_state(self, stock_profile: Dict) -> str:
        """
        Convert stock profile to Q-learning state.

        State representation:
        - sector: banking, it, pharma, etc.
        - volatility: high/medium/low
        - predictability: high/medium/low
        - best_horizon: short/medium/long (based on which targets work)
        """
        sector = stock_profile.get("sector", "other")
        predictability = stock_profile.get("predictability", "medium")

        # Infer volatility from AUC variance
        auc_std = stock_profile.get("auc_std", 0.05)
        if auc_std > 0.08:
            volatility = "high"
        elif auc_std > 0.04:
            volatility = "medium"
        else:
            volatility = "low"

        # Infer best horizon from target sensitivity
        target_sens = stock_profile.get("target_sensitivity", {})
        if target_sens:
            best_target = max(target_sens.items(), key=lambda x: x[1])[0]
            if "1d" in best_target:
                horizon = "short"
            elif "5d" in best_target or "10d" in best_target:
                horizon = "medium"
            else:
                horizon = "long"
        else:
            horizon = "medium"

        return f"{sector}|{volatility}|{predictability}|{horizon}"

    def get_action_key(self, target: str, timeframe: str) -> str:
        """Convert action to string key."""
        return f"{target}|{timeframe}"

    def get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for state-action pair."""
        key = f"{state}||{action}"
        if key in self.Q:
            return self.Q[key]["value"]
        return 0.5  # Optimistic initialization

    def get_best_action(self, state: str, available_actions: List[str]) -> Tuple[str, float]:
        """
        Get best action using epsilon-greedy policy.

        Returns:
            (action, q_value)
        """
        import random

        # Exploration
        if random.random() < self.epsilon:
            action = random.choice(available_actions)
            return action, self.get_q_value(state, action)

        # Exploitation
        best_action = None
        best_value = -float("inf")

        for action in available_actions:
            q = self.get_q_value(state, action)
            if q > best_value:
                best_value = q
                best_action = action

        return best_action or available_actions[0], best_value

    def compute_reward(self, result: Dict) -> float:
        """
        Compute reward from ML/backtest result.

        Reward = base_auc + sharpe_bonus - drawdown_penalty

        For 5-day trading focus:
        - AUC > 0.65 is good
        - Sharpe > 1.5 gets bonus
        - Drawdown > 20% gets penalty
        """
        auc = result.get("auc", 0.5)
        backtest = result.get("backtest", {})

        # Base reward from AUC (normalized 0-1)
        reward = (auc - 0.5) * 2  # 0.5 AUC = 0, 0.75 AUC = 0.5, 1.0 AUC = 1.0

        # Sharpe bonus (if backtesting available)
        if backtest:
            strategy = backtest.get("strategy", {})
            sharpe = strategy.get("sharpe", 0)
            if sharpe > 2.0:
                reward += 0.3
            elif sharpe > 1.5:
                reward += 0.2
            elif sharpe > 1.0:
                reward += 0.1

            # Drawdown penalty
            mdd = abs(strategy.get("max_drawdown", 0))
            if mdd > 30:
                reward -= 0.3
            elif mdd > 20:
                reward -= 0.15

            # Outperformance bonus
            outperform = backtest.get("outperformance", 0)
            if outperform > 10:
                reward += 0.2
            elif outperform > 0:
                reward += 0.1

        return max(0.0, min(1.0, reward))

    def update(self, state: str, action: str, reward: float, next_state: str = None) -> Any:
        """
        Update Q-value using Q-learning update rule.

        Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        """
        key = f"{state}||{action}"
        current_q = self.get_q_value(state, action)

        # For stock ML, we don't have strong state transitions
        # Use simplified update: Q(s,a) = Q(s,a) + α * (r - Q(s,a))
        new_q = current_q + self.alpha * (reward - current_q)

        if key not in self.Q:
            self.Q[key] = {"value": new_q, "count": 1, "last_reward": reward}
        else:
            self.Q[key]["value"] = new_q
            self.Q[key]["count"] = self.Q[key].get("count", 0) + 1
            self.Q[key]["last_reward"] = reward

        # Decay exploration rate
        self.epsilon = max(0.05, self.epsilon * 0.995)

        # Save after each update
        self._save_q_table()

    def get_recommendations(self, state: str, top_k: int = 3) -> List[Dict]:
        """
        Get top-k action recommendations for a state.
        """
        # Define all possible actions
        targets = ["next_5d_up", "next_10d_up", "next_1d_up"]
        timeframes = ["60minute", "day"]

        actions = []
        for target in targets:
            for tf in timeframes:
                action = self.get_action_key(target, tf)
                q = self.get_q_value(state, action)
                count = self.Q.get(f"{state}||{action}", {}).get("count", 0)
                actions.append(
                    {
                        "action": action,
                        "target": target,
                        "timeframe": tf,
                        "q_value": q,
                        "confidence": min(1.0, count / 10),  # More visits = more confident
                    }
                )

        # Sort by Q-value
        actions.sort(key=lambda x: -x["q_value"])
        return actions[:top_k]

    def get_transfer_learning_context(self, state: str) -> str:
        """
        Get learned context to help with new stocks.

        This is the key value of swarm learning - transfer knowledge!
        """
        # Parse state
        parts = state.split("|")
        sector = parts[0] if len(parts) > 0 else "unknown"

        # Find similar states (same sector)
        similar_learnings = []
        for key, data in self.Q.items():
            if f"{sector}|" in key and data.get("count", 0) >= 3:
                state_part, action_part = key.split("||")
                similar_learnings.append(
                    {
                        "state": state_part,
                        "action": action_part,
                        "value": data["value"],
                        "count": data["count"],
                    }
                )

        if not similar_learnings:
            return ""

        # Sort by value and format as context
        similar_learnings.sort(key=lambda x: -x["value"])

        lines = [f"Learned patterns for {sector} sector:"]
        for learn in similar_learnings[:5]:
            action_parts = learn["action"].split("|")
            target, tf = action_parts[0], action_parts[1] if len(action_parts) > 1 else "day"
            lines.append(
                f"  - {target} with {tf}: Q={learn['value']:.2f} (tested {learn['count']}x)"
            )

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get Q-learner statistics."""
        if not self.Q:
            return {"entries": 0, "avg_q": 0, "max_q": 0}

        values = [v["value"] for v in self.Q.values()]
        counts = [v.get("count", 1) for v in self.Q.values()]

        return {
            "entries": len(self.Q),
            "avg_q": sum(values) / len(values),
            "max_q": max(values),
            "total_updates": sum(counts),
            "epsilon": self.epsilon,
        }
