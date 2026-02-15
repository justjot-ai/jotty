"""
Model Chat Agent
================

Natural language interface to interact with ML models and experiments.
Enables users to query, compare, analyze, and get improvement suggestions
for their ML models through conversation.

Usage:
    agent = ModelChatAgent()
    result = await agent.chat("What's my best model for RELIANCE?")
    print(result['response'])
"""

from typing import Dict, Any, List, Optional
import json
import logging
import re

logger = logging.getLogger(__name__)


class ModelChatAgent:
    """
    Chat interface for ML models - query, compare, predict, improve.

    Example queries:
    - "What's my best performing model for RELIANCE?"
    - "Compare accuracy across all banking stocks"
    - "Why is SBIN underperforming?"
    - "Show me the top features for HDFCBANK"
    - "Suggest improvements for my INFY model"
    - "Run a prediction for RELIANCE using the best model"
    """

    SYSTEM_PROMPT = '''You are an ML model assistant with access to MLflow experiments.
    You can:
    1. Query model performance metrics (AUC, accuracy, Sharpe, ROMAD)
    2. Compare models across stocks, timeframes, targets
    3. Analyze feature importance patterns
    4. Suggest model improvements based on patterns
    5. Load and run predictions with trained models

    Available tools:
    - list_runs: Get recent experiment runs
    - get_best_run: Find best model by metric
    - compare_runs: Compare multiple runs
    - get_feature_importance: Get top features for a model
    - load_model: Load a model for predictions
    - suggest_improvements: Analyze and suggest improvements

    Always be specific with metrics and back claims with data.'''

    # Intent patterns for classification
    INTENT_PATTERNS = {
        'query_best': [
            r'best\s+(model|run|result)',
            r'top\s+(model|performer)',
            r'highest\s+(auc|accuracy|score)',
            r'what.*best',
        ],
        'compare': [
            r'compare',
            r'vs\.?',
            r'versus',
            r'difference\s+between',
            r'how\s+does.*compare',
        ],
        'feature_analysis': [
            r'feature',
            r'importance',
            r'top\s+features',
            r'what\s+features',
            r'which\s+features',
        ],
        'suggest_improvement': [
            r'suggest',
            r'improve',
            r'why.*underperform',
            r'how\s+to\s+make.*better',
            r'recommendations?',
        ],
        'predict': [
            r'predict',
            r'forecast',
            r'run\s+(a\s+)?prediction',
            r'what\s+will',
        ],
        'list_runs': [
            r'list\s+(all\s+)?runs',
            r'show\s+(all\s+)?runs',
            r'recent\s+runs',
            r'all\s+experiments',
        ],
    }

    # Stock symbol patterns
    STOCK_PATTERN = r'\b([A-Z]{2,15})\b'
    COMMON_STOCKS = {
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
        'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'AXISBANK',
        'ASIANPAINT', 'MARUTI', 'TITAN', 'SUNPHARMA', 'BAJFINANCE',
        'WIPRO', 'HCLTECH', 'TECHM', 'TATAMOTORS', 'TATASTEEL', 'ONGC',
        'NTPC', 'POWERGRID', 'COALINDIA', 'ADANIENT', 'ADANIPORTS',
    }

    # Metric keywords
    METRICS = {
        'auc': ['auc', 'roc', 'roc-auc', 'roc_auc'],
        'accuracy': ['accuracy', 'acc'],
        'f1': ['f1', 'f1-score', 'f1_score'],
        'sharpe': ['sharpe', 'sharpe_ratio', 'sharpe ratio'],
        'romad': ['romad', 'return over max drawdown'],
        'annual_return': ['annual', 'yearly', 'annual_return', 'annual return'],
        'max_drawdown': ['drawdown', 'max_drawdown', 'mdd'],
    }

    def __init__(self, llm_model: str = '') -> None:
        """
        Initialize ModelChatAgent.

        Args:
            llm_model: Model to use for LLM reasoning (sonnet, opus, haiku)
        """
        from Jotty.core.infrastructure.foundation.config_defaults import DEFAULT_MODEL_ALIAS
        self._tracker = None
        self._llm = None
        self._llm_model = llm_model or DEFAULT_MODEL_ALIAS
        self.conversation_history: List[Dict[str, str]] = []
        self._initialized = False

    async def _ensure_initialized(self) -> Any:
        """Lazy initialization of dependencies."""
        if self._initialized:
            return

        # Initialize MLflow tracker
        try:
            from core.skills.ml import MLflowTrackerSkill
            self._tracker = MLflowTrackerSkill()
            await self._tracker.init()
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow tracker: {e}")
            self._tracker = None

        # Initialize LLM
        try:
            from core.integration.direct_claude_cli_lm import DirectClaudeCLI
            self._llm = DirectClaudeCLI(model=self._llm_model)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            self._llm = None

        self._initialized = True

    async def chat(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """
        Handle natural language query about models.

        Args:
            query: User's natural language query
            context: Optional additional context

        Returns:
            Dict with 'response' and optionally 'data'
        """
        await self._ensure_initialized()

        # Add to history
        self.conversation_history.append({"role": "user", "content": query})

        # Detect intent
        intent = self._detect_intent(query)
        logger.info(f"Detected intent: {intent}")

        # Execute based on intent
        try:
            if intent['type'] == 'query_best':
                result = await self._handle_best_model_query(intent)
            elif intent['type'] == 'compare':
                result = await self._handle_comparison(intent)
            elif intent['type'] == 'feature_analysis':
                result = await self._handle_feature_analysis(intent)
            elif intent['type'] == 'suggest_improvement':
                result = await self._handle_improvement_suggestion(intent)
            elif intent['type'] == 'predict':
                result = await self._handle_prediction(intent)
            elif intent['type'] == 'list_runs':
                result = await self._handle_list_runs(intent)
            else:
                result = await self._handle_general_query(query)
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            result = {"response": f"Error processing query: {str(e)}", "error": str(e)}

        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": result['response']})

        return result

    def _detect_intent(self, query: str) -> Dict:
        """
        Detect query intent using pattern matching.

        Args:
            query: User query string

        Returns:
            Dict with type, symbols, metric, etc.
        """
        query_lower = query.lower()

        # Detect intent type
        intent_type = 'general'
        for itype, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent_type = itype
                    break
            if intent_type != 'general':
                break

        # Extract symbols
        symbols = []
        for match in re.finditer(self.STOCK_PATTERN, query):
            symbol = match.group(1)
            if symbol in self.COMMON_STOCKS or len(symbol) >= 3:
                symbols.append(symbol)

        # Remove common false positives
        false_positives = {'AUC', 'ROC', 'THE', 'AND', 'FOR', 'WITH', 'FROM', 'BEST', 'MODEL'}
        symbols = [s for s in symbols if s not in false_positives]

        # Detect metric of interest
        metric = 'test_auc'  # default
        for metric_name, keywords in self.METRICS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    metric = metric_name
                    break

        # Detect timeframe
        timeframe = None
        if '60min' in query_lower or 'hourly' in query_lower:
            timeframe = '60minute'
        elif '15min' in query_lower:
            timeframe = '15minute'
        elif 'daily' in query_lower or 'day' in query_lower:
            timeframe = 'day'

        return {
            'type': intent_type,
            'symbols': symbols,
            'metric': metric,
            'timeframe': timeframe,
            'query': query,
        }

    async def _handle_best_model_query(self, intent: Dict) -> Dict:
        """Find best model for given criteria."""
        if not self._tracker:
            return {"response": "MLflow tracker not available. Run some /stock-ml experiments first."}

        symbol = intent.get('symbols', [''])[0] if intent.get('symbols') else ''
        metric = intent.get('metric', 'test_auc')

        # Query MLflow
        experiment_name = f"stock_ml_{symbol}" if symbol else None
        runs = await self._tracker.list_runs(
            experiment_name=experiment_name,
            max_results=10
        )

        if not runs:
            return {"response": f"No models found for {symbol or 'any stock'}. Run /stock-ml first."}

        # Find best run by metric
        best = None
        best_value = -float('inf')
        metric_key = f"metrics.{metric}" if not metric.startswith('metrics.') else metric

        for run in runs:
            value = run.get(metric_key, run.get(metric, 0))
            if value and value > best_value:
                best_value = value
                best = run

        if not best:
            best = runs[0]

        # Format response
        response = f"**Best model for {symbol or 'all stocks'}:**\n\n"
        response += f"**Run:** {best.get('tags.mlflow.runName', best.get('run_id', 'N/A')[:8])}\n"
        response += f"**{metric}:** {best_value:.4f}\n"

        # Add additional metrics if available
        params = {k.replace('params.', ''): v for k, v in best.items() if k.startswith('params.')}
        metrics = {k.replace('metrics.', ''): v for k, v in best.items() if k.startswith('metrics.')}

        if params.get('symbol'):
            response += f"**Symbol:** {params.get('symbol')}\n"
        if params.get('target_type'):
            response += f"**Target:** {params.get('target_type')}\n"
        if params.get('timeframe'):
            response += f"**Timeframe:** {params.get('timeframe')}\n"

        if metrics:
            response += "\n**Performance Metrics:**\n"
            for k, v in sorted(metrics.items())[:10]:
                if isinstance(v, (int, float)):
                    response += f"- {k}: {v:.4f}\n"

        return {"response": response, "data": best}

    async def _handle_comparison(self, intent: Dict) -> Dict:
        """Compare models across runs."""
        if not self._tracker:
            return {"response": "MLflow tracker not available."}

        symbols = intent.get('symbols', [])
        metric = intent.get('metric', 'test_auc')

        if len(symbols) < 2:
            return {"response": "Please specify at least 2 symbols to compare (e.g., 'Compare RELIANCE vs TCS')"}

        comparison_data = []
        for symbol in symbols:
            runs = await self._tracker.list_runs(
                experiment_name=f"stock_ml_{symbol}",
                max_results=1
            )
            if runs:
                run = runs[0]
                metrics = {k.replace('metrics.', ''): v for k, v in run.items() if k.startswith('metrics.')}
                comparison_data.append({
                    'symbol': symbol,
                    metric: metrics.get(metric, 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'backtest_annual_return': metrics.get('backtest_annual_return', 0),
                    'backtest_sharpe': metrics.get('backtest_sharpe', 0),
                })

        if not comparison_data:
            return {"response": "No data found for the specified symbols."}

        # Format as table
        table = self._format_comparison_table(comparison_data, metric)
        return {"response": table, "data": comparison_data}

    async def _handle_feature_analysis(self, intent: Dict) -> Dict:
        """Analyze feature importance for a model."""
        if not self._tracker:
            return {"response": "MLflow tracker not available."}

        symbol = intent.get('symbols', [''])[0] if intent.get('symbols') else ''

        runs = await self._tracker.list_runs(
            experiment_name=f"stock_ml_{symbol}" if symbol else None,
            max_results=1
        )

        if not runs:
            return {"response": f"No models found for {symbol or 'any stock'}."}

        run = runs[0]
        run_id = run.get('run_id')

        # Try to get feature importance from metrics
        feature_metrics = {
            k.replace('metrics.', ''): v
            for k, v in run.items()
            if k.startswith('metrics.feature_importance')
        }

        if not feature_metrics:
            return {"response": f"No feature importance data found for {symbol}. Run /stock-ml with --mlflow flag."}

        # Sort and format
        sorted_features = sorted(feature_metrics.items(), key=lambda x: -abs(x[1]))[:15]

        response = f"**Top Features for {symbol or 'latest model'}:**\n\n"
        response += "| Rank | Feature | Importance |\n"
        response += "|------|---------|------------|\n"
        for i, (name, value) in enumerate(sorted_features, 1):
            # Clean up feature name
            clean_name = name.replace('feature_importance_', '').split('_', 1)[-1] if '_' in name else name
            response += f"| {i} | {clean_name} | {value:.4f} |\n"

        return {"response": response, "data": dict(sorted_features)}

    async def _handle_improvement_suggestion(self, intent: Dict) -> Dict:
        """Analyze model and suggest improvements using LLM."""
        symbol = intent.get('symbols', [''])[0] if intent.get('symbols') else ''

        if not self._tracker:
            return {"response": "MLflow tracker not available."}

        runs = await self._tracker.list_runs(
            experiment_name=f"stock_ml_{symbol}" if symbol else None,
            max_results=5
        )

        if not runs:
            return {"response": f"No models found for {symbol}. Run /stock-ml first."}

        # Gather data for analysis
        metrics_list = []
        for run in runs:
            metrics = {k.replace('metrics.', ''): v for k, v in run.items() if k.startswith('metrics.')}
            params = {k.replace('params.', ''): v for k, v in run.items() if k.startswith('params.')}
            metrics_list.append({**metrics, **params})

        # Calculate aggregates
        best_auc = max(m.get('test_auc', m.get('auc', 0)) for m in metrics_list if m.get('test_auc', m.get('auc')))
        best_sharpe = max(m.get('backtest_sharpe', 0) for m in metrics_list if m.get('backtest_sharpe'))

        # Get feature importance from best run
        best_run = runs[0]
        feature_metrics = {
            k.replace('metrics.', ''): v
            for k, v in best_run.items()
            if k.startswith('metrics.feature_importance')
        }
        top_features = sorted(feature_metrics.items(), key=lambda x: -abs(x[1]))[:10]

        # Use LLM if available for smart suggestions
        if self._llm:
            prompt = f'''Analyze this ML model performance and suggest 3 specific improvements:

Symbol: {symbol or 'Unknown'}
Best AUC: {best_auc:.4f}
Best Sharpe: {best_sharpe:.2f}
Top Features: {[f[0].split('_')[-1] for f in top_features[:5]]}

Recent configurations tried:
{json.dumps(metrics_list[:3], indent=2, default=str)[:1500]}

Provide specific, actionable suggestions in these categories:
1. Feature engineering (new features to add based on what's working)
2. Model hyperparameters (what to tune given current performance)
3. Data/target changes (timeframe, target horizon adjustments)

Be concise and specific.'''

            try:
                suggestions = self._llm(prompt)[0]
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
                suggestions = self._generate_rule_based_suggestions(best_auc, best_sharpe, top_features)
        else:
            suggestions = self._generate_rule_based_suggestions(best_auc, best_sharpe, top_features)

        return {
            "response": f"## Improvement Suggestions for {symbol or 'your model'}\n\n{suggestions}",
            "suggestions": suggestions,
            "context": {
                'symbol': symbol,
                'best_auc': best_auc,
                'best_sharpe': best_sharpe,
            }
        }

    def _generate_rule_based_suggestions(self, auc: float, sharpe: float, top_features: List) -> str:
        """Generate rule-based improvement suggestions."""
        suggestions = []

        # Based on AUC
        if auc < 0.55:
            suggestions.append("**Feature Engineering:** Consider adding momentum divergence features (RSI vs price), volume-price relationship features, and sector-relative performance metrics.")
        elif auc < 0.60:
            suggestions.append("**Feature Engineering:** Try adding lagged features (5d, 10d returns), cross-sectional rank features, and volatility regime indicators.")
        else:
            suggestions.append("**Feature Engineering:** Model is performing well. Consider adding market regime features or alternative data sources.")

        # Based on Sharpe
        if sharpe < 0.5:
            suggestions.append("**Model Tuning:** Low Sharpe suggests high variance. Try increasing regularization, reducing tree depth, or using ensemble methods.")
        elif sharpe < 1.0:
            suggestions.append("**Model Tuning:** Consider optimizing for Sharpe directly using custom loss functions or threshold tuning.")
        else:
            suggestions.append("**Model Tuning:** Good risk-adjusted returns. Consider position sizing optimization.")

        # General
        suggestions.append("**Data/Target:** Try longer prediction horizons (10d, 20d) which often have better signal, or use 60-minute data for more samples.")

        return "\n\n".join(suggestions)

    async def _handle_prediction(self, intent: Dict) -> Dict:
        """Load model and prepare for prediction."""
        if not self._tracker:
            return {"response": "MLflow tracker not available."}

        symbol = intent.get('symbols', [''])[0] if intent.get('symbols') else ''

        best = await self._tracker.get_best_run(
            experiment_name=f"stock_ml_{symbol}" if symbol else None,
            metric="test_auc"
        )

        if not best:
            return {"response": f"No model found for {symbol}. Run /stock-ml first."}

        run_id = best.get('run_id')

        # Try to load model
        try:
            model = await self._tracker.load_model(run_id=run_id)
            if model:
                return {
                    "response": f"Model loaded successfully for {symbol}.\n\nRun `/stock-ml {symbol} --predict` to generate predictions with latest data.",
                    "model_loaded": True,
                    "run_id": run_id
                }
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")

        return {
            "response": f"Model reference found for {symbol} (run: {run_id[:8]}). Use `/stock-ml {symbol} --predict` to run predictions.",
            "model_loaded": False,
            "run_id": run_id
        }

    async def _handle_list_runs(self, intent: Dict) -> Dict:
        """List recent experiment runs."""
        if not self._tracker:
            return {"response": "MLflow tracker not available."}

        symbol = intent.get('symbols', [''])[0] if intent.get('symbols') else ''

        runs = await self._tracker.list_runs(
            experiment_name=f"stock_ml_{symbol}" if symbol else None,
            max_results=20
        )

        if not runs:
            return {"response": "No runs found. Run /stock-ml to create experiments."}

        response = f"**Recent Runs{' for ' + symbol if symbol else ''}:**\n\n"
        response += "| Run | Symbol | Target | AUC | Sharpe |\n"
        response += "|-----|--------|--------|-----|--------|\n"

        for run in runs[:15]:
            params = {k.replace('params.', ''): v for k, v in run.items() if k.startswith('params.')}
            metrics = {k.replace('metrics.', ''): v for k, v in run.items() if k.startswith('metrics.')}

            run_name = run.get('tags.mlflow.runName', run.get('run_id', 'N/A')[:8])
            sym = params.get('symbol', 'N/A')
            target = params.get('target_type', 'N/A')[:12]
            auc = metrics.get('test_auc', metrics.get('auc', 0))
            sharpe = metrics.get('backtest_sharpe', 0)

            auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) else 'N/A'
            sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else 'N/A'

            response += f"| {run_name[:8]} | {sym} | {target} | {auc_str} | {sharpe_str} |\n"

        return {"response": response, "data": runs}

    async def _handle_general_query(self, query: str) -> Dict:
        """Handle general queries using LLM."""
        if self._llm:
            context = f"""You are an ML model assistant. The user is asking about their stock prediction models.

Available commands they can use:
- /stock-ml SYMBOL --backtest - Train and backtest a model
- /model-chat - Chat interface (current)
- /mlflow list - List MLflow experiments

User query: {query}

Provide a helpful response about ML models for stock prediction."""

            try:
                response = self._llm(context)[0]
                return {"response": response}
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")

        return {
            "response": "I can help you with:\n"
                       "- Finding your best models: 'What's my best model for RELIANCE?'\n"
                       "- Comparing stocks: 'Compare HDFCBANK vs ICICIBANK'\n"
                       "- Feature analysis: 'Show top features for TCS'\n"
                       "- Improvement suggestions: 'Suggest improvements for INFY'\n"
                       "- Listing runs: 'List all runs'\n\n"
                       "Try one of these queries!"
        }

    def _format_comparison_table(self, data: List[Dict], primary_metric: str) -> str:
        """Format comparison data as markdown table."""
        if not data:
            return "No data to compare"

        headers = ['Symbol', primary_metric, 'Accuracy', 'Annual Return', 'Sharpe']
        rows = []
        for d in sorted(data, key=lambda x: -(x.get(primary_metric) or 0)):
            rows.append([
                d['symbol'],
                f"{d.get(primary_metric, 0):.4f}",
                f"{d.get('accuracy', 0):.4f}",
                f"{d.get('backtest_annual_return', 0):.1f}%",
                f"{d.get('backtest_sharpe', 0):.2f}"
            ])

        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            table += "| " + " | ".join(row) + " |\n"

        return table

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
