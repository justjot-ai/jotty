"""Research Swarm - DSPy Signatures."""

import dspy

class StockAnalysisSignature(dspy.Signature):
    """Analyze stock data and provide investment recommendation.

    You are a senior equity research analyst at a top investment bank.
    Analyze the provided financial data and news to generate a professional rating.
    """
    ticker: str = dspy.InputField(desc="Stock ticker symbol")
    company_name: str = dspy.InputField(desc="Company name")
    financial_data: str = dspy.InputField(desc="JSON financial metrics")
    news_summary: str = dspy.InputField(desc="Recent news and developments")

    rating: str = dspy.OutputField(desc="Rating: STRONG BUY, BUY, HOLD, SELL, or STRONG SELL")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0")
    thesis: str = dspy.OutputField(desc="3-5 bullet points for investment thesis, separated by |")
    risks: str = dspy.OutputField(desc="3-5 key risks, separated by |")
    reasoning: str = dspy.OutputField(desc="2-3 sentence reasoning for the rating")


class SentimentAnalysisSignature(dspy.Signature):
    """Analyze sentiment from news headlines and content.

    You are a sentiment analysis expert. Score the overall sentiment
    from the provided news about a company.
    """
    company: str = dspy.InputField(desc="Company name")
    news_text: str = dspy.InputField(desc="News headlines and snippets")

    sentiment_score: float = dspy.OutputField(desc="Score from -1.0 (very negative) to 1.0 (very positive)")
    sentiment_label: str = dspy.OutputField(desc="VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, or VERY_POSITIVE")
    key_themes: str = dspy.OutputField(desc="Top 3 themes in news, separated by |")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for sentiment score")


class PeerSelectionSignature(dspy.Signature):
    """Select appropriate peer companies for comparison.

    Given a company's sector and industry, identify the most relevant
    publicly traded peer companies for comparison.
    """
    company: str = dspy.InputField(desc="Company name")
    sector: str = dspy.InputField(desc="Company sector")
    industry: str = dspy.InputField(desc="Company industry")
    exchange: str = dspy.InputField(desc="Stock exchange (NSE, NYSE, etc.)")

    peers: str = dspy.OutputField(desc="5 peer ticker symbols separated by comma (e.g., INFY,TCS,WIPRO,HCLTECH,TECHM)")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for peer selection")


class SocialSentimentSignature(dspy.Signature):
    """Analyze social sentiment from multiple sources.

    You are a sentiment analysis expert analyzing news, forums, and analyst discussions
    about a company to provide comprehensive sentiment analysis.
    """
    company: str = dspy.InputField(desc="Company name")
    news_text: str = dspy.InputField(desc="News headlines and snippets")
    forum_text: str = dspy.InputField(desc="Forum discussions and social media posts")

    overall_sentiment: float = dspy.OutputField(desc="Score from -1.0 (very bearish) to 1.0 (very bullish)")
    sentiment_label: str = dspy.OutputField(desc="BEARISH, NEUTRAL, or BULLISH")
    key_themes: str = dspy.OutputField(desc="Top 3-5 themes discussed, separated by |")
    sentiment_drivers: str = dspy.OutputField(desc="Key positive factors | Key negative factors (separated by ||)")


class TechnicalSignalsSignature(dspy.Signature):
    """Analyze technical indicators to generate trading signals.

    You are a technical analysis expert. Analyze the indicator values and generate
    a comprehensive technical outlook.
    """
    ticker: str = dspy.InputField(desc="Stock ticker symbol")
    indicator_summary: str = dspy.InputField(desc="JSON summary of technical indicators")
    price_data: str = dspy.InputField(desc="Recent price levels and changes")

    trend: str = dspy.OutputField(desc="BULLISH, NEUTRAL, or BEARISH")
    signal_strength: float = dspy.OutputField(desc="Signal strength 0.0-1.0")
    support_levels: str = dspy.OutputField(desc="Key support levels separated by comma")
    resistance_levels: str = dspy.OutputField(desc="Key resistance levels separated by comma")
    key_observations: str = dspy.OutputField(desc="Top 3 technical observations, separated by |")


# =============================================================================
# RESEARCH SWARM
# =============================================================================

