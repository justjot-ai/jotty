from typing import List, Optional, Tuple

class StockAnalyzer:
    """A class for analyzing stock price data and generating trading signals."""

    def calculate_sma(self, prices: List[float], window: int) -> List[float]:
        """
        Calculate Simple Moving Average for a given list of prices.

        Args:
            prices: List of stock prices
            window: Window size for the moving average

        Returns:
            List of SMA values (shorter than input by window-1)
        """
        if len(prices) < window:
            return []
        
        sma_values = []
        for i in range(len(prices) - window + 1):
            window_avg = sum(prices[i:i + window]) / window
            sma_values.append(window_avg)
        
        return sma_values

    def calculate_ema(self, prices: List[float], window: int, smoothing: float = 2.0) -> List[float]:
        """
        Calculate Exponential Moving Average for a given list of prices.

        Args:
            prices: List of stock prices
            window: Window size for the moving average
            smoothing: Smoothing factor (default is 2.0)

        Returns:
            List of EMA values
        """
        if len(prices) < window:
            return []
        
        ema_values = []
        multiplier = smoothing / (window + 1)
        
        # First EMA is SMA
        initial_sma = sum(prices[:window]) / window
        ema_values.append(initial_sma)
        
        # Calculate subsequent EMAs
        for i in range(window, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values

    def detect_crossover(
        self,
        prices: List[float],
        short_window: int,
        long_window: int,
        ma_type: str = "sma"
    ) -> List[Tuple[int, str]]:
        """
        Detect crossover signals when short MA crosses long MA.

        Args:
            prices: List of stock prices
            short_window: Window size for short-term moving average
            long_window: Window size for long-term moving average
            ma_type: Type of moving average ('sma' or 'ema')

        Returns:
            List of tuples containing (index, signal_type) where signal_type is 'bullish' or 'bearish'
        """
        if len(prices) < long_window:
            return []
        
        if ma_type.lower() == "sma":
            short_ma = self.calculate_sma(prices, short_window)
            long_ma = self.calculate_sma(prices, long_window)
        elif ma_type.lower() == "ema":
            short_ma = self.calculate_ema(prices, short_window)
            long_ma = self.calculate_ema(prices, long_window)
        else:
            raise ValueError("ma_type must be 'sma' or 'ema'")
        
        # Align the two MA series
        offset = len(short_ma) - len(long_ma)
        if offset > 0:
            short_ma = short_ma[offset:]
        elif offset < 0:
            long_ma = long_ma[-offset:]
        
        crossovers = []
        for i in range(1, len(short_ma)):
            # Bullish crossover: short MA crosses above long MA
            if short_ma[i - 1] <= long_ma[i - 1] and short_ma[i] > long_ma[i]:
                crossovers.append((i, "bullish"))
            # Bearish crossover: short MA crosses below long MA
            elif short_ma[i - 1] >= long_ma[i - 1] and short_ma[i] < long_ma[i]:
                crossovers.append((i, "bearish"))
        
        return crossovers