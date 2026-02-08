"""
technical-analysis skill — Multi-timeframe technical analysis for NSE stocks.

Pure computation: data loading, indicator calculation, signal generation.
No agent/LLM dependencies.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEFRAME_DIRS = {
    '15minute': '15minuteData',
    '30minute': '30minuteData',
    '60minute': '60minuteData',
    'day': 'DayData',
    'week': 'WeekData',
}

INDICATOR_CONFIG = {
    'trend': {
        'sma': [20, 50, 200],
        'ema': [9, 21],
        'wma': [20],
        'adx': 14,
        'supertrend': {'length': 10, 'multiplier': 3},
    },
    'momentum': {
        'rsi': 14,
        'stoch': {'k': 14, 'd': 3, 'smooth_k': 3},
        'cci': 20,
        'willr': 14,
        'mfi': 14,
        'roc': 12,
    },
    'volatility': {
        'bbands': {'length': 20, 'std': 2},
        'atr': 14,
        'kc': {'length': 20, 'scalar': 1.5},
    },
    'volume': {
        'obv': True,
        'vwap': True,
        'pvt': True,
        'ad': True,
    },
    'overlap': {
        'ichimoku': {'tenkan': 9, 'kijun': 26, 'senkou': 52},
    },
}

DEFAULT_DATA_PATH = "/var/www/sites/personal/stock_market/common/Data/NSE/"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ohlcv(ticker: str, timeframe: str, data_path: str = DEFAULT_DATA_PATH) -> Optional['pd.DataFrame']:
    """Load OHLCV data from NSE compressed CSV files.

    Returns a DataFrame indexed by date with columns: open, high, low, close, volume.
    Returns ``None`` when no data can be found/loaded.
    """
    import pandas as pd

    timeframe_dir = TIMEFRAME_DIRS.get(timeframe.lower())
    if not timeframe_dir:
        logger.warning(f"Unknown timeframe: {timeframe}")
        return None

    data_dir = Path(data_path) / timeframe_dir
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return None

    # Find matching files for the ticker
    pattern = f"*-{ticker}-*.csv.gz"
    files = list(data_dir.glob(pattern))
    if not files:
        pattern = f"*{ticker}*.csv.gz"
        files = list(data_dir.glob(pattern))
    if not files:
        logger.warning(f"No data files found for {ticker} in {data_dir}")
        return None

    files = sorted(files, key=lambda f: f.name)
    recent_files = files[-3:] if timeframe.lower() == 'day' else files[-1:]

    dfs: List[pd.DataFrame] = []
    for file in recent_files:
        try:
            df = pd.read_csv(file, compression='gzip', on_bad_lines='skip')
            dfs.append(df)
        except Exception as e:
            logger.debug(f"Failed to read {file}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]

    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        logger.warning(f"Missing required columns in data for {ticker}")
        return None

    date_col = df['date'].copy()
    if date_col.dtype == 'object':
        date_col = pd.to_datetime(date_col, errors='coerce', utc=True)
        if date_col.dt.tz is not None:
            date_col = date_col.dt.tz_localize(None)
    else:
        date_col = pd.to_datetime(date_col, errors='coerce')

    df['date'] = date_col
    df = df.dropna(subset=['date'])
    df = df.set_index('date').sort_index()
    df = df[~df.index.duplicated(keep='last')]

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df.tail(500)


# ---------------------------------------------------------------------------
# Indicator calculation
# ---------------------------------------------------------------------------

def _detect_ta_library():
    """Detect which TA library is available."""
    try:
        import pandas_ta  # noqa: F401
        return 'pandas_ta'
    except ImportError:
        pass
    try:
        import ta  # noqa: F401
        return 'ta'
    except ImportError:
        return None


def add_all_indicators(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Add all technical indicators to the DataFrame.

    Tries pandas_ta first, falls back to ta library.
    """
    lib = _detect_ta_library()
    if lib is None:
        logger.warning("No technical analysis library available (pandas_ta or ta)")
        return df
    try:
        if lib == 'pandas_ta':
            return _add_indicators_pandas_ta(df)
        else:
            return _add_indicators_ta(df)
    except Exception as e:
        logger.warning(f"Error adding indicators: {e}")
        return df


def _add_indicators_pandas_ta(df: 'pd.DataFrame') -> 'pd.DataFrame':
    import pandas_ta as ta

    try:
        for period in INDICATOR_CONFIG['trend']['sma']:
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)
        for period in INDICATOR_CONFIG['trend']['ema']:
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)

        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None:
            df = df.join(adx_df)

        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd_df is not None:
            df = df.join(macd_df)

        st_cfg = INDICATOR_CONFIG['trend']['supertrend']
        st_df = ta.supertrend(df['high'], df['low'], df['close'],
                              length=st_cfg['length'], multiplier=st_cfg['multiplier'])
        if st_df is not None:
            df = df.join(st_df)

        df['rsi'] = ta.rsi(df['close'], length=14)
        stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        if stoch_df is not None:
            df = df.join(stoch_df)
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        df['roc'] = ta.roc(df['close'], length=12)

        bb_df = ta.bbands(df['close'], length=20, std=2)
        if bb_df is not None:
            df = df.join(bb_df)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        kc_df = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
        if kc_df is not None:
            df = df.join(kc_df)

        df['obv'] = ta.obv(df['close'], df['volume'])
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['pvt'] = ta.pvt(df['close'], df['volume'])
        df['ad'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])

        ichi_df = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52)
        if ichi_df is not None and len(ichi_df) == 2:
            df = df.join(ichi_df[0])

        pivot_df = ta.pivot(df['high'], df['low'], df['close'])
        if pivot_df is not None:
            pivot_df.columns = [c.lower().replace('_', '') for c in pivot_df.columns]
            for col in pivot_df.columns:
                df[col] = pivot_df[col]

    except Exception as e:
        logger.warning(f"Error adding pandas_ta indicators: {e}")
    return df


def _add_indicators_ta(df: 'pd.DataFrame') -> 'pd.DataFrame':
    import ta
    import numpy as np

    try:
        for period in INDICATOR_CONFIG['trend']['sma']:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
        for period in INDICATOR_CONFIG['trend']['ema']:
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)

        adx_ind = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['ADX_14'] = adx_ind.adx()
        df['DMP_14'] = adx_ind.adx_pos()
        df['DMN_14'] = adx_ind.adx_neg()

        macd_ind = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD_12_26_9'] = macd_ind.macd()
        df['MACDs_12_26_9'] = macd_ind.macd_signal()
        df['MACDh_12_26_9'] = macd_ind.macd_diff()

        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['STOCHk_14_3_3'] = stoch.stoch()
        df['STOCHd_14_3_3'] = stoch.stoch_signal()

        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        df['willr'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        df['roc'] = ta.momentum.roc(df['close'], window=12)

        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BBL_20_2.0'] = bb.bollinger_lband()
        df['BBM_20_2.0'] = bb.bollinger_mavg()
        df['BBU_20_2.0'] = bb.bollinger_hband()

        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20)
        df['KCL_20'] = kc.keltner_channel_lband()
        df['KCM_20'] = kc.keltner_channel_mband()
        df['KCU_20'] = kc.keltner_channel_hband()

        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
        df['ad'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])

        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
        df['ITS_9'] = ichimoku.ichimoku_conversion_line()
        df['IKS_26'] = ichimoku.ichimoku_base_line()
        df['ISA_9'] = ichimoku.ichimoku_a()
        df['ISB_26'] = ichimoku.ichimoku_b()

        pp = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['pivot'] = pp
        df['s1'] = 2 * pp - df['high'].shift(1)
        df['s2'] = pp - (df['high'].shift(1) - df['low'].shift(1))
        df['r1'] = 2 * pp - df['low'].shift(1)
        df['r2'] = pp + (df['high'].shift(1) - df['low'].shift(1))

        atr_vals = df['atr']
        hl2 = (df['high'] + df['low']) / 2
        multiplier = 3
        upper_band = hl2 + (multiplier * atr_vals)
        lower_band = hl2 - (multiplier * atr_vals)

        supertrend = np.zeros(len(df))
        supertrend[0] = upper_band.iloc[0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > supertrend[i - 1]:
                supertrend[i] = (
                    max(lower_band.iloc[i], supertrend[i - 1])
                    if supertrend[i - 1] == lower_band.iloc[i - 1] or supertrend[i - 1] < lower_band.iloc[i]
                    else lower_band.iloc[i]
                )
            else:
                supertrend[i] = (
                    min(upper_band.iloc[i], supertrend[i - 1])
                    if supertrend[i - 1] == upper_band.iloc[i - 1] or supertrend[i - 1] > upper_band.iloc[i]
                    else upper_band.iloc[i]
                )
        df['SUPERT_10_3.0'] = supertrend

    except Exception as e:
        logger.warning(f"Error adding ta library indicators: {e}")
    return df


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def generate_signals(df: 'pd.DataFrame', timeframe: str) -> Dict[str, Any]:
    """Generate trading signals from a DataFrame with indicators already added."""
    signals: Dict[str, Any] = {
        'trend_signal': 0,
        'momentum_signal': 0,
        'volatility_signal': 0,
        'buy_signals': [],
        'sell_signals': [],
        'observations': [],
    }
    if df.empty:
        return signals

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    try:
        trend_score = 0.0

        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            if latest['sma_50'] > latest['sma_200']:
                trend_score += 1
                if prev['sma_50'] <= prev['sma_200']:
                    signals['buy_signals'].append('Golden Cross (SMA50 > SMA200)')
            else:
                trend_score -= 1
                if prev['sma_50'] >= prev['sma_200']:
                    signals['sell_signals'].append('Death Cross (SMA50 < SMA200)')

        if 'sma_200' in df.columns:
            trend_score += 0.5 if latest['close'] > latest['sma_200'] else -0.5

        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                trend_score += 0.5
                if prev['MACD_12_26_9'] <= prev['MACDs_12_26_9']:
                    signals['buy_signals'].append('MACD Bullish Crossover')
            else:
                trend_score -= 0.5
                if prev['MACD_12_26_9'] >= prev['MACDs_12_26_9']:
                    signals['sell_signals'].append('MACD Bearish Crossover')

        if 'ADX_14' in df.columns:
            adx = latest['ADX_14']
            if adx > 25:
                signals['observations'].append(f'Strong trend (ADX={adx:.1f})')
            elif adx < 20:
                signals['observations'].append(f'Weak trend (ADX={adx:.1f})')

        supert_col = [c for c in df.columns if c.startswith('SUPERT_')]
        if supert_col:
            trend_score += 0.5 if latest['close'] > latest[supert_col[0]] else -0.5

        signals['trend_signal'] = max(-1, min(1, trend_score / 3))

        momentum_score = 0.0

        if 'rsi' in df.columns:
            rsi = latest['rsi']
            if rsi > 70:
                momentum_score -= 1
                signals['sell_signals'].append(f'RSI Overbought ({rsi:.1f})')
            elif rsi < 30:
                momentum_score += 1
                signals['buy_signals'].append(f'RSI Oversold ({rsi:.1f})')
            elif rsi > 50:
                momentum_score += 0.3
            else:
                momentum_score -= 0.3

        if 'mfi' in df.columns:
            mfi = latest['mfi']
            if mfi > 80:
                momentum_score -= 0.5
            elif mfi < 20:
                momentum_score += 0.5

        if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
            stoch_k = latest['STOCHk_14_3_3']
            stoch_d = latest['STOCHd_14_3_3']
            if stoch_k < 20 and stoch_k > stoch_d:
                momentum_score += 0.5
                signals['buy_signals'].append('Stochastic Bullish (oversold)')
            elif stoch_k > 80 and stoch_k < stoch_d:
                momentum_score -= 0.5
                signals['sell_signals'].append('Stochastic Bearish (overbought)')

        signals['momentum_signal'] = max(-1, min(1, momentum_score / 2))

        if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            bb_lower = latest['BBL_20_2.0']
            bb_upper = latest['BBU_20_2.0']
            if latest['close'] <= bb_lower:
                signals['volatility_signal'] = 1
                signals['observations'].append('Price at lower Bollinger Band')
            elif latest['close'] >= bb_upper:
                signals['volatility_signal'] = -1
                signals['observations'].append('Price at upper Bollinger Band')

    except Exception as e:
        logger.debug(f"Signal generation error: {e}")

    return signals


# ---------------------------------------------------------------------------
# Latest indicator extraction
# ---------------------------------------------------------------------------


def get_latest_indicators(df: 'pd.DataFrame') -> Dict[str, Any]:
    """Extract a dict of the most recent indicator values."""
    import pandas as pd

    if df.empty:
        return {}

    latest = df.iloc[-1]
    indicators: Dict[str, Any] = {}

    base_cols = [
        'rsi', 'mfi', 'cci', 'willr', 'roc', 'atr', 'obv', 'vwap',
        'sma_20', 'sma_50', 'sma_200', 'ema_9', 'ema_21',
    ]
    for col in base_cols:
        if col in df.columns and not pd.isna(latest[col]):
            indicators[col] = round(float(latest[col]), 2)

    if 'MACD_12_26_9' in df.columns:
        indicators['macd'] = round(float(latest['MACD_12_26_9']), 2) if not pd.isna(latest['MACD_12_26_9']) else None
        indicators['macd_signal'] = round(float(latest.get('MACDs_12_26_9', 0)), 2) if not pd.isna(latest.get('MACDs_12_26_9')) else None

    if 'ADX_14' in df.columns:
        indicators['adx'] = round(float(latest['ADX_14']), 2) if not pd.isna(latest['ADX_14']) else None

    if 'BBL_20_2.0' in df.columns:
        indicators['bb_lower'] = round(float(latest['BBL_20_2.0']), 2) if not pd.isna(latest['BBL_20_2.0']) else None
        indicators['bb_upper'] = round(float(latest['BBU_20_2.0']), 2) if not pd.isna(latest['BBU_20_2.0']) else None

    if 'STOCHk_14_3_3' in df.columns:
        indicators['stoch_k'] = round(float(latest['STOCHk_14_3_3']), 2) if not pd.isna(latest['STOCHk_14_3_3']) else None
        indicators['stoch_d'] = round(float(latest.get('STOCHd_14_3_3', 0)), 2) if not pd.isna(latest.get('STOCHd_14_3_3')) else None

    return indicators


# ---------------------------------------------------------------------------
# Main tool function (skill entry point)
# ---------------------------------------------------------------------------


def technical_analysis_tool(params: dict) -> dict:
    """Run multi-timeframe technical analysis on a stock.

    Parameters (via ``params`` dict):
        ticker (str): NSE stock symbol, e.g. 'RELIANCE'.
        timeframes (list[str], optional): Default ['60minute', 'Day'].
        data_path (str, optional): Override data directory.
        format (str, optional): 'json' (default) or 'summary'.

    Returns:
        dict with ``success``, ``data`` (timeframes, signals, support/resistance, trend, indicators).
    """
    import pandas as pd

    ticker = params.get('ticker') or params.get('company_name', '')
    timeframes = params.get('timeframes', ['60minute', 'Day'])
    data_path = params.get('data_path', DEFAULT_DATA_PATH)

    if not ticker:
        return {'success': False, 'error': 'ticker is required'}

    result: Dict[str, Any] = {
        'ticker': ticker,
        'timeframes': {},
        'signals': {},
        'support_levels': [],
        'resistance_levels': [],
        'trend': 'NEUTRAL',
    }

    ta_lib = _detect_ta_library()
    if ta_lib is None:
        return {'success': False, 'error': 'No TA library available (install pandas_ta or ta)'}

    all_signals: List[float] = []

    for tf in timeframes:
        try:
            df = load_ohlcv(ticker, tf.lower(), data_path)
            if df is None or df.empty or len(df) < 60:
                logger.warning(f"Insufficient data for {ticker} @ {tf}")
                continue

            df = add_all_indicators(df)
            signals = generate_signals(df, tf)

            result['timeframes'][tf] = {
                'last_price': float(df['close'].iloc[-1]) if 'close' in df.columns else 0,
                'indicators': get_latest_indicators(df),
                'signals': signals,
            }

            if signals.get('trend_signal'):
                all_signals.append(signals['trend_signal'])

            # Support / resistance from pivot points
            if 'pivot' in df.columns:
                for key, target in [('s1', 'support_levels'), ('s2', 'support_levels'),
                                    ('r1', 'resistance_levels'), ('r2', 'resistance_levels')]:
                    if key in df.columns:
                        val = df[key].iloc[-1]
                        if val is not None and not pd.isna(val):
                            result[target].append(float(val))

        except Exception as e:
            logger.warning(f"Technical analysis error for {ticker} @ {tf}: {e}")
            continue

    if all_signals:
        bullish = sum(1 for s in all_signals if s > 0)
        bearish = sum(1 for s in all_signals if s < 0)
        if bullish > bearish:
            result['trend'] = 'BULLISH'
        elif bearish > bullish:
            result['trend'] = 'BEARISH'

    result['support_levels'] = sorted(set(result['support_levels']), reverse=True)[:5]
    result['resistance_levels'] = sorted(set(result['resistance_levels']))[:5]

    return {'success': bool(result['timeframes']), 'data': result}
