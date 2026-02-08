-- Stock master data table
-- Stores fundamental information about publicly traded stocks
CREATE TABLE stock (
    symbol VARCHAR(20) PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap_usd DECIMAL(20, 2) CHECK (market_cap_usd >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Historical price data table
-- Stores daily OHLCV data for each stock
CREATE TABLE stock_price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    open_price DECIMAL(12, 4) NOT NULL CHECK (open_price >= 0),
    high_price DECIMAL(12, 4) NOT NULL CHECK (high_price >= 0),
    low_price DECIMAL(12, 4) NOT NULL CHECK (low_price >= 0),
    close_price DECIMAL(12, 4) NOT NULL CHECK (close_price >= 0),
    trading_volume INTEGER NOT NULL CHECK (trading_volume >= 0),
    adjusted_close_price DECIMAL(12, 4) NOT NULL CHECK (adjusted_close_price >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol) REFERENCES stock(symbol) ON DELETE CASCADE,
    UNIQUE(symbol, trade_date),
    -- Business rule: high >= low, and both should be between open/close
    CHECK (high_price >= low_price),
    CHECK (high_price >= open_price AND high_price >= close_price),
    CHECK (low_price <= open_price AND low_price <= close_price)
);

-- Stock screening criteria definitions
-- Stores reusable screening rules for stock selection
CREATE TABLE screening_criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    criteria_name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    criteria_rules TEXT NOT NULL, -- JSON string containing screening logic
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Stock screening execution results
-- Stores results from applying screening criteria to stocks
CREATE TABLE screening_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screening_criteria_id INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    execution_date DATE NOT NULL,
    screening_score DECIMAL(10, 4) NOT NULL,
    result_metadata TEXT, -- JSON string with additional screening details
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (screening_criteria_id) REFERENCES screening_criteria(id) ON DELETE CASCADE,
    FOREIGN KEY (symbol) REFERENCES stock(symbol) ON DELETE CASCADE,
    INDEX idx_screening_result_criteria_date (screening_criteria_id, execution_date),
    INDEX idx_screening_result_symbol (symbol)
);

-- Backtesting strategy definitions
-- Defines complete trading strategies with entry/exit conditions
CREATE TABLE backtesting_strategy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name VARCHAR(255) NOT NULL UNIQUE,
    entry_criteria_id INTEGER NOT NULL,
    exit_criteria_id INTEGER,
    strategy_parameters TEXT NOT NULL, -- JSON string with strategy configuration
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entry_criteria_id) REFERENCES screening_criteria(id),
    FOREIGN KEY (exit_criteria_id) REFERENCES screening_criteria(id)
);

-- Backtesting execution results
-- Stores comprehensive results from strategy backtesting runs
CREATE TABLE backtesting_execution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    backtest_start_date DATE NOT NULL,
    backtest_end_date DATE NOT NULL,
    initial_capital_usd DECIMAL(15, 2) NOT NULL CHECK (initial_capital_usd > 0),
    final_portfolio_value_usd DECIMAL(15, 2) NOT NULL CHECK (final_portfolio_value_usd >= 0),
    total_return_percentage DECIMAL(10, 4) NOT NULL,
    sharpe_ratio DECIMAL(10, 4),
    maximum_drawdown_percentage DECIMAL(10, 4),
    total_trades_count INTEGER DEFAULT 0 CHECK (total_trades_count >= 0),
    winning_trades_percentage DECIMAL(5, 4) CHECK (winning_trades_percentage BETWEEN 0 AND 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES backtesting_strategy(id) ON DELETE CASCADE
);

-- Individual trading transactions
-- Records each buy/sell transaction from backtesting execution
CREATE TABLE trading_transaction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtesting_execution_id INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    entry_date DATE NOT NULL,
    exit_date DATE,
    entry_price_usd DECIMAL(12, 4) NOT NULL CHECK (entry_price_usd > 0),
    exit_price_usd DECIMAL(12, 4) CHECK (exit_price_usd > 0),
    share_quantity INTEGER NOT NULL CHECK (share_quantity > 0),
    profit_loss_usd DECIMAL(15, 2),
    position_type VARCHAR(10) NOT NULL CHECK (position_type IN ('LONG', 'SHORT')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (backtesting_execution_id) REFERENCES backtesting_execution(id) ON DELETE CASCADE,
    FOREIGN KEY (symbol) REFERENCES stock(symbol)
);

-- Performance optimization indexes
-- Optimized for common query patterns in stock analysis

-- Stock price queries by symbol and date range
CREATE INDEX idx_stock_price_symbol_date_range ON stock_price_history(symbol, trade_date);

-- Stock lookup by sector and industry for screening
CREATE INDEX idx_stock_sector_industry_lookup ON stock(sector, industry);

-- Trading transaction analysis by execution and symbol
CREATE INDEX idx_trading_transaction_execution ON trading_transaction(backtesting_execution_id);
CREATE INDEX idx_trading_transaction_symbol_dates ON trading_transaction(symbol, entry_date, exit_date);

-- Strategy performance analysis
CREATE INDEX idx_backtesting_execution_strategy ON backtesting_execution(strategy_id);

-- Active screening criteria lookup
CREATE INDEX idx_screening_criteria_active ON screening_criteria(is_active, criteria_name);