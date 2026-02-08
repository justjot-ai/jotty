from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any
import json

from sqlalchemy import (
    Column, Integer, String, DateTime, Date, Numeric, Boolean, Text, 
    ForeignKey, CheckConstraint, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

# Constants for better readability
TRADE_TYPES = ('LONG', 'SHORT')
DEFAULT_PRECISION = (10, 4)
CURRENCY_PRECISION = (15, 2)
SYMBOL_MAX_LENGTH = 20
NAME_MAX_LENGTH = 255

Base = declarative_base()


class JSONMixin:
    """Mixin class for handling JSON serialization/deserialization."""
    
    def _parse_json_field(self, field_value: Optional[str]) -> Dict[str, Any]:
        """Parse JSON string field to dictionary.
        
        Args:
            field_value: JSON string or None
            
        Returns:
            Parsed dictionary or empty dict if None
        """
        return json.loads(field_value) if field_value else {}
    
    def _serialize_json_field(self, data_dict: Dict[str, Any]) -> str:
        """Serialize dictionary to JSON string.
        
        Args:
            data_dict: Dictionary to serialize
            
        Returns:
            JSON string representation
        """
        return json.dumps(data_dict)


class Stock(Base):
    """Stock entity representing a tradable security.
    
    Stores basic stock information including symbol, name, exchange,
    and classification details like sector and industry.
    """
    __tablename__ = 'stock'
    
    # Primary identification
    symbol = Column(String(SYMBOL_MAX_LENGTH), primary_key=True)
    name = Column(String(NAME_MAX_LENGTH), nullable=False)
    exchange = Column(String(50), nullable=False)
    
    # Classification
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Numeric(*CURRENCY_PRECISION))
    
    # Timestamps
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, 
        default=func.current_timestamp(), 
        onupdate=func.current_timestamp()
    )
    
    # Relationships
    prices = relationship(
        "StockPrice", 
        back_populates="stock", 
        cascade="all, delete-orphan"
    )
    screen_results = relationship(
        "ScreenResult", 
        back_populates="stock", 
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index('idx_stock_sector_industry', 'sector', 'industry'),
    )
    
    def __repr__(self) -> str:
        return f"<Stock(symbol='{self.symbol}', name='{self.name}')>"


class StockPrice(Base):
    """Historical price data for stocks.
    
    Stores daily OHLCV (Open, High, Low, Close, Volume) data
    with adjusted close prices for historical analysis.
    """
    __tablename__ = 'stock_price'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(
        String(SYMBOL_MAX_LENGTH), 
        ForeignKey('stock.symbol', ondelete='CASCADE'), 
        nullable=False
    )
    date = Column(Date, nullable=False)
    
    # OHLC prices
    open_price = Column(Numeric(*DEFAULT_PRECISION), nullable=False)
    high_price = Column(Numeric(*DEFAULT_PRECISION), nullable=False)
    low_price = Column(Numeric(*DEFAULT_PRECISION), nullable=False)
    close_price = Column(Numeric(*DEFAULT_PRECISION), nullable=False)
    adj_close_price = Column(Numeric(*DEFAULT_PRECISION), nullable=False)
    
    # Volume data
    volume = Column(Integer, nullable=False)
    
    # Relationships
    stock = relationship("Stock", back_populates="prices")
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_stock_price_symbol_date'),
        Index('idx_stock_price_symbol_date', 'symbol', 'date'),
    )
    
    @validates('volume')
    def validate_volume(self, key: str, volume: int) -> int:
        """Validate volume is non-negative."""
        if volume < 0:
            raise ValueError(f"Volume cannot be negative, got: {volume}")
        return volume
    
    def __repr__(self) -> str:
        return (
            f"<StockPrice(symbol='{self.symbol}', "
            f"date='{self.date}', close={self.close_price})>"
        )


class Screen(Base, JSONMixin):
    """Stock screening criteria and configuration.
    
    Defines rules and parameters for filtering stocks based on
    various financial and technical criteria.
    """
    __tablename__ = 'screen'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(NAME_MAX_LENGTH), nullable=False)
    description = Column(Text)
    
    # JSON stored criteria
    criteria = Column(Text, nullable=False)
    
    # Metadata
    created_by = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, 
        default=func.current_timestamp(), 
        onupdate=func.current_timestamp()
    )
    
    # Relationships
    screen_results = relationship(
        "ScreenResult", 
        back_populates="screen", 
        cascade="all, delete-orphan"
    )
    entry_strategies = relationship(
        "BacktestStrategy", 
        foreign_keys="BacktestStrategy.entry_screen_id", 
        back_populates="entry_screen"
    )
    exit_strategies = relationship(
        "BacktestStrategy", 
        foreign_keys="BacktestStrategy.exit_screen_id", 
        back_populates="exit_screen"
    )
    
    def get_criteria(self) -> Dict[str, Any]:
        """Get screening criteria as dictionary.
        
        Returns:
            Dictionary containing screening criteria
        """
        return self._parse_json_field(self.criteria)
    
    def set_criteria(self, criteria_dict: Dict[str, Any]) -> None:
        """Set screening criteria from dictionary.
        
        Args:
            criteria_dict: Dictionary containing screening criteria
        """
        self.criteria = self._serialize_json_field(criteria_dict)
    
    def __repr__(self) -> str:
        return (
            f"<Screen(id={self.id}, name='{self.name}', "
            f"active={self.is_active})>"
        )


class ScreenResult(Base, JSONMixin):
    """Results from running a stock screen.
    
    Stores which stocks passed screening criteria on specific dates,
    along with their scores and additional metadata.
    """
    __tablename__ = 'screen_result'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    screen_id = Column(
        Integer, 
        ForeignKey('screen.id', ondelete='CASCADE'), 
        nullable=False
    )
    symbol = Column(
        String(SYMBOL_MAX_LENGTH), 
        ForeignKey('stock.symbol', ondelete='CASCADE'), 
        nullable=False
    )
    run_date = Column(Date, nullable=False)
    score = Column(Numeric(*DEFAULT_PRECISION), nullable=False)
    
    # Additional metadata as JSON
    metadata = Column(Text)
    
    # Relationships
    screen = relationship("Screen", back_populates="screen_results")
    stock = relationship("Stock", back_populates="screen_results")
    
    __table_args__ = (
        Index('idx_screen_result_screen_date', 'screen_id', 'run_date'),
        Index('idx_screen_result_symbol', 'symbol'),
    )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get result metadata as dictionary.
        
        Returns:
            Dictionary containing result metadata
        """
        return self._parse_json_field(self.metadata)
    
    def set_metadata(self, metadata_dict: Dict[str, Any]) -> None:
        """Set result metadata from dictionary.
        
        Args:
            metadata_dict: Dictionary containing result metadata
        """
        self.metadata = self._serialize_json_field(metadata_dict)
    
    def __repr__(self) -> str:
        return (
            f"<ScreenResult(screen_id={self.screen_id}, "
            f"symbol='{self.symbol}', score={self.score})>"
        )


class BacktestStrategy(Base, JSONMixin):
    """Backtesting strategy configuration.
    
    Defines entry/exit screens and parameters for backtesting
    trading strategies against historical data.
    """
    __tablename__ = 'backtest_strategy'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(NAME_MAX_LENGTH), nullable=False)
    
    # Screen references
    entry_screen_id = Column(Integer, ForeignKey('screen.id'), nullable=False)
    exit_screen_id = Column(Integer, ForeignKey('screen.id'), nullable=True)
    
    # Strategy parameters as JSON
    parameters = Column(Text, nullable=False)
    
    # Timestamp
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    entry_screen = relationship(
        "Screen", 
        foreign_keys=[entry_screen_id], 
        back_populates="entry_strategies"
    )
    exit_screen = relationship(
        "Screen", 
        foreign_keys=[exit_screen_id], 
        back_populates="exit_strategies"
    )
    backtest_runs = relationship(
        "BacktestRun", 
        back_populates="strategy", 
        cascade="all, delete-orphan"
    )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters as dictionary.
        
        Returns:
            Dictionary containing strategy parameters
        """
        return self._parse_json_field(self.parameters)
    
    def set_parameters(self, parameters_dict: Dict[str, Any]) -> None:
        """Set strategy parameters from dictionary.
        
        Args:
            parameters_dict: Dictionary containing strategy parameters
        """
        self.parameters = self._serialize_json_field(parameters_dict)
    
    def __repr__(self) -> str:
        return f"<BacktestStrategy(id={self.id}, name='{self.name}')>"


class BacktestRun(Base):
    """Historical backtest execution results.
    
    Stores performance metrics and summary statistics from running
    a backtesting strategy over a specific time period.
    """
    __tablename__ = 'backtest_run'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(
        Integer, 
        ForeignKey('backtest_strategy.id', ondelete='CASCADE'), 
        nullable=False
    )
    
    # Date range
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    
    # Portfolio values
    initial_capital = Column(Numeric(*CURRENCY_PRECISION), nullable=False)
    final_portfolio_value = Column(Numeric(*CURRENCY_PRECISION), nullable=False)
    
    # Performance metrics
    total_return = Column(Numeric(*DEFAULT_PRECISION), nullable=False)
    sharpe_ratio = Column(Numeric(*DEFAULT_PRECISION))
    max_drawdown = Column(Numeric(*DEFAULT_PRECISION))
    
    # Trade statistics
    trades_count = Column(Integer, default=0)
    win_rate = Column(Numeric(5, 4))
    
    # Timestamp
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    strategy = relationship("BacktestStrategy", back_populates="backtest_runs")
    trades = relationship(
        "Trade", 
        back_populates="backtest_run", 
        cascade="all, delete-orphan"
    )
    
    @validates('initial_capital', 'final_portfolio_value')
    def validate_capital_amounts(self, key: str, value: Decimal) -> Decimal:
        """Validate capital amounts are positive."""
        if value <= 0:
            raise ValueError(
                f"{key.replace('_', ' ').title()} must be positive, got: {value}"
            )
        return value
    
    @validates('start_date', 'end_date')
    def validate_date_range(self, key: str, value: date) -> date:
        """Validate date range is logical."""
        if (hasattr(self, 'start_date') and hasattr(self, 'end_date') and 
            key == 'end_date' and self.start_date and value < self.start_date):
            raise ValueError(
                f"End date ({value}) must be after start date ({self.start_date})"
            )
        return value
    
    def __repr__(self) -> str:
        return (
            f"<BacktestRun(id={self.id}, strategy_id={self.strategy_id}, "
            f"return={self.total_return})>"
        )


class Trade(Base):
    """Individual trade execution record.
    
    Represents a single buy/sell transaction within a backtest,
    tracking entry/exit prices and profit/loss calculations.
    """
    __tablename__ = 'trade'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id = Column(
        Integer, 
        ForeignKey('backtest_run.id', ondelete='CASCADE'), 
        nullable=False
    )
    symbol = Column(String(SYMBOL_MAX_LENGTH), nullable=False)
    
    # Trade timing
    entry_date = Column(Date, nullable=False)
    exit_date = Column(Date, nullable=True)
    
    # Trade pricing
    entry_price = Column(Numeric(*DEFAULT_PRECISION), nullable=False)
    exit_price = Column(Numeric(*DEFAULT_PRECISION), nullable=True)
    quantity = Column(Integer, nullable=False)
    
    # Trade results
    pnl = Column(Numeric(*CURRENCY_PRECISION), nullable=True)
    trade_type = Column(String(10), nullable=False)
    
    # Relationships
    backtest_run = relationship("BacktestRun", back_populates="trades")
    
    __table_args__ = (
        CheckConstraint(
            f"trade_type IN {TRADE_TYPES}", 
            name='ck_trade_type'
        ),
        Index('idx_trade_backtest_run', 'backtest_run_id'),
        Index('idx_trade_symbol', 'symbol'),
    )
    
    @validates('quantity')
    def validate_share_quantity(self, key: str, quantity: int) -> int:
        """Validate quantity is positive."""
        if quantity <= 0:
            raise ValueError(f"Share quantity must be positive, got: {quantity}")
        return quantity
    
    @validates('entry_price', 'exit_price')
    def validate_price_values(self, key: str, price: Optional[Decimal]) -> Optional[Decimal]:
        """Validate price values are positive when set."""
        if price is not None and price <= 0:
            raise ValueError(
                f"{key.replace('_', ' ').title()} must be positive, got: {price}"
            )
        return price
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open (no exit date)."""
        return self.exit_date is None
    
    def calculate_pnl(self) -> Optional[Decimal]:
        """Calculate profit/loss for completed trades.
        
        Returns:
            Profit/loss amount for closed trades, None for open trades
        """
        if self.exit_price is None:
            return None
        
        price_difference = (
            self.exit_price - self.entry_price 
            if self.trade_type == 'LONG' 
            else self.entry_price - self.exit_price
        )
        
        return price_difference * self.quantity
    
    def __repr__(self) -> str:
        status = "OPEN" if self.is_open else "CLOSED"
        return (
            f"<Trade(id={self.id}, symbol='{self.symbol}', "
            f"type={self.trade_type}, status={status})>"
        )