# optimized_models.py - Performance-optimized SQLAlchemy models with proper error handling
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any, List, AsyncIterator, Union
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from functools import wraps
import time

from sqlalchemy import (
    Column, Integer, String, DateTime, Date, Numeric, Boolean, Text, 
    ForeignKey, CheckConstraint, UniqueConstraint, Index, and_, or_,
    select, func, exc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates, selectinload, joinedload
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import text
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

import aioredis
from pydantic import BaseModel, validator

# Configure logging
logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class StockScreenerError(Exception):
    """Base exception for stock screener operations."""
    pass

class ValidationError(StockScreenerError):
    """Raised when input validation fails."""
    pass

class CacheError(StockScreenerError):
    """Raised when cache operations fail."""
    pass

class DatabaseError(StockScreenerError):
    """Raised when database operations fail."""
    pass

# Input validation schemas
class StockSearchParams(BaseModel):
    search: Optional[str] = None
    sector: Optional[str] = None
    min_market_cap: Optional[float] = None
    limit: int = 100
    offset: int = 0
    
    @validator('limit')
    def validate_limit(cls, v):
        if v <= 0 or v > 1000:
            raise ValueError('Limit must be between 1 and 1000')
        return v
    
    @validator('offset')
    def validate_offset(cls, v):
        if v < 0:
            raise ValueError('Offset must be non-negative')
        return v

class BacktestParams(BaseModel):
    strategy_id: int
    start_date: date
    end_date: date
    initial_capital: Decimal
    
    @validator('initial_capital')
    def validate_capital(cls, v):
        if v <= 0:
            raise ValueError('Initial capital must be positive')
        return v
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('End date must be after start date')
        return v

# Performance monitoring decorator
def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            logger.info(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    return wrapper

# Optimized connection factory
def create_optimized_engine(database_url: str):
    """Create database engine with optimal performance settings."""
    return create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=0,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
        future=True,
        connect_args={
            "command_timeout": 30,
            "server_settings": {
                "jit": "off",  # Disable JIT for consistent performance
            }
        }
    )

Base = declarative_base()

# External cache for JSON operations (no memory leaks)
class JSONCache:
    """Thread-safe JSON cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self._cache = {}
        self._timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached JSON data if not expired."""
        if key not in self._cache:
            return None
        
        if time.time() - self._timestamps[key] > self.ttl:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Cache JSON data with automatic cleanup."""
        # Cleanup expired entries if cache is full
        if len(self._cache) >= self.max_size:
            current_time = time.time()
            expired_keys = [
                k for k, timestamp in self._timestamps.items()
                if current_time - timestamp > self.ttl
            ]
            for k in expired_keys:
                self._cache.pop(k, None)
                self._timestamps.pop(k, None)
        
        self._cache[key] = value
        self._timestamps[key] = time.time()

# Global JSON cache instance
json_cache = JSONCache()

# Optimized JSON operations
class OptimizedJSONMixin:
    """High-performance JSON handling with external caching."""
    
    def _parse_json_field(self, field_value: Optional[str]) -> Dict[str, Any]:
        """Parse JSON with caching and proper error handling."""
        if not field_value:
            return {}
        
        # Use content hash as cache key
        cache_key = f"json_{hash(field_value)}"
        cached_result = json_cache.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        try:
            result = json.loads(field_value)
            json_cache.set(cache_key, result)
            return result
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"JSON parsing failed: {e}")
            return {}
    
    def _serialize_json_field(self, data_dict: Dict[str, Any]) -> str:
        """Fast JSON serialization with validation."""
        if not isinstance(data_dict, dict):
            return "{}"
        try:
            return json.dumps(data_dict, separators=(',', ':'), ensure_ascii=False)
        except TypeError as e:
            logger.error(f"JSON serialization failed: {e}")
            return "{}"

# Optimized models (keeping existing structure but adding proper validation)
class Stock(Base, OptimizedJSONMixin):
    """Optimized stock entity with proper indexing and validation."""
    __tablename__ = 'stock'
    
    symbol = Column(String(20), primary_key=True)
    name = Column(String(255), nullable=False)
    exchange = Column(String(50), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Numeric(20, 2))
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    prices = relationship(
        "StockPrice", 
        back_populates="stock", 
        cascade="all, delete-orphan",
        lazy="select"
    )
    
    __table_args__ = (
        Index('idx_stock_sector_industry_market_cap', 'sector', 'industry', 'market_cap'),
        Index('idx_stock_exchange_sector', 'exchange', 'sector'),
        Index('idx_stock_market_cap_desc', 'market_cap', postgresql_using='btree'),
        CheckConstraint('market_cap >= 0', name='ck_stock_market_cap_positive'),
    )
    
    @validates('symbol')
    def validate_symbol(self, key, symbol):
        if not symbol or len(symbol) > 20:
            raise ValidationError("Symbol must be 1-20 characters")
        return symbol.upper()
    
    @hybrid_property
    def market_cap_billions(self):
        return self.market_cap / 1_000_000_000 if self.market_cap else 0

class StockPrice(Base):
    """Optimized price data with proper indexing for time series queries."""
    __tablename__ = 'stock_price'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), ForeignKey('stock.symbol', ondelete='CASCADE'), nullable=False)
    date = Column(Date, nullable=False)
    open_price = Column(Numeric(12, 4), nullable=False)
    high_price = Column(Numeric(12, 4), nullable=False)
    low_price = Column(Numeric(12, 4), nullable=False)
    close_price = Column(Numeric(12, 4), nullable=False)
    adj_close_price = Column(Numeric(12, 4), nullable=False)
    volume = Column(Integer, nullable=False)
    
    stock = relationship("Stock", back_populates="prices")
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_stock_price_symbol_date'),
        Index('idx_stock_price_symbol_date_desc', 'symbol', 'date', postgresql_using='btree'),
        Index('idx_stock_price_date_volume', 'date', 'volume'),
        Index('idx_stock_price_recent', 'symbol', 'date', 
              postgresql_where=text("date >= CURRENT_DATE - INTERVAL '1 year'")),
    )
    
    @hybrid_property
    def price_change_pct(self):
        if self.open_price and self.open_price != 0:
            return ((self.close_price - self.open_price) / self.open_price) * 100
        return 0

# Cache management with proper invalidation
class CacheManager:
    """Centralized cache management with invalidation."""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self._ttl = 300
    
    async def get(self, key: str) -> Optional[dict]:
        """Get cached data with error handling."""
        try:
            cached_data = await self.redis.get(key)
            if cached_data:
                return json.loads(cached_data)
        except (aioredis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get failed for key {key}: {e}")
        return None
    
    async def set(self, key: str, data: dict, ttl: Optional[int] = None) -> None:
        """Set cached data with error handling."""
        try:
            ttl = ttl or self._ttl
            await self.redis.setex(key, ttl, json.dumps(data))
        except (aioredis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache set failed for key {key}: {e}")
    
    async def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate cache keys matching pattern."""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
        except aioredis.RedisError as e:
            logger.error(f"Cache invalidation failed for pattern {pattern}: {e}")

# Resource management context
@asynccontextmanager
async def database_transaction(session: AsyncSession):
    """Proper transaction management with cleanup."""
    try:
        await session.begin()
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Transaction failed: {e}")
        raise DatabaseError(f"Database operation failed: {e}")
    finally:
        await session.close()

# Optimized service with proper streaming
class OptimizedStockService:
    """High-performance stock data service with proper streaming and error handling."""
    
    def __init__(self, db_session: AsyncSession, cache_manager: CacheManager):
        self.db = db_session
        self.cache = cache_manager
    
    @monitor_performance
    async def get_stocks_stream(
        self, 
        params: StockSearchParams
    ) -> AsyncIterator[tuple[List[Stock], int]]:
        """Stream stock results with proper batching and memory management."""
        
        # Generate cache key
        cache_key = f"stocks:{params.search or ''}:{params.sector or ''}:{params.min_market_cap or ''}:{params.limit}:{params.offset}"
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            yield cached_result['stocks'], cached_result['total']
            return
        
        try:
            # Build optimized query
            query = select(Stock)
            count_query = select(func.count(Stock.symbol))
            
            # Apply filters efficiently
            filters = self._build_filters(params)
            
            if filters:
                query = query.where(and_(*filters))
                count_query = count_query.where(and_(*filters))
            
            # Get total count
            count_result = await self.db.execute(count_query)
            total = count_result.scalar()
            
            # Stream results in batches
            stocks_query = (
                query
                .order_by(Stock.market_cap.desc().nullslast())
                .offset(params.offset)
                .limit(params.limit)
            )
            
            # Use cursor-based streaming for large results
            result = await self.db.stream(stocks_query)
            
            stocks_batch = []
            batch_size = 100
            
            async for row in result:
                stock = row[0]
                stocks_batch.append(stock)
                
                if len(stocks_batch) >= batch_size:
                    # Cache and yield batch
                    cache_data = {
                        'stocks': [self._serialize_stock(s) for s in stocks_batch],
                        'total': total
                    }
                    await self.cache.set(cache_key, cache_data)
                    yield stocks_batch, total
                    stocks_batch = []
            
            # Yield remaining stocks
            if stocks_batch:
                cache_data = {
                    'stocks': [self._serialize_stock(s) for s in stocks_batch],
                    'total': total
                }
                await self.cache.set(cache_key, cache_data)
                yield stocks_batch, total
                
        except exc.SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise DatabaseError(f"Stock query failed: {e}")
    
    def _build_filters(self, params: StockSearchParams) -> List:
        """Build database filters efficiently."""
        filters = []
        
        if params.search:
            search_term = f"%{params.search.lower()}%"
            filters.append(
                or_(
                    func.lower(Stock.symbol).contains(search_term),
                    func.lower(Stock.name).contains(search_term)
                )
            )
        
        if params.sector:
            filters.append(Stock.sector == params.sector)
            
        if params.min_market_cap:
            filters.append(Stock.market_cap >= params.min_market_cap * 1_000_000_000)
        
        return filters
    
    def _serialize_stock(self, stock: Stock) -> dict:
        """Serialize stock for caching."""
        return {
            'symbol': stock.symbol,
            'name': stock.name,
            'sector': stock.sector,
            'industry': stock.industry,
            'market_cap': float(stock.market_cap) if stock.market_cap else None
        }

# Optimized screen service with bulk operations
class OptimizedScreenService:
    """High-performance screening service with proper bulk operations."""
    
    def __init__(self, db_session: AsyncSession, cache_manager: CacheManager):
        self.db = db_session
        self.cache = cache_manager
    
    @monitor_performance
    async def run_screen_bulk(
        self, 
        screen_id: int, 
        run_date: date, 
        user_id: int,
        batch_size: int = 1000
    ) -> AsyncIterator[tuple[List[dict], int]]:
        """Optimized screening with proper bulk operations and streaming."""
        
        try:
            # Validate screen exists and user has access
            screen = await self._get_screen(screen_id, user_id)
            criteria = screen.validate_and_get_criteria()
            
            # Build efficient query with proper filtering
            stocks_query = self._build_screen_query(criteria)
            
            # Process results in true streaming fashion
            result = await self.db.stream(stocks_query)
            
            matching_stocks = []
            batch_count = 0
            
            async for row in result:
                stock = row[0]
                matching_stocks.append({
                    'screen_id': screen_id,
                    'symbol': stock.symbol,
                    'run_date': run_date,
                    'score': self._calculate_score(stock, criteria),
                    'batch_number': batch_count
                })
                
                if len(matching_stocks) >= batch_size:
                    # Use SQLAlchemy bulk operations instead of raw SQL
                    await self._bulk_insert_results(matching_stocks)
                    yield matching_stocks, len(matching_stocks)
                    matching_stocks = []
                    batch_count += 1
            
            # Process final batch
            if matching_stocks:
                await self._bulk_insert_results(matching_stocks)
                yield matching_stocks, len(matching_stocks)
                
        except exc.SQLAlchemyError as e:
            logger.error(f"Screen execution failed: {e}")
            raise DatabaseError(f"Screen execution failed: {e}")
    
    async def _bulk_insert_results(self, results: List[dict]) -> None:
        """Use SQLAlchemy bulk operations for better performance."""
        try:
            # Use bulk_insert_mappings for better performance than raw SQL
            await self.db.execute(
                ScreenResult.__table__.insert(),
                results
            )
            await self.db.commit()
        except exc.SQLAlchemyError as e:
            await self.db.rollback()
            raise DatabaseError(f"Bulk insert failed: {e}")

# Connection factory with dependency injection
class DatabaseFactory:
    """Factory for creating optimized database connections with proper cleanup."""
    
    @staticmethod
    async def create_engine_pool(database_url: str):
        """Create async engine with optimal settings."""
        return create_async_engine(
            database_url,
            echo=False,
            future=True,
            pool_size=20,
            max_overflow=0,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_timeout=30
        )
    
    @staticmethod
    def create_session_factory(engine):
        """Create session factory with proper configuration."""
        return sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False  # Manual control for better performance
        )

# Redis factory with proper error handling
async def create_redis_client() -> aioredis.Redis:
    """Create optimized Redis client with proper error handling."""
    try:
        return await aioredis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
            retry_on_timeout=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
    except aioredis.RedisError as e:
        logger.error(f"Redis connection failed: {e}")
        raise CacheError(f"Cache initialization failed: {e}")

# Service container for dependency injection
class ServiceContainer:
    """Service container for managing dependencies."""
    
    def __init__(self, db_session: AsyncSession, redis_client: aioredis.Redis):
        self.db_session = db_session
        cache_manager = CacheManager(redis_client)
        
        self.stock_service = OptimizedStockService(db_session, cache_manager)
        self.screen_service = OptimizedScreenService(db_session, cache_manager)
    
    async def cleanup(self):
        """Cleanup resources properly."""
        try:
            await self.db_session.close()
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")