from enum import Enum
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseConfig(BaseSettings):
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    username: str = Field(default="postgres", env="DB_USERNAME")
    password: str = Field(default="password", env="DB_PASSWORD")
    database: str = Field(default="trading_system", env="DB_DATABASE")

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @validator("port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class RedisConfig(BaseSettings):
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    @validator("port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @validator("db")
    def validate_db(cls, v):
        if not 0 <= v <= 15:
            raise ValueError("Redis DB must be between 0 and 15")
        return v


class ServiceConfig(BaseSettings):
    order_service_port: int = Field(default=8001, env="ORDER_SERVICE_PORT")
    portfolio_service_port: int = Field(default=8002, env="PORTFOLIO_SERVICE_PORT")
    risk_service_port: int = Field(default=8003, env="RISK_SERVICE_PORT")
    gateway_port: int = Field(default=8000, env="GATEWAY_PORT")

    order_service_host: str = Field(default="localhost", env="ORDER_SERVICE_HOST")
    portfolio_service_host: str = Field(default="localhost", env="PORTFOLIO_SERVICE_HOST")
    risk_service_host: str = Field(default="localhost", env="RISK_SERVICE_HOST")

    @property
    def order_service_url(self) -> str:
        return f"http://{self.order_service_host}:{self.order_service_port}"

    @property
    def portfolio_service_url(self) -> str:
        return f"http://{self.portfolio_service_host}:{self.portfolio_service_port}"

    @property
    def risk_service_url(self) -> str:
        return f"http://{self.risk_service_host}:{self.risk_service_port}"


class SecurityConfig(BaseSettings):
    secret_key: str = Field(env="SECRET_KEY", min_length=32)
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")

    # API Keys for external services
    market_data_api_key: Optional[str] = Field(default=None, env="MARKET_DATA_API_KEY")
    risk_engine_api_key: Optional[str] = Field(default=None, env="RISK_ENGINE_API_KEY")

    @validator("secret_key")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v


class FeatureFlags(BaseSettings):
    enable_real_time_risk: bool = Field(default=True, env="ENABLE_REAL_TIME_RISK")
    enable_market_data_feed: bool = Field(default=True, env="ENABLE_MARKET_DATA_FEED")
    enable_position_limits: bool = Field(default=True, env="ENABLE_POSITION_LIMITS")
    enable_trade_notifications: bool = Field(default=False, env="ENABLE_TRADE_NOTIFICATIONS")
    enable_advanced_analytics: bool = Field(default=False, env="ENABLE_ADVANCED_ANALYTICS")
    maintenance_mode: bool = Field(default=False, env="MAINTENANCE_MODE")


class LoggingConfig(BaseSettings):
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class Settings(BaseSettings):
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Service configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    services: ServiceConfig = ServiceConfig()
    security: SecurityConfig = SecurityConfig()
    features: FeatureFlags = FeatureFlags()
    logging: LoggingConfig = LoggingConfig()

    # API Configuration
    api_title: str = Field(default="Trading System API", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    api_description: str = Field(
        default="Microservices-based trading system with order management, portfolio tracking, and risk assessment",
        env="API_DESCRIPTION",
    )

    # Performance settings
    max_connections_count: int = Field(default=10, env="MAX_CONNECTIONS_COUNT")
    min_connections_count: int = Field(default=1, env="MIN_CONNECTIONS_COUNT")

    # CORS settings
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")

    class Config:
        env_file = ".env"
        case_sensitive = False
        validate_assignment = True

    @validator("environment", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @validator("debug", pre=True)
    def set_debug_from_env(cls, v, values):
        # Auto-enable debug in development
        env = values.get("environment")
        if env == Environment.DEVELOPMENT:
            return True
        return v

    @validator("allowed_hosts", pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency function to get settings instance.
    Useful for FastAPI dependency injection.
    """
    return settings


# Environment-specific overrides
def is_development() -> bool:
    return settings.environment == Environment.DEVELOPMENT


def is_staging() -> bool:
    return settings.environment == Environment.STAGING


def is_production() -> bool:
    return settings.environment == Environment.PRODUCTION
