"""Configuration management using pydantic-settings."""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/LoopMind_Memory",
        description="PostgreSQL database URL with asyncpg driver",
    )
    postgres_user: str = Field(default="postgres", description="PostgreSQL user")
    postgres_password: str = Field(default="postgres", description="PostgreSQL password")
    postgres_db: str = Field(default="LoopMind_Memory", description="PostgreSQL database name")

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_username: Optional[str] = Field(default=None, description="Redis username (for cloud Redis)")
    redis_password: Optional[str] = Field(default=None, description="Redis password (for cloud Redis)")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key (for cloud Qdrant)")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="neo4j", description="Neo4j password")

    # JWT Authentication
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT token signing",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(
        default=30, description="JWT access token expiration in minutes"
    )

    # Ed25519 Keys
    ed25519_private_key_path: str = Field(
        default="./keys/private_key.pem", description="Path to Ed25519 private key"
    )
    ed25519_public_key_path: str = Field(
        default="./keys/public_key.pem", description="Path to Ed25519 public key"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_reload: bool = Field(default=True, description="Enable auto-reload in development")
    api_log_level: str = Field(default="info", description="API log level")

    # Feature Flags
    enable_antiviral: bool = Field(default=True, description="Enable ANTIVIRAL scanning")
    enable_neuralguard: bool = Field(default=True, description="Enable NeuralGuard")
    enable_adapter_training: bool = Field(
        default=False, description="Enable adapter training pipeline"
    )

    # Embedding Configuration
    embedding_model: str = Field(
        default="openai", description="Embedding model: local, openai, azure, gemini"
    )
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension (text-embedding-3-small: 1536)")
    embedding_batch_size: int = Field(default=32, description="Embedding batch size")

    # OpenAI (primary embedding provider)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )

    # Azure
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_embedding_deployment: Optional[str] = Field(
        default=None, description="Azure embedding deployment name"
    )

    # Gemini
    google_api_key: Optional[str] = Field(default=None, description="Google API key for Gemini")

    # S3/Object Store
    s3_endpoint_url: Optional[str] = Field(default=None, description="S3 endpoint URL")
    s3_access_key_id: Optional[str] = Field(default=None, description="S3 access key ID")
    s3_secret_access_key: Optional[str] = Field(default=None, description="S3 secret access key")
    s3_bucket_name: str = Field(
        default="LoopMind_Memory-audit", description="S3 bucket name for audit snapshots"
    )
    s3_region: str = Field(default="us-east-1", description="S3 region")

    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")

    # Security
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="Comma-separated list of allowed CORS origins",
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"],
        description="List of CORS origins",
        exclude=True,
    )

    # LangCache (Semantic Caching)
    langcache_api_key: Optional[str] = Field(
        default=None, 
        description="LangCache API key for semantic caching"
    )
    langcache_server_url: str = Field(
        default="https://gcp-us-east4.langcache.redis.io",
        description="LangCache server URL"
    )
    langcache_cache_id: str = Field(
        default="a61f984fdf7b49f486541cb0d20f26c5",
        description="LangCache cache ID"
    )
    enable_langcache: bool = Field(
        default=True,
        description="Enable LangCache semantic caching for queries"
    )

    def __init__(self, **kwargs):
        """Initialize settings and parse CORS origins."""
        super().__init__(**kwargs)
        # Parse comma-separated CORS origins
        if isinstance(self.allowed_origins, str):
            self.cors_origins = [origin.strip() for origin in self.allowed_origins.split(",")]


# Global settings instance
settings = Settings()

