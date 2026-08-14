"""配置管理：从 .env 和环境变量读取配置."""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（项目根目录）
load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    """全局配置单例."""

    # DeepSeek API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///replenishment.db")

    # App
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_DEBUG: bool = os.getenv("APP_DEBUG", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Decision Engine
    DECISION_MAX_TOOL_ROUNDS: int = int(os.getenv("DECISION_MAX_TOOL_ROUNDS", "10"))
    DECISION_TEMPERATURE: float = float(os.getenv("DECISION_TEMPERATURE", "0.3"))
    DECISION_LLM_MODEL: str = os.getenv("DECISION_LLM_MODEL", "deepseek-chat")

    # Conversation Engine
    CONVERSATION_TEMPERATURE: float = float(os.getenv("CONVERSATION_TEMPERATURE", "0.7"))
    CONVERSATION_LLM_MODEL: str = os.getenv("CONVERSATION_LLM_MODEL", "deepseek-chat")
    CONVERSATION_MAX_HISTORY: int = int(os.getenv("CONVERSATION_MAX_HISTORY", "10"))

    # Embedding（阶段二 向量检索；不配 Key → 自动回退结构化检索）
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    EMBEDDING_TOP_K: int = int(os.getenv("EMBEDDING_TOP_K", "3"))

    @classmethod
    def validate(cls) -> list[str]:
        """验证必要配置，返回缺失项列表."""
        missing = []
        if not cls.DEEPSEEK_API_KEY:
            missing.append("DEEPSEEK_API_KEY")
        return missing


config = Config()
