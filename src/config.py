import os
from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional

class ModelType(Enum):
    STABLE_DIFFUSION = "stable_diffusion"
    DALLE = "dalle"
    LANGCHAIN_OPENAI = "langchain_openai"
    TEST_MODE = "test_mode"

class Config(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # Test Mode
    test_mode: bool = Field(False, env="TEST_MODE")
    
    # Image Processing
    allowed_image_types: tuple = ("jpg", "jpeg", "png")
    max_image_dimension: int = 1024  # Max width/height for optimization
    
    # Model Configuration  
    model_type: ModelType = ModelType.LANGCHAIN_OPENAI
    
    @property
    def effective_model_type(self) -> ModelType:
        """Get the effective model type based on test mode"""
        if self.test_mode:
            return ModelType.TEST_MODE
        return self.model_type
    stable_diffusion_model: str = "stabilityai/stable-diffusion-2-1"
    dalle_model: str = "dall-e-3"
    
    # LangChain Configuration
    langchain_temperature: float = 0.7
    langchain_max_tokens: int = 1000
    langchain_model_name: str = "gpt-4-vision-preview"
    
    # Production Settings
    cache_ttl: int = 3600  # 1 hour
    max_retries: int = 3
    request_timeout: int = 30
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

config = Config()
