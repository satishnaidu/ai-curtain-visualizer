import os
from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional

try:
    import streamlit as st
    _has_streamlit = True
except ImportError:
    _has_streamlit = False

class ModelType(Enum):
    STABLE_DIFFUSION = "stable_diffusion"
    DALLE = "dalle"
    LANGCHAIN_OPENAI = "langchain_openai"
    REPLICATE_CONTROLNET = "replicate_controlnet"
    REPLICATE_REALISTIC = "replicate_realistic"
    TEST_MODE = "test_mode"

class CurtainStyle(Enum):
    CLOSED = "closed"
    HALF_OPEN = "half_open"
    WITH_SHEERS = "with_sheers"

class Config(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = Field("your_api_key_here", env="OPENAI_API_KEY")
    replicate_api_token: Optional[str] = Field(None, env="REPLICATE_API_TOKEN")
    stripe_secret_key: Optional[str] = Field(None, env="STRIPE_SECRET_KEY")
    stripe_publishable_key: Optional[str] = Field(None, env="STRIPE_PUBLISHABLE_KEY")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Try to get API key from Streamlit secrets only in cloud environment
        if _has_streamlit:
            try:
                if 'OPENAI_API_KEY' in st.secrets:
                    self.openai_api_key = st.secrets['OPENAI_API_KEY']
                if 'REPLICATE_API_TOKEN' in st.secrets:
                    self.replicate_api_token = st.secrets['REPLICATE_API_TOKEN']
                if 'STRIPE_SECRET_KEY' in st.secrets:
                    self.stripe_secret_key = st.secrets['STRIPE_SECRET_KEY']
                if 'STRIPE_PUBLISHABLE_KEY' in st.secrets:
                    self.stripe_publishable_key = st.secrets['STRIPE_PUBLISHABLE_KEY']
                if 'TEST_MODE' in st.secrets:
                    self.test_mode = st.secrets['TEST_MODE']
            except:
                pass  # Ignore secrets errors in local development
    
    # Test Mode
    test_mode: bool = Field(False, env="TEST_MODE")
    
    # Image Processing
    allowed_image_types: tuple = ("jpg", "jpeg", "png")
    max_image_dimension: int = 1024  # Max width/height for optimization
    
    # Model Configuration  
    model_type: ModelType = ModelType.TEST_MODE  # Better composite approach
    
    # Curtain Style Configuration
    default_curtain_style: CurtainStyle = CurtainStyle.HALF_OPEN
    curtain_style_descriptions: dict = {
        CurtainStyle.CLOSED: "Elegant floor-length curtains hanging in graceful folds",
        CurtainStyle.HALF_OPEN: "Curtains elegantly pulled to one side, creating an asymmetrical drape",
        CurtainStyle.WITH_SHEERS: "Layered curtains with sheer inner layer for filtered light"
    }
    
    @property
    def effective_model_type(self) -> ModelType:
        """Get the effective model type with fallback priority"""
        if self.test_mode:
            return ModelType.TEST_MODE
        
        # Priority: OpenAI -> Replicate -> Test
        if self.openai_api_key and self.openai_api_key != "your_api_key_here":
            return ModelType.LANGCHAIN_OPENAI
        elif self.replicate_api_token:
            return ModelType.REPLICATE_CONTROLNET
        else:
            return ModelType.TEST_MODE
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
        # Reference images for different curtain styles
        curtain_folded_image_path = "assets/reference/curtain_folded.png"
        curtain_half_open_image_path = "assets/reference/curtain_half_open.png"
        curtain_sheers_image_path = "assets/reference/curtain_sheers.png"

# Initialize config safely
try:
    config = Config()
except Exception as e:
    # Fallback config for development
    import os
    config = Config(
        openai_api_key=os.getenv('OPENAI_API_KEY', 'your_api_key_here'),
        replicate_api_token=os.getenv('REPLICATE_API_TOKEN'),
        test_mode=os.getenv('TEST_MODE', 'false').lower() == 'true'
    )
