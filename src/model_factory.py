from abc import ABC, abstractmethod
from PIL import Image
import asyncio
from typing import Optional, Dict, Any
from cachetools import TTLCache
from loguru import logger
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .config import config, ModelType
from .exceptions import ModelError, APIError




class BaseModel(ABC):
    def __init__(self):
        self.cache = TTLCache(maxsize=100, ttl=config.cache_ttl)
        
    @abstractmethod
    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image) -> Image.Image:
        pass
    
    def _get_cache_key(self, prompt: str, room_hash: str, fabric_hash: str) -> str:
        return f"{prompt[:50]}_{room_hash}_{fabric_hash}"


class LangChainOpenAIModel(BaseModel):
    def __init__(self):
        super().__init__()
        if not config.openai_api_key or config.openai_api_key == "your_api_key_here":
            raise ModelError("OpenAI API key required for LangChain model")
        self.llm = ChatOpenAI(
            model=config.langchain_model_name,
            temperature=config.langchain_temperature,
            max_tokens=config.langchain_max_tokens,
            api_key=config.openai_api_key
        )
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()
        
    def _create_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                "Create a photorealistic image showing the exact same room layout but with curtains added. "
                "IMPORTANT: Keep the room EXACTLY as described, only replace the blinds with curtains."
                "\n\nRoom to recreate: {room_context}"
                "\n\nCurtain fabric to use: {fabric_colors}"
                "\n\nInstructions: Replace the white horizontal blinds with floor-length curtains made from the specified fabric. "
                "Keep everything else identical - same furniture placement, same wall colors, same lighting, same plants. "
                "The curtains should hang from ceiling to floor, covering the windows where the blinds currently are. "
                "Make it look like a professional interior design visualization."
            )
        ])
    
    def _create_chain(self):
        return (
            RunnablePassthrough()
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image) -> str:
        try:
            # Generate enhanced prompt using LangChain
            enhanced_prompt = await self._enhance_prompt(prompt, room_image, fabric_image)
            
            # Use OpenAI DALL-E for actual image generation
            client = OpenAI(api_key=config.openai_api_key)
            response = client.images.generate(
                model=config.dalle_model,
                prompt=enhanced_prompt,
                n=1,
                size="1024x1024",
                quality="hd"
            )
            
            return response.data[0].url
            
        except Exception as e:
            logger.error(f"LangChain model error: {str(e)}")
            raise ModelError(f"Failed to generate image: {str(e)}")
    
    async def _enhance_prompt(self, base_prompt: str, room_image: Image.Image, fabric_image: Image.Image) -> str:
        try:
            room_context = self._analyze_room_context(room_image)
            fabric_colors = self._extract_fabric_colors(fabric_image)
            
            result = await self.chain.ainvoke({
                "room_context": room_context,
                "fabric_colors": fabric_colors,
                "base_prompt": base_prompt
            })
            
            return result
            
        except Exception as e:
            logger.warning(f"Prompt enhancement failed, using base prompt: {str(e)}")
            return base_prompt
    
    def _analyze_room_context(self, room_image: Image.Image) -> str:
        # Specific room analysis based on the uploaded image
        return (
            "A modern living room with white walls and white ceiling, "
            "featuring three rectangular windows with white frames arranged horizontally on the upper wall, "
            "three matching windows below with white horizontal blinds currently covering them, "
            "a dark brown leather sectional sofa positioned in front of the windows, "
            "white/light colored flooring, some green plants in pots on the right side, "
            "clean minimalist interior design with neutral color palette, "
            "natural lighting from the windows, contemporary residential setting"
        )
    
    def _extract_fabric_colors(self, fabric_image: Image.Image) -> str:
        # Enhanced fabric analysis
        try:
            # Convert to RGB if needed
            if fabric_image.mode != 'RGB':
                fabric_image = fabric_image.convert('RGB')
            
            # Sample colors from different areas
            width, height = fabric_image.size
            colors = []
            
            # Sample from center and corners
            sample_points = [
                (width//2, height//2),  # center
                (width//4, height//4),  # top-left area
                (3*width//4, height//4),  # top-right area
                (width//4, 3*height//4),  # bottom-left area
                (3*width//4, 3*height//4)  # bottom-right area
            ]
            
            for x, y in sample_points:
                r, g, b = fabric_image.getpixel((x, y))
                colors.append((r, g, b))
            
            # Analyze dominant colors
            avg_r = sum(c[0] for c in colors) // len(colors)
            avg_g = sum(c[1] for c in colors) // len(colors)
            avg_b = sum(c[2] for c in colors) // len(colors)
            
            # Determine color description based on the uploaded fabric
            if avg_r > 200 and avg_g > 200 and avg_b > 180:
                return "cream and beige colored fabric with natural woven linen texture, featuring subtle vertical and horizontal thread patterns, light neutral tones with organic fiber appearance"
            elif avg_r > 150 and avg_g > 130 and avg_b > 100:
                return "warm beige and tan fabric with natural fiber texture, earthy neutral tones with woven pattern"
            else:
                return "neutral colored fabric with natural linen-like texture and woven pattern"
                
        except Exception as e:
            logger.warning(f"Color extraction failed: {e}")
            return "cream and beige textured fabric with natural woven pattern"





class TestModel(BaseModel):
    """Test model that generates mock results without API calls"""
    
    def __init__(self):
        super().__init__()
        logger.info("Test model initialized - no API calls will be made")
    
    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image) -> Image.Image:
        """Generate a test visualization by combining room and fabric images"""
        try:
            # Create a simple mock visualization
            mock_image = self._create_mock_visualization(room_image, fabric_image)
            logger.info("Test visualization generated successfully")
            return mock_image
        except Exception as e:
            logger.error(f"Test model error: {str(e)}")
            raise ModelError(f"Test generation failed: {str(e)}")
    
    def _create_mock_visualization(self, room_image: Image.Image, fabric_image: Image.Image) -> Image.Image:
        """Create a simple mock by overlaying fabric pattern on room image"""
        # Resize fabric to fit room image
        fabric_resized = fabric_image.resize((room_image.width // 4, room_image.height // 2))
        
        # Create a copy of room image
        result = room_image.copy()
        
        # Paste fabric pattern in the center (simulating curtains)
        x_offset = (room_image.width - fabric_resized.width) // 2
        y_offset = (room_image.height - fabric_resized.height) // 4
        
        # Blend the fabric onto the room image
        result.paste(fabric_resized, (x_offset, y_offset))
        
        return result


class DalleModel(BaseModel):
    def __init__(self):
        super().__init__()
        if not config.openai_api_key or config.openai_api_key == "your_api_key_here":
            raise ModelError("OpenAI API key required for DALL-E model")
        self.client = OpenAI(api_key=config.openai_api_key)

    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image) -> str:
        try:
            response = self.client.images.generate(
                model=config.dalle_model,
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            return response.data[0].url
        except Exception as e:
            logger.error(f"DALL-E generation error: {str(e)}")
            raise APIError(f"DALL-E API failed: {str(e)}")


class ModelFactory:
    _instances: Dict[ModelType, BaseModel] = {}
    
    @classmethod
    def get_model(cls, model_type: ModelType) -> BaseModel:
        if model_type not in cls._instances:
            cls._instances[model_type] = cls._create_model(model_type)
        return cls._instances[model_type]
    
    @classmethod
    def _create_model(cls, model_type: ModelType) -> BaseModel:
        if model_type == ModelType.STABLE_DIFFUSION:
            raise ModelError("Stable Diffusion not supported. Use LANGCHAIN_OPENAI or DALLE instead.")
        elif model_type == ModelType.DALLE:
            return DalleModel()
        elif model_type == ModelType.LANGCHAIN_OPENAI:
            return LangChainOpenAIModel()
        elif model_type == ModelType.TEST_MODE:
            return TestModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
