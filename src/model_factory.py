from abc import ABC, abstractmethod
from PIL import Image
import asyncio
import os
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
    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> Image.Image:
        pass
    
    def _get_cache_key(self, prompt: str, room_hash: str, fabric_hash: str) -> str:
        return f"{prompt[:50]}_{room_hash}_{fabric_hash}"


class LangChainOpenAIModel(BaseModel):
    def __init__(self):
        super().__init__()
        if not config.openai_api_key or config.openai_api_key == "your_api_key_here":
            raise ModelError("OpenAI API key required for LangChain model")
        self.client = OpenAI(api_key=config.openai_api_key)
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

    def get_style_prompt(self, style: str) -> str:
        """Return appropriate prompt based on treatment style (curtains or blinds)"""
        from .config import CurtainStyle, BlindsStyle
        
          # Curtain style prompts
        curtain_prompts = {
            CurtainStyle.CLOSED.value: """Transform the first image (room) by installing floor-to-ceiling curtains mounted at the ceiling line, hanging straight down in graceful folds, covering all windows completely. 
                        CRITICAL: The curtain fabric MUST use the EXACT pattern, texture, colors, and design from the second image (fabric photo). 
                        Tile the fabric pattern seamlessly across the entire curtain surface, maintaining the original scale and details of the pattern. 
                        Curtains must extend from ceiling to floor with no gap at the top.""",

            CurtainStyle.HALF_OPEN.value: """Transform the first image (room) by installing floor-to-ceiling curtains mounted at the ceiling line, parted in the middle with panels gathered to both sides. 
                           CRITICAL: The curtain fabric MUST use the EXACT pattern, texture, colors, and design from the second image (fabric photo). 
                           Tile the fabric pattern seamlessly across the entire curtain surface, maintaining the original scale and details of the pattern. 
                           Curtains must extend from ceiling to floor with no gap at the top, creating a symmetrical opening in the center.""",

            CurtainStyle.WITH_SHEERS.value: """Transform the first image (room) by installing a double-layer curtain system mounted at the ceiling line: 
                             LAYER 1 (Inner): Install sheer white semi-transparent curtains closest to the window, visible through the opening between the main curtains. 
                             LAYER 2 (Outer): Install main curtains parted in the middle and gathered to both sides. 
                             CRITICAL FABRIC PATTERN: The main outer curtain fabric MUST use the EXACT pattern, texture, colors, and design from the second image (fabric photo). 
                             TILE the fabric pattern SEAMLESSLY and REPEATEDLY across the ENTIRE curtain surface, maintaining the original scale and details of the pattern. 
                             The pattern should repeat multiple times vertically to cover the full curtain height like continuous fabric or wallpaper. 
                             VISIBILITY: The white sheer curtains MUST be clearly visible in the center opening between the parted main curtains, creating a layered effect. 
                             Both layers must extend from ceiling to floor with no gap at the top. Show the sheers as a distinct white layer behind the patterned main curtains."""
        }
        
        # Blinds style prompts - ENHANCED for seamless pattern tiling
        blinds_prompts = {
            BlindsStyle.ROLLER.value: """Transform the first image (room) by installing sleek roller blinds that fit EXACTLY within each window frame. 
                        CRITICAL FABRIC PATTERN: The blind material MUST use the EXACT pattern, texture, colors, and design from the second image (fabric photo). 
                        TILE the fabric pattern SEAMLESSLY and REPEATEDLY across the ENTIRE blind surface from top to bottom and left to right, maintaining the original scale and details. 
                        The pattern should repeat multiple times to cover the full blind area like wallpaper or continuous fabric. 
                        INSTALLATION: 1) Detect ALL windows precisely 2) Install roller blinds INSIDE each window recess 3) Each window gets separate blind fitted to exact dimensions 
                        4) Show blinds 70% lowered 5) Show roller tube mechanism at top 6) Maintain room layout unchanged 7) Realistic shadows showing inside mount.""",

            BlindsStyle.VENETIAN.value: """Transform the first image (room) by installing classic venetian blinds with horizontal slats that fit EXACTLY within each window frame. 
                           CRITICAL FABRIC PATTERN: Each horizontal slat MUST use the EXACT colors and pattern from the second image (fabric photo). 
                           TILE the pattern SEAMLESSLY across ALL slats, creating a continuous repeating pattern from top to bottom of the blind. 
                           The pattern should flow across multiple slats like a tiled surface or continuous fabric. 
                           INSTALLATION: 1) Detect ALL windows precisely 2) Install venetian blinds INSIDE each window recess 3) Each window gets separate blind 
                           4) Show horizontal slats in partially open position 5) Show tilt mechanism 6) Maintain room layout unchanged 7) Realistic shadows showing inside mount.""",

            BlindsStyle.VERTICAL.value: """Transform the first image (room) by installing modern vertical blinds that fit EXACTLY within each window frame. 
                             CRITICAL FABRIC PATTERN: Each vertical slat MUST use the EXACT pattern, texture, and colors from the second image (fabric photo). 
                             TILE the fabric pattern SEAMLESSLY and REPEATEDLY across ALL vertical slats from left to right, maintaining the original scale. 
                             The pattern should repeat multiple times across the width like a continuous fabric surface or wallpaper. 
                             INSTALLATION: 1) Detect ALL windows precisely 2) Install vertical blinds INSIDE each window recess 3) Each window gets separate blind 
                             4) Show vertical slats partially open 5) Show track system at top 6) Maintain room layout unchanged 7) Realistic shadows showing inside mount.""",

            BlindsStyle.ROMAN.value: """Transform the first image (room) by installing elegant roman blinds with soft fabric folds that fit EXACTLY within each window frame. 
                           CRITICAL FABRIC PATTERN: The blind fabric MUST use the EXACT pattern, texture, colors, and design from the second image (fabric photo). 
                           TILE the fabric pattern SEAMLESSLY and REPEATEDLY across the ENTIRE blind surface, maintaining the original scale and details. 
                           The pattern should repeat multiple times vertically and horizontally to cover the full blind area like continuous fabric or wallpaper. 
                           INSTALLATION: 1) Detect ALL windows precisely 2) Install roman blinds INSIDE each window recess 3) Each window gets separate blind 
                           4) Show roman blinds 70% lowered with visible horizontal folds 5) Show fabric fold mechanism 6) Maintain room layout unchanged 7) Realistic shadows showing inside mount."""
        }
        
        # Try curtain prompts first, then blinds prompts
        return curtain_prompts.get(style) or blinds_prompts.get(style) or curtain_prompts[CurtainStyle.HALF_OPEN.value]

    def get_style_reference_image(self, style: str, config) -> bytes:
        """Return reference image bytes based on curtain style"""
        from .config import CurtainStyle
        
        style_images = {
            CurtainStyle.CLOSED.value: config.Config.curtain_closed_image_path,
            CurtainStyle.HALF_OPEN.value: config.Config.curtain_half_open_image_path,
            CurtainStyle.WITH_SHEERS.value: config.Config.curtain_sheers_image_path
        }

        image_path = style_images.get(style)
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                return f.read()
        return None

    def _create_curtain_composite(self, room_image: Image.Image, fabric_image: Image.Image, style: str) -> Image.Image:
        """Create composite image with fabric pattern tiled on curtain areas"""
        from PIL import ImageDraw
        
        # Create base composite
        composite = room_image.copy()
        width, height = composite.size
        
        # Tile fabric to cover full image
        fabric_width, fabric_height = fabric_image.size
        tiles_x = (width // fabric_width) + 2
        tiles_y = (height // fabric_height) + 2
        
        fabric_tiled = Image.new('RGB', (fabric_width * tiles_x, fabric_height * tiles_y))
        for x in range(tiles_x):
            for y in range(tiles_y):
                fabric_tiled.paste(fabric_image, (x * fabric_width, y * fabric_height))
        
        fabric_tiled = fabric_tiled.crop((0, 0, width, height))
        
        # Blend fabric onto curtain areas (left and right thirds for half-open)
        if 'half' in style or 'sheers' in style:
            # Left curtain panel
            left_panel = fabric_tiled.crop((0, 0, width // 3, height))
            composite.paste(left_panel, (0, 0))
            # Right curtain panel
            right_panel = fabric_tiled.crop((2 * width // 3, 0, width, height))
            composite.paste(right_panel, (2 * width // 3, 0))
        else:
            # Full coverage for closed curtains
            composite = Image.blend(composite, fabric_tiled, 0.6)
        
        return composite
    
    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> str:
        try:
            from io import BytesIO
            import requests
            import time
            import hashlib
            
            # Convert to PNG bytes
            room_bytes = BytesIO()
            room_image.save(room_bytes, format='PNG')
            room_bytes.seek(0)
            
            fabric_bytes = BytesIO()
            fabric_image.save(fabric_bytes, format='PNG')
            fabric_bytes.seek(0)
            
            # Generate unique hash from fabric image to force fresh generation
            fabric_hash = hashlib.md5(fabric_bytes.getvalue()).hexdigest()[:8]
            timestamp = int(time.time())
            
            # Get style-specific prompt (supports both curtains and blinds) with fabric details and unique identifiers
            style_prompt = self.get_style_prompt(curtain_style)
            enhanced_prompt = f"{style_prompt} FABRIC_ID:{fabric_hash} TIMESTAMP:{timestamp} "
            
            # Reset BytesIO positions after hashing
            room_bytes.seek(0)
            fabric_bytes.seek(0)
            
            
            # Prepare multipart form data with image[] array format
            files = [
                ('image[]', ('room.png', room_bytes, 'image/png')),
                ('image[]', ('fabric.png', fabric_bytes, 'image/png'))
            ]
            
            data = {
                'model': 'gpt-image-1',
                'prompt': enhanced_prompt,
                'n': 1,
                'size': '1024x1024'
            }
            
            headers = {
                'Authorization': f'Bearer {config.openai_api_key}'
            }
            
            response = await asyncio.to_thread(
                requests.post,
                'https://api.openai.com/v1/images/edits',
                files=files,
                data=data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"API Response: {result}")
                if 'data' in result and len(result['data']) > 0:
                    if 'url' in result['data'][0]:
                        return result['data'][0]['url']
                    elif 'b64_json' in result['data'][0]:
                        # Return base64 as data URL
                        b64_data = result['data'][0]['b64_json']
                        return f"data:image/png;base64,{b64_data}"
                return str(result)
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                raise ModelError(f"API error {response.status_code}: {response.text}")
            
        except Exception as e:
            logger.error(f"OpenAI image edit error: {str(e)}")
            raise ModelError(f"Failed to edit image: {str(e)}")
    
    
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        import base64
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()




class ReplicateModel(BaseModel):
    def __init__(self):
        super().__init__()
        if not config.replicate_api_token:
            raise ModelError("Replicate API token required")
        import replicate
        import ssl
        import urllib3
        # Disable SSL warnings and verification for corporate networks
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        ssl._create_default_https_context = ssl._create_unverified_context
        self.client = replicate.Client(api_token=config.replicate_api_token)
    
    def get_curtain_style_prompt(self, style: str) -> str:
        """Return appropriate prompt based on curtain style"""
        from .config import CurtainStyle
        style_prompts = {
            CurtainStyle.CLOSED.value: """Transform the room by installing one continuous curtain track mounted at the ceiling line, spanning the entire wall width above all windows. 
                        Install floor-to-ceiling curtains that hang straight down in graceful folds, covering all windows completely from ceiling to floor with no gap at the top. 
                        The curtain fabric should feature the SMALL REPEATING PATTERN from the second image, tiled seamlessly 
                        across the entire curtain surface - the pattern should repeat many times in a regular grid, 
                        maintaining the original small scale of the pattern elements. CRITICAL: Curtains must be mounted at the ceiling line and extend fully from ceiling to floor as one unified treatment across the entire wall.""",
            CurtainStyle.HALF_OPEN.value: """Transform the room by installing one continuous curtain track mounted at the ceiling line, spanning the entire wall width above all windows. 
                           Install floor-to-ceiling curtains that are parted in the middle, with panels elegantly gathered and pulled to both left and right sides, extending from ceiling to floor with no gap at the top. 
                           The curtain fabric should feature the SMALL REPEATING PATTERN from the second image, tiled seamlessly across the entire curtain surface - 
                           the pattern should repeat many times in a regular grid, maintaining the original small scale of the pattern elements. 
                           The curtains should create a symmetrical opening in the center, allowing natural light through the middle while framing the windows with fabric on both sides. 
                           CRITICAL: This should be ONE continuous curtain treatment mounted at the ceiling line with panels drawn to both sides, extending fully from ceiling to floor, not separate curtains for each window.""",
            CurtainStyle.WITH_SHEERS.value: """Transform the room by installing a double-track curtain system mounted at the ceiling line with sheer white curtains 
                             closest to the window and main curtains on the outer track. The main curtains should feature 
                             the SMALL REPEATING PATTERN from the second image, tiled seamlessly across the curtain surface - 
                             the pattern should repeat many times in a regular grid, maintaining the original small scale. 
                             CRITICAL: Both layers must be mounted at the ceiling line and extend fully from ceiling to floor with no gap at the top."""
        }
        return style_prompts.get(style, style_prompts[CurtainStyle.HALF_OPEN.value])
    
    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> str:
        try:
            room_url = await self._upload_image(room_image)
            fabric_url = await self._upload_image(fabric_image)
            
            # Get style-specific prompt
            style_prompt = self.get_curtain_style_prompt(curtain_style)
            
            output = await asyncio.to_thread(
                self.client.run,
                "google/nano-banana",
                input={
                    "prompt": style_prompt,
                    "image_input": [room_url, fabric_url]
                }
            )
            # Handle FileOutput object or list
            if hasattr(output, 'url'):
                return output.url
            elif isinstance(output, list) and len(output) > 0:
                return output[0].url if hasattr(output[0], 'url') else output[0]
            return output
        except Exception as e:
            logger.error(f"Replicate error: {e}")
            raise ModelError(f"Replicate generation failed: {e}")
    
    async def _upload_image(self, image: Image.Image) -> str:
        import base64
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    
    def _analyze_fabric_colors(self, fabric_image: Image.Image) -> str:
        """Analyze fabric colors for prompt enhancement"""
        try:
            width, height = fabric_image.size
            colors = []
            for x in range(0, width, width//5):
                for y in range(0, height, height//5):
                    if x < width and y < height:
                        colors.append(fabric_image.getpixel((x, y)))
            
            avg_r = sum(c[0] for c in colors) // len(colors)
            avg_g = sum(c[1] for c in colors) // len(colors)
            avg_b = sum(c[2] for c in colors) // len(colors)
            
            if avg_r > 200 and avg_g > 200 and avg_b > 180:
                return "cream and beige colored"
            elif avg_r > avg_g and avg_r > avg_b:
                return "warm red and burgundy"
            elif avg_g > avg_r and avg_g > avg_b:
                return "green and sage"
            elif avg_b > avg_r and avg_b > avg_g:
                return "blue and navy"
            else:
                return "neutral colored"
        except:
            return "textured fabric"


class TestModel(BaseModel):
    """Test model that generates mock results without API calls"""
    
    def __init__(self):
        super().__init__()
        logger.info("Test model initialized - no API calls will be made")
    
    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> Image.Image:
        """Generate a test visualization by combining room and fabric images"""
        try:
            # Create a simple mock visualization with style consideration
            mock_image = self._create_mock_visualization(room_image, fabric_image, curtain_style)
            logger.info(f"Test visualization generated successfully with style: {curtain_style}")
            return mock_image
        except Exception as e:
            logger.error(f"Test model error: {str(e)}")
            raise ModelError(f"Test generation failed: {str(e)}")
    
    def _create_mock_visualization(self, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> Image.Image:
        """Create curtain visualization by intelligently compositing fabric onto window areas"""
        from PIL import ImageDraw, ImageFilter
        
        result = room_image.copy()
        width, height = result.size
        
        # Create curtain overlays for window areas
        # Detect potential window areas (typically upper portion of image)
        window_height = height // 2
        curtain_width = width // 3
        
        # Create fabric pattern for curtains
        fabric_resized = fabric_image.resize((curtain_width, window_height))
        
        # Add transparency for realistic curtain effect
        fabric_with_alpha = fabric_resized.convert('RGBA')
        
        # Create curtain overlay with some transparency
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Place curtains at window positions (left, center, right)
        positions = [0, width//3, 2*width//3]
        
        for x_pos in positions:
            if x_pos + curtain_width <= width:
                # Add slight transparency and blend
                curtain = fabric_with_alpha.copy()
                alpha = curtain.split()[-1]
                alpha = alpha.point(lambda p: int(p * 0.8))  # 80% opacity
                curtain.putalpha(alpha)
                
                overlay.paste(curtain, (x_pos, 0), curtain)
        
        # Composite the curtains onto the room
        result = result.convert('RGBA')
        result = Image.alpha_composite(result, overlay)
        
        return result.convert('RGB')


class DalleModel(BaseModel):
    def __init__(self):
        super().__init__()
        if not config.openai_api_key or config.openai_api_key == "your_api_key_here":
            raise ModelError("OpenAI API key required for DALL-E model")
        self.client = OpenAI(api_key=config.openai_api_key)

    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> str:
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
    def clear_cache(cls):
        cls._instances.clear()
    
    @classmethod
    def get_model(cls, model_type: ModelType) -> BaseModel:
        if model_type not in cls._instances:
            cls._instances[model_type] = cls._create_model(model_type)
        return cls._instances[model_type]
    
    @classmethod
    def _create_model(cls, model_type: ModelType) -> BaseModel:
        if model_type == ModelType.STABLE_DIFFUSION:
            raise ModelError("Stable Diffusion not supported. Use REPLICATE_CONTROLNET or DALLE instead.")
        elif model_type == ModelType.DALLE:
            return DalleModel()
        elif model_type == ModelType.LANGCHAIN_OPENAI:
            return LangChainOpenAIModel()
        elif model_type in [ModelType.REPLICATE_CONTROLNET, ModelType.REPLICATE_REALISTIC]:
            return ReplicateModel()
        elif model_type == ModelType.REALISTIC_FOLD:
            from .realistic_curtain_model import RealisticCurtainModel
            return RealisticCurtainModel(config.openai_api_key)
        elif model_type == ModelType.TEST_MODE:
            return TestModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
