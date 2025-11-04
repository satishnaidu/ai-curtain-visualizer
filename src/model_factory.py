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

    def get_curtain_style_prompt(self, style: str) -> str:
        """Return appropriate prompt based on curtain style"""
        from .config import CurtainStyle
        
        style_prompts = {
            CurtainStyle.CLOSED.value: """Transform the room by installing curtain tracks on the ceiling and hanging full-height pleated curtains 
                        with deep, elegant folds that extend from the ceiling track to the floor. The curtains should have 
                        consistent, evenly-spaced pleats creating a luxurious, formal appearance. Use the fabric pattern from 
                        the second image and ensure the pleats are well-defined and uniform with style referenced in third image""",

            CurtainStyle.HALF_OPEN.value: """Transform the room by installing curtain tracks on the ceiling and creating a dramatic swept-aside 
                           curtain arrangement. The curtains should be elegantly pulled to the sides of each window, creating 
                           graceful sweeping folds while still extending from ceiling to floor. Use the fabric pattern from the 
                           second image and ensure the side-swept arrangement looks natural and sophisticated with style referenced in third image""",

            CurtainStyle.WITH_SHEERS.value: """Transform the room by installing a double-track curtain system on the ceiling with both sheer 
                             and regular curtains. The sheer white curtains should be mounted closest to the window, with the 
                             main curtains using the provided fabric pattern mounted on the outer track. Both layers should 
                             extend from ceiling to floor, creating an elegant layered effect with style referenced in third image""",
                             
            CurtainStyle.PLEATED.value: """Transform the room by installing formal pleated curtains with structured folds 
                        using the fabric pattern from the second image.""",
                        
            CurtainStyle.VALANCE.value: """Transform the room by installing curtains with a decorative valance at the top 
                         using the fabric pattern from the second image."""
        }

        return style_prompts.get(style, style_prompts[CurtainStyle.HALF_OPEN.value])

    def get_style_reference_image(self, style: str, config) -> bytes:
        """Return reference image bytes based on curtain style"""
        from .config import CurtainStyle
        
        style_images = {
            CurtainStyle.CLOSED.value: config.Config.curtain_closed_image_path,
            CurtainStyle.HALF_OPEN.value: config.Config.curtain_half_open_image_path,
            CurtainStyle.WITH_SHEERS.value: config.Config.curtain_sheers_image_path,
            CurtainStyle.PLEATED.value: config.Config.curtain_pleated_image_path,
            CurtainStyle.VALANCE.value: config.Config.curtain_valance_image_path
        }

        image_path = style_images.get(style)
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                return f.read()
        return None

    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> str:
        try:
            # Convert both images to bytes for OpenAI
            from io import BytesIO
            import requests
            
            room_bytes = BytesIO()
            room_image.save(room_bytes, format='PNG')
            room_bytes.seek(0)
            
            fabric_bytes = BytesIO()
            fabric_image.save(fabric_bytes, format='PNG')
            fabric_bytes.seek(0)
            
            # Use OpenAI image edit API with multiple images
            files = [
                ('image[]', ('room.png', room_bytes, 'image/png')),
                ('image[]', ('fabric.png', fabric_bytes, 'image/png'))
            ]
            # Add style reference image if available
            style_bytes = self.get_style_reference_image(curtain_style, config)
            if style_bytes:
                files.append(
                    ('image[]', (f'style_{curtain_style}.png', style_bytes, 'image/png'))
                )

            # Get appropriate prompt for the selected style
            style_prompt = self.get_curtain_style_prompt(curtain_style)

            data = {
                'model': 'gpt-image-1',
                'prompt': style_prompt,
                'n': 1,
                'size': '1024x1024'
            }

            headers = {
                'Authorization': f'Bearer {config.openai_api_key}'
            }
            
            response = requests.post(
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
                raise Exception(f"API error: {response.text}")
            
        except Exception as e:
            logger.error(f"OpenAI image edit error: {str(e)}")
            if "'url'" in str(e):
                raise ModelError(f"Response format error - check API response structure")
            raise ModelError(f"Failed to edit image: {str(e)}")
    
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
        # Generic room analysis for any interior space
        try:
            width, height = room_image.size
            
            # Sample room to determine general characteristics
            center_color = room_image.getpixel((width//2, height//2))
            avg_brightness = sum(center_color) // 3
            
            if avg_brightness > 200:
                room_desc = "A bright, airy interior space with light-colored walls and good natural lighting"
            elif avg_brightness > 120:
                room_desc = "A well-lit interior room with comfortable ambient lighting and medium-toned decor"
            else:
                room_desc = "A cozy interior space with warm, intimate lighting and rich color tones"
            
            return f"{room_desc}, featuring windows that would benefit from elegant curtain treatments, maintaining the existing furniture layout and architectural elements"
            
        except Exception:
            return "An interior room with windows suitable for curtain installation, preserving the existing decor and layout"
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        import base64
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    
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
    
    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> str:
        try:
            room_url = await self._upload_image(room_image)
            fabric_url = await self._upload_image(fabric_image)
            
            # Create a composite prompt with fabric description
            fabric_colors = self._analyze_fabric_colors(fabric_image)
            enhanced_prompt = f"Transform this room by adding elegant floor-length curtains with {fabric_colors} fabric to all windows. {prompt}"
            
            output = await asyncio.to_thread(
                self.client.run,
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={
                    "prompt": f"Cover all the windows with {fabric_colors} with long curtains, keep room identical",
                    "image": room_url,
                    "strength": 0.3,
                    "num_inference_steps": 15,
                    "guidance_scale": 6
                }
            )
            return output[0] if output else None
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
        elif model_type == ModelType.TEST_MODE:
            return TestModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
