import replicate
import asyncio
from PIL import Image
import requests
from io import BytesIO
import base64
import ssl
from loguru import logger
from .exceptions import ModelError
from .config import CurtainStyle

class ReplicateModel:
    """Replicate API model for better image-to-image transformations"""
    
    def __init__(self, api_token: str):
        # Disable SSL verification for corporate proxy
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        self.client = replicate.Client(api_token=api_token)
        
        # Configure requests session to bypass SSL verification
        if hasattr(self.client, '_session'):
            self.client._session.verify = False
    
    def get_curtain_style_prompt(self, style: str) -> str:
        """Return appropriate prompt based on curtain style"""
        style_prompts = {
            CurtainStyle.CLOSED.value: "floor-to-ceiling curtains mounted at the ceiling line, hanging straight down in graceful folds to the floor, covering all windows completely from top to bottom",
            CurtainStyle.HALF_OPEN.value: "floor-to-ceiling curtains mounted at the ceiling line, elegantly parted in the middle with panels gathered and pulled to both sides, extending from ceiling to floor",
            CurtainStyle.WITH_SHEERS.value: "floor-to-ceiling double-layer curtains mounted at the ceiling line, with sheer white curtains closest to the window and main curtains on the outer track, both extending from ceiling to floor"
        }
        return style_prompts.get(style, style_prompts[CurtainStyle.HALF_OPEN.value])
        
    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> str:
        """Generate image with curtain style support"""
        return await self.generate_with_controlnet(prompt, room_image, fabric_image, curtain_style)
    
    async def generate_with_controlnet(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> str:
        """Use ControlNet for structure-preserving transformations"""
        try:
            # Convert images to base64
            room_url = await self._upload_image(room_image)
            fabric_url = await self._upload_image(fabric_image)
            
            # Enhance prompt with curtain style
            style_desc = self.get_curtain_style_prompt(curtain_style) if curtain_style else "elegant floor-to-ceiling curtains mounted at the ceiling line, extending all the way down to the floor"
            enhanced_prompt = f"{prompt} Replace window treatments with {style_desc} using the fabric pattern from the second image. CRITICAL: Curtains must be mounted at the ceiling line with no gap at the top, extending fully from ceiling to floor."
            
            output = await asyncio.to_thread(
                self.client.run,
                "jagilley/controlnet-canny:aff48af9c68d162388d230a2ab003f68d2638d88307bdaf1c2f1ac95079c9613",
                input={
                    "image": room_url,
                    "prompt": enhanced_prompt,
                    "num_samples": 1,
                    "image_resolution": "1024",
                    "strength": 0.7,
                    "guidance_scale": 7.5,
                    "seed": -1
                }
            )
            
            return output[0] if output else None
            
        except Exception as e:
            logger.error(f"Replicate ControlNet error: {e}")
            raise ModelError(f"ControlNet generation failed: {e}")
    
    async def generate_with_realistic_vision(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> str:
        """Use Realistic Vision for photorealistic results"""
        try:
            room_url = await self._upload_image(room_image)
            fabric_url = await self._upload_image(fabric_image)
            
            # Enhance prompt with curtain style
            style_desc = self.get_curtain_style_prompt(curtain_style) if curtain_style else "elegant floor-to-ceiling curtains mounted at the ceiling line, extending all the way down to the floor"
            enhanced_prompt = f"{prompt} Replace window treatments with {style_desc} using the fabric pattern provided. CRITICAL: Curtains must be mounted at the ceiling line with no gap at the top, extending fully from ceiling to floor."
            
            output = await asyncio.to_thread(
                self.client.run,
                "adirik/realistic-vision-v5:6348c96b81c5d5c1e0e4b6e4e7b5b5b5b5b5b5b5",
                input={
                    "image": room_url,
                    "prompt": enhanced_prompt,
                    "negative_prompt": "blurry, low quality, distorted, unrealistic",
                    "width": 1024,
                    "height": 1024,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 25,
                    "strength": 0.8
                }
            )
            
            return output[0] if output else None
            
        except Exception as e:
            logger.error(f"Replicate Realistic Vision error: {e}")
            raise ModelError(f"Realistic Vision generation failed: {e}")
    
    async def _upload_image(self, image: Image.Image) -> str:
        """Convert PIL image to base64 data URL for Replicate API"""
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"