import replicate
import asyncio
from PIL import Image
import requests
from io import BytesIO
import base64
from loguru import logger
from .exceptions import ModelError

class ReplicateModel:
    """Replicate API model for better image-to-image transformations"""
    
    def __init__(self, api_token: str):
        self.client = replicate.Client(api_token=api_token)
        
    async def generate_with_controlnet(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image) -> str:
        """Use ControlNet for structure-preserving transformations"""
        try:
            # Convert images to base64 or upload to temporary storage
            room_url = await self._upload_image(room_image)
            
            output = await asyncio.to_thread(
                self.client.run,
                "jagilley/controlnet-canny:aff48af9c68d162388d230a2ab003f68d2638d88307bdaf1c2f1ac95079c9613",
                input={
                    "image": room_url,
                    "prompt": prompt,
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
    
    async def generate_with_realistic_vision(self, prompt: str, room_image: Image.Image) -> str:
        """Use Realistic Vision for photorealistic results"""
        try:
            room_url = await self._upload_image(room_image)
            
            output = await asyncio.to_thread(
                self.client.run,
                "adirik/realistic-vision-v5:6348c96b81c5d5c1e0e4b6e4e7b5b5b5b5b5b5b5",
                input={
                    "image": room_url,
                    "prompt": prompt,
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