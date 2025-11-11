from PIL import Image
import io
import asyncio
import hashlib
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple
from loguru import logger
from .config import config, CurtainStyle, TreatmentType, BlindsStyle
from .exceptions import ImageValidationError, APIError, ImageProcessingError, ModelError
from .model_factory import ModelFactory
from .gallery_manager import GalleryManager


class ImageProcessor:
    def __init__(self):
        self.model = ModelFactory.get_model(config.effective_model_type)
        self.base_output_dir = Path("generated_images")
        self.base_output_dir.mkdir(exist_ok=True)
        self.gallery_manager = GalleryManager()
        logger.info(f"ImageProcessor initialized with {config.effective_model_type.value} model")

    def validate_image(self, image_file) -> None:
        """Validate uploaded image file"""
        if not image_file:
            raise ImageValidationError("No image file provided", "MISSING_FILE")
        
        # Validate file type only
        file_extension = image_file.name.split('.')[-1].lower() if hasattr(image_file, 'name') else ''
        if file_extension not in config.allowed_image_types:
            raise ImageValidationError(
                f"Unsupported file type: {file_extension}. Allowed: {config.allowed_image_types}",
                "INVALID_FILE_TYPE"
            )

    async def process_images(self, room_photo, fabric_photo, user_phone=None, treatment_type: TreatmentType = TreatmentType.CURTAINS, curtain_style: CurtainStyle = None, blinds_style: BlindsStyle = None) -> Union[Image.Image, str]:
        """Process room and fabric photos asynchronously"""
        try:
            # Validate images
            self.validate_image(room_photo)
            self.validate_image(fabric_photo)

            # Convert and optimize images
            room_image = self._optimize_image(Image.open(room_photo).convert("RGB"))
            fabric_image = self._optimize_image(Image.open(fabric_photo).convert("RGB"))
            
            logger.info(f"Processing optimized images: room={room_image.size}, fabric={fabric_image.size}")

            result, saved_path = await self.generate_visualization(room_image, fabric_image, user_phone, treatment_type, curtain_style, blinds_style, room_photo.name, fabric_photo.name)
            
            # Add to gallery after successful generation with treatment type
            self.gallery_manager.add_to_gallery(room_photo, fabric_photo, saved_path, user_phone, treatment_type.value)
            
            return result, saved_path

        except (ImageValidationError, APIError, ModelError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in image processing: {str(e)}")
            raise ImageProcessingError(f"Error processing images: {str(e)}", "PROCESSING_FAILED")

    async def generate_visualization(self, room_image: Image.Image, fabric_image: Image.Image, user_phone=None, treatment_type: TreatmentType = TreatmentType.CURTAINS, curtain_style: CurtainStyle = None, blinds_style: BlindsStyle = None, room_name="room.jpg", fabric_name="fabric.jpg") -> Union[Image.Image, str]:
        """Generate window treatment visualization using selected model with retry logic"""
        for attempt in range(config.max_retries):
            try:
            
                
                # Generate enhanced prompt based on treatment type
                if treatment_type == TreatmentType.BLINDS:
                    prompt = self.generate_blinds_prompt(blinds_style)
                else:
                    prompt = self.generate_curtains_prompt(curtain_style)
                
                logger.info(f"Generating {treatment_type.value} visualization (attempt {attempt + 1}/{config.max_retries})")
                
                # Generate image using the selected model
                style_value = (blinds_style.value if treatment_type == TreatmentType.BLINDS else curtain_style.value) if (blinds_style or curtain_style) else None
                result = await self.model.generate_image(prompt, room_image, fabric_image, curtain_style=style_value)
                
                # Save result to filesystem with user phone
                saved_path = await self._save_result(result, user_phone, treatment_type.value)
                
                logger.success(f"{treatment_type.value.title()} visualization generated and saved to {saved_path}")
                return result, saved_path

            except (APIError, ModelError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == config.max_retries - 1:
                    raise APIError(f"Failed after {config.max_retries} attempts: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error in visualization generation: {str(e)}")
                raise APIError(f"Error generating visualization: {str(e)}")


    def generate_curtains_prompt(self, curtain_style: CurtainStyle = None) -> str:
        """Generate enhanced prompt for curtain transformation"""
        style = curtain_style or config.default_curtain_style
        style_desc = config.curtain_style_descriptions[style]
        
        return (
            f"Transform this interior room by replacing any existing window treatments (blinds, shades, or bare windows) with {style_desc}. "
            f"CRITICAL: Install curtain rods at the ceiling line and hang curtains that extend fully from the ceiling all the way down to the floor with no gap at the top. "
            f"Maintain the exact same room layout, furniture placement, wall colors, and architectural features. "
            f"The curtains should be styled as {style.value.replace('_', ' ')}, mounted at ceiling height and hanging down to the floor, properly fitted to all windows in the space. "
            f"Show realistic fabric draping, natural folds, and proper proportions with curtains covering the full height from ceiling to floor. "
            f"Professional interior design photography with natural lighting and shadows, high resolution."
        )
    
    def generate_blinds_prompt(self, blinds_style: BlindsStyle = None) -> str:
        """Generate enhanced prompt for blinds transformation using GPT-4 Vision"""
        style = blinds_style or config.default_blinds_style
        style_desc = config.blinds_style_descriptions[style]
        
        return (
            f"Transform this interior room by installing {style_desc} that fit EXACTLY to each window frame. "
            f"CRITICAL REQUIREMENTS: "
            f"1. Detect ALL windows in the room precisely "
            f"2. Install {style.value} blinds that fit EXACTLY within each window frame - width and height must match the window opening perfectly "
            f"3. Blinds must be mounted INSIDE the window recess, not outside "
            f"4. Each window gets its own separate blind fitted to its exact dimensions "
            f"5. FABRIC PATTERN: Use the fabric pattern from the second image and TILE IT SEAMLESSLY across the entire blind surface. "
            f"The pattern must REPEAT uniformly and consistently across all blinds with perfect alignment. "
            f"Create a seamless, repeating pattern with no distortion or randomness. "
            f"6. Maintain the exact same room layout, furniture, walls, floor, and all other elements unchanged "
            f"7. Show blinds in a partially lowered position (70% down) to display the tiled fabric pattern clearly "
            f"8. Ensure realistic shadows and depth showing blinds are inside the window frame "
            f"9. For {style.value} blinds: show appropriate mechanism (roller tube, slats, or fabric folds) "
            f"Professional interior design photography, photorealistic, high resolution, natural lighting."
        )
    
    def _optimize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Optimize image size and quality for model processing"""
        # Get original dimensions
        width, height = image.size
        
        # Calculate new dimensions if image is too large
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
            
            # Resize with high-quality resampling
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        
        return image
    
    async def _save_result(self, result: Union[Image.Image, str], user_phone: str = None, treatment_type: str = "curtain") -> str:
        """Save generated result to filesystem organized by user phone"""
        # Create user-specific directory
        if user_phone:
            clean_phone = user_phone.replace('+', '').replace('-', '').replace(' ', '')
            user_dir = self.base_output_dir / clean_phone
            user_dir.mkdir(exist_ok=True)
        else:
            user_dir = self.base_output_dir
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{treatment_type}_visualization_{timestamp}.png"
        filepath = user_dir / filename
        
        if isinstance(result, str):  # URL from API
            if result.startswith('data:image/'):
                # Handle data URL (base64)
                import base64
                header, data = result.split(',', 1)
                image_data = base64.b64decode(data)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
            else:
                # Handle regular URL
                response = requests.get(result)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
        elif hasattr(result, 'url'):  # Replicate FileOutput object
            # Download from FileOutput URL
            response = requests.get(result.url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
        else:  # PIL Image
            result.save(filepath, 'PNG')
        
        logger.info(f"Image saved to {filepath}")
        return str(filepath)
    
    def _generate_image_hash(self, image: Image.Image) -> str:
        """Generate hash for image caching"""
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        return hashlib.md5(image_bytes.getvalue()).hexdigest()[:16]
