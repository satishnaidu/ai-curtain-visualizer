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
from .config import config
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

    async def process_images(self, room_photo, fabric_photo, user_phone=None) -> Union[Image.Image, str]:
        """Process room and fabric photos asynchronously"""
        try:
            # Validate images
            self.validate_image(room_photo)
            self.validate_image(fabric_photo)

            # Convert and optimize images
            room_image = self._optimize_image(Image.open(room_photo).convert("RGB"))
            fabric_image = self._optimize_image(Image.open(fabric_photo).convert("RGB"))
            
            logger.info(f"Processing optimized images: room={room_image.size}, fabric={fabric_image.size}")

            result, saved_path = await self.generate_curtain_visualization(room_image, fabric_image, user_phone, room_photo.name, fabric_photo.name)
            
            # Add to gallery after successful generation
            self.gallery_manager.add_to_gallery(room_photo, fabric_photo, saved_path, user_phone)
            
            return result, saved_path

        except (ImageValidationError, APIError, ModelError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in image processing: {str(e)}")
            raise ImageProcessingError(f"Error processing images: {str(e)}", "PROCESSING_FAILED")

    async def generate_curtain_visualization(self, room_image: Image.Image, fabric_image: Image.Image, user_phone=None, room_name="room.jpg", fabric_name="fabric.jpg") -> Union[Image.Image, str]:
        """Generate curtain visualization using selected model with retry logic"""
        for attempt in range(config.max_retries):
            try:
                # Extract fabric characteristics
                fabric_analysis = self.analyze_fabric(fabric_image)
                room_analysis = self.analyze_room(room_image)
                
                # Generate enhanced prompt
                prompt = self.generate_enhanced_prompt(room_analysis, fabric_analysis)
                
                logger.info(f"Generating visualization (attempt {attempt + 1}/{config.max_retries})")
                
                # Generate image using the selected model
                result = await self.model.generate_image(prompt, room_image, fabric_image)
                
                # Save result to filesystem with user phone
                saved_path = await self._save_result(result, user_phone)
                
                # Add to gallery for public viewing (pass original photo objects)
                # Note: We need to pass the original photos from the calling function
                # This will be handled in the process_images method
                
                logger.success(f"Curtain visualization generated and saved to {saved_path}")
                return result, saved_path

            except (APIError, ModelError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == config.max_retries - 1:
                    raise APIError(f"Failed after {config.max_retries} attempts: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error in visualization generation: {str(e)}")
                raise APIError(f"Error generating curtain visualization: {str(e)}")

    def analyze_fabric(self, fabric_image: Image.Image) -> dict:
        """Analyze fabric characteristics"""
        # Enhanced fabric analysis
        colors = self.extract_fabric_colors(fabric_image)
        texture = self.analyze_texture(fabric_image)
        pattern = self.detect_pattern(fabric_image)
        
        return {
            "colors": colors,
            "texture": texture,
            "pattern": pattern
        }
    
    def analyze_room(self, room_image: Image.Image) -> dict:
        """Analyze room characteristics dynamically"""
        try:
            width, height = room_image.size
            
            # Sample room colors to determine scheme
            sample_points = [(width//4, height//4), (width//2, height//2), (3*width//4, 3*height//4)]
            room_colors = []
            for x, y in sample_points:
                r, g, b = room_image.getpixel((x, y))
                room_colors.append((r, g, b))
            
            avg_brightness = sum(sum(c) for c in room_colors) // (len(room_colors) * 3)
            
            # Determine room characteristics
            if avg_brightness > 200:
                color_scheme = "bright and airy with light colors"
                lighting = "well-lit with natural light"
            elif avg_brightness > 120:
                color_scheme = "warm and inviting with medium tones"
                lighting = "comfortable ambient lighting"
            else:
                color_scheme = "cozy and intimate with darker tones"
                lighting = "soft and moody lighting"
            
            return {
                "lighting": lighting,
                "style": "contemporary interior",
                "color_scheme": color_scheme,
                "window_treatment": "existing window coverings"
            }
        except Exception as e:
            logger.warning(f"Room analysis failed: {e}")
            return {
                "lighting": "natural lighting",
                "style": "modern interior",
                "color_scheme": "neutral color palette",
                "window_treatment": "current window treatments"
            }

    def extract_fabric_colors(self, fabric_image: Image.Image) -> str:
        """Extract dominant colors from fabric image using advanced color analysis"""
        try:
            # Sample multiple points for better color analysis
            width, height = fabric_image.size
            sample_points = [
                (width//4, height//4), (width//2, height//4), (3*width//4, height//4),
                (width//4, height//2), (width//2, height//2), (3*width//4, height//2),
                (width//4, 3*height//4), (width//2, 3*height//4), (3*width//4, 3*height//4)
            ]
            
            colors = []
            for x, y in sample_points:
                r, g, b = fabric_image.getpixel((x, y))
                colors.append((r, g, b))
            
            # Calculate average color
            avg_r = sum(c[0] for c in colors) // len(colors)
            avg_g = sum(c[1] for c in colors) // len(colors)
            avg_b = sum(c[2] for c in colors) // len(colors)
            
            # Determine color description based on RGB values
            if avg_r > avg_g and avg_r > avg_b:
                if avg_r > 180: return "warm red and pink tones"
                elif avg_r > 120: return "rich burgundy and wine colors"
                else: return "deep red and maroon shades"
            elif avg_g > avg_r and avg_g > avg_b:
                if avg_g > 180: return "fresh green and sage colors"
                elif avg_g > 120: return "forest and olive green tones"
                else: return "deep emerald and hunter green"
            elif avg_b > avg_r and avg_b > avg_g:
                if avg_b > 180: return "soft blue and sky tones"
                elif avg_b > 120: return "navy and royal blue colors"
                else: return "deep indigo and midnight blue"
            elif avg_r > 200 and avg_g > 200 and avg_b > 180:
                return "cream, beige and natural linen colors"
            elif avg_r > 150 and avg_g > 130 and avg_b > 100:
                return "warm brown, tan and earth tones"
            elif avg_r < 80 and avg_g < 80 and avg_b < 80:
                return "charcoal, black and dark gray tones"
            elif avg_r > 180 and avg_g > 180 and avg_b > 180:
                return "white, off-white and light neutral tones"
            else:
                return "mixed neutral and natural fabric colors"
                
        except Exception as e:
            logger.warning(f"Color extraction failed: {e}")
            return "natural fabric colors"
    
    def analyze_texture(self, fabric_image: Image.Image) -> str:
        """Analyze fabric texture dynamically"""
        try:
            # Convert to grayscale for texture analysis
            gray_fabric = fabric_image.convert('L')
            width, height = gray_fabric.size
            
            # Sample texture variation
            pixels = []
            for x in range(0, width, width//10):
                for y in range(0, height, height//10):
                    if x < width and y < height:
                        pixels.append(gray_fabric.getpixel((x, y)))
            
            # Calculate texture variation
            if len(pixels) > 1:
                avg_pixel = sum(pixels) // len(pixels)
                variation = sum(abs(p - avg_pixel) for p in pixels) // len(pixels)
                
                if variation > 30:
                    return "richly textured fabric with visible weave patterns"
                elif variation > 15:
                    return "subtly textured fabric with natural fiber appearance"
                else:
                    return "smooth fabric with fine texture"
            
            return "natural fabric texture"
        except Exception:
            return "quality fabric texture"
    
    def detect_pattern(self, fabric_image: Image.Image) -> str:
        """Detect fabric patterns dynamically"""
        try:
            # Sample different areas to detect pattern consistency
            width, height = fabric_image.size
            
            # Compare different quadrants for pattern detection
            quad1 = fabric_image.crop((0, 0, width//2, height//2))
            quad2 = fabric_image.crop((width//2, 0, width, height//2))
            
            # Simple pattern detection based on color variation
            q1_colors = quad1.getcolors(maxcolors=256)
            q2_colors = quad2.getcolors(maxcolors=256)
            
            if q1_colors and q2_colors:
                q1_variety = len(q1_colors)
                q2_variety = len(q2_colors)
                
                if abs(q1_variety - q2_variety) > 20:
                    return "distinctive patterns and designs"
                elif abs(q1_variety - q2_variety) > 10:
                    return "subtle patterns and texture variations"
                else:
                    return "solid color with natural fabric texture"
            
            return "natural fabric appearance"
        except Exception:
            return "classic fabric styling"

    def generate_enhanced_prompt(self, room_analysis: dict, fabric_analysis: dict) -> str:
        """Generate enhanced prompt for any room transformation"""
        return (
            f"Transform this interior room by replacing any existing window treatments (blinds, shades, or bare windows) with elegant floor-length curtains. "
            f"Maintain the exact same room layout, furniture placement, wall colors, and architectural features. "
            f"Keep the same {room_analysis['lighting']} and {room_analysis['color_scheme']}. "
            f"Add beautiful curtains made from fabric with {fabric_analysis['colors']} and {fabric_analysis['texture']} featuring {fabric_analysis['pattern']}. "
            f"The curtains should hang gracefully from ceiling to floor, properly fitted to all windows in the space. "
            f"Preserve the room's {room_analysis['style']} aesthetic while enhancing it with the new curtains. "
            f"Show realistic fabric draping, natural folds, and proper proportions. "
            f"Professional interior design photography with natural lighting and shadows, high resolution."
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
    
    async def _save_result(self, result: Union[Image.Image, str], user_phone: str = None) -> str:
        """Save generated result to filesystem organized by user phone"""
        # Create user-specific directory
        if user_phone:
            clean_phone = user_phone.replace('+', '').replace('-', '').replace(' ', '')
            user_dir = self.base_output_dir / clean_phone
            user_dir.mkdir(exist_ok=True)
        else:
            user_dir = self.base_output_dir
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"curtain_visualization_{timestamp}.png"
        filepath = user_dir / filename
        
        if isinstance(result, str):  # URL from API
            # Download and save image from URL
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
