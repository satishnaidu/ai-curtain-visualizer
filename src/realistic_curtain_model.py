import cv2
import numpy as np
from PIL import Image
import math
from io import BytesIO
from pathlib import Path
from datetime import datetime
from loguru import logger
from openai import OpenAI
from .exceptions import ModelError

class RealisticCurtainModel:
    """Advanced curtain generation with fabric folding and tone matching"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.debug_dir = Path("debug_realistic_model")
        self.debug_dir.mkdir(exist_ok=True)
        self.sam_predictor = None
    
    async def generate_image(self, prompt: str, room_image: Image.Image, fabric_image: Image.Image, curtain_style: str = None) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save input images
            room_image.save(self.debug_dir / f"{timestamp}_01_input_room.png")
            fabric_image.save(self.debug_dir / f"{timestamp}_02_input_fabric.png")
            logger.info(f"Saved input images to {self.debug_dir}")
            
            # Step 1: Tone match fabric to room lighting
            fabric_matched = self._match_lighting_tone(fabric_image, room_image)
            fabric_matched.save(self.debug_dir / f"{timestamp}_03_tone_matched_fabric.png")
            logger.info("Step 1: Tone matching complete")
            
            # Step 2: Detect window mask using SAM
            window_mask = self._detect_window_mask(room_image)
            Image.fromarray(window_mask).save(self.debug_dir / f"{timestamp}_03_window_mask_sam.png")
            logger.info("Step 2: Window detection (SAM) complete")
            
            # Step 3: Apply AdaIN style transfer to curtain region
            styled_room = self._apply_adain_style_transfer(room_image, fabric_matched, window_mask)
            styled_room.save(self.debug_dir / f"{timestamp}_04_adain_styled.png")
            logger.info("Step 3: AdaIN style transfer complete")
            
            # Step 4: Create folded fabric texture
            folded_fabric = self._apply_fabric_folds(fabric_matched, room_image.size)
            folded_fabric.save(self.debug_dir / f"{timestamp}_05_folded_fabric.png")
            logger.info("Step 4: Fabric folding complete")
            
            # Step 5: Composite onto room
            composite = self._composite_curtain(styled_room, folded_fabric, window_mask)
            composite.save(self.debug_dir / f"{timestamp}_06_composite.png")
            logger.info("Step 5: Compositing complete")
            
            # Step 6: Refine with OpenAI
            result_url = await self._refine_with_openai(composite, prompt, timestamp, window_mask)
            logger.info(f"Step 6: OpenAI refinement complete. All debug images saved to {self.debug_dir}")
            
            return result_url
            
        except Exception as e:
            logger.error(f"Realistic curtain generation error: {e}")
            raise ModelError(f"Failed to generate realistic curtain: {e}")
    
    def _match_lighting_tone(self, fabric_img: Image.Image, room_img: Image.Image) -> Image.Image:
        """Match fabric lighting to room using LAB color space"""
        fabric_cv = cv2.cvtColor(np.array(fabric_img), cv2.COLOR_RGB2LAB)
        room_cv = cv2.cvtColor(np.array(room_img), cv2.COLOR_RGB2LAB)
        
        # Extract luminance channel
        L_room = room_cv[:, :, 0]
        mean_room, std_room = cv2.meanStdDev(L_room)
        mean_fabric, std_fabric = cv2.meanStdDev(fabric_cv[:, :, 0])
        
        # Scale and shift fabric luminance
        L_fabric = fabric_cv[:, :, 0].astype(np.float32)
        L_fabric = (L_fabric - mean_fabric[0][0]) * (std_room[0][0] / (std_fabric[0][0] + 1e-5)) + mean_room[0][0]
        L_fabric = np.clip(L_fabric, 0, 255).astype(np.uint8)
        fabric_cv[:, :, 0] = L_fabric
        
        matched = cv2.cvtColor(fabric_cv, cv2.COLOR_LAB2RGB)
        return Image.fromarray(matched)
    
    def _apply_fabric_folds(self, fabric_img: Image.Image, target_size: tuple) -> Image.Image:
        """Apply realistic vertical folds to fabric"""
        w, h = target_size
        fabric_cv = np.array(fabric_img)
        
        # Tile fabric to target size
        tiled = self._tile_image(fabric_cv, w, h)
        
        # Generate fold displacement
        dx_map, dy_map = self._generate_fold_displacement(w, h)
        
        # Create remap coordinates
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + dx_map).astype(np.float32)
        map_y = (map_y + dy_map).astype(np.float32)
        
        # Warp fabric
        warped = cv2.remap(tiled, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return Image.fromarray(warped)
    
    def _tile_image(self, img_array: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Tile image to cover target dimensions"""
        src_h, src_w = img_array.shape[:2]
        cols = int(np.ceil(target_w / src_w))
        rows = int(np.ceil(target_h / src_h))
        tiled = np.tile(img_array, (rows, cols, 1))
        return tiled[:target_h, :target_w, :]
    
    def _generate_fold_displacement(self, width: int, height: int, amplitude: int = 18, frequency: float = 3.0) -> tuple:
        """Generate displacement maps for vertical curtain folds"""
        xs = np.linspace(0, 2 * math.pi * frequency, width)
        ys = np.linspace(0, 1, height)
        X, Y = np.meshgrid(xs, ys)
        
        # Base sine wave for folds
        base = np.sin(X)
        dx = (base * amplitude).astype(np.float32)
        
        # Add noise for realism
        noise = np.random.randn(height, width).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=6.0, sigmaY=6.0)
        dx += noise * 0.3 * amplitude
        
        # Vertical displacement
        dy = (np.cos(X) * (amplitude * 0.2)).astype(np.float32)
        
        # Gravity gradient
        vertical_grad = np.linspace(0.6, 1.0, height).reshape(height, 1).astype(np.float32)
        dx *= vertical_grad
        dy *= vertical_grad
        
        return dx, dy
    
    def _init_sam(self):
        """Initialize SAM model (lazy loading)"""
        if self.sam_predictor is None:
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                import torch
                
                model_type = "vit_b"
                checkpoint_path = "sam_vit_b_01ec64.pth"
                
                if not Path(checkpoint_path).exists():
                    logger.warning("SAM checkpoint not found, using fallback")
                    return None
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                sam.to(device=device)
                
                self.sam_predictor = SamAutomaticMaskGenerator(sam)
                logger.info(f"SAM initialized on {device}")
            except Exception as e:
                logger.warning(f"SAM init failed: {e}, using fallback")
                self.sam_predictor = None
        
        return self.sam_predictor
    
    def _detect_window_mask_sam(self, room_img: Image.Image) -> np.ndarray:
        """Detect windows/blinds using SAM"""
        room_cv = np.array(room_img)
        h, w = room_cv.shape[:2]
        
        sam = self._init_sam()
        if sam is None:
            return self._detect_window_mask_fallback(room_img)
        
        try:
            masks = sam.generate(room_cv)
            window_masks = []
            
            for mask_data in masks:
                mask = mask_data['segmentation']
                area = np.sum(mask)
                
                y_indices, x_indices = np.where(mask)
                if len(y_indices) == 0:
                    continue
                
                y_min, y_max = y_indices.min(), y_indices.max()
                center_y = (y_min + y_max) / 2
                is_upper = center_y < h * 0.7
                
                bbox_area = (y_max - y_min) * (x_indices.max() - x_indices.min())
                fill_ratio = area / (bbox_area + 1e-5)
                is_large = area > (h * w) * 0.03
                is_rectangular = fill_ratio > 0.6
                
                if is_upper and is_large and is_rectangular:
                    window_masks.append(mask)
            
            if window_masks:
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                for mask in window_masks:
                    combined_mask = np.maximum(combined_mask, mask.astype(np.uint8) * 255)
                combined_mask = cv2.GaussianBlur(combined_mask, (31, 31), 0)
                return combined_mask
            else:
                return self._detect_window_mask_fallback(room_img)
                
        except Exception as e:
            logger.error(f"SAM detection failed: {e}")
            return self._detect_window_mask_fallback(room_img)
    
    def _detect_window_mask_fallback(self, room_img: Image.Image) -> np.ndarray:
        """Fallback window detection"""
        room_cv = np.array(room_img)
        h, w = room_cv.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(room_cv, cv2.COLOR_RGB2GRAY)
        
        # Detect bright areas (windows typically have more light)
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Detect edges to find window frames
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine bright areas and edges
        combined = cv2.bitwise_or(bright_mask, edges)
        
        # Morphological operations to fill gaps
        kernel = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask from largest contours (likely windows)
        mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            # Filter contours by size (windows should be reasonably large)
            min_area = (h * w) * 0.05  # At least 5% of image
            large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if large_contours:
                cv2.drawContours(mask, large_contours, -1, 255, -1)
            else:
                # Fallback: use upper portion of image where windows typically are
                mask[int(h*0.1):int(h*0.7), int(w*0.1):int(w*0.9)] = 255
        else:
            # Fallback: use upper portion of image
            mask[int(h*0.1):int(h*0.7), int(w*0.1):int(w*0.9)] = 255
        
        # Smooth mask edges
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        return mask
    
    def _detect_window_mask(self, room_img: Image.Image) -> np.ndarray:
        """Detect windows using SAM or fallback"""
        return self._detect_window_mask_sam(room_img)
    
    def _apply_adain_style_transfer(self, content_img: Image.Image, style_img: Image.Image, mask: np.ndarray) -> Image.Image:
        """Apply AdaIN style transfer to blend fabric texture onto curtain region"""
        content_cv = np.array(content_img).astype(np.float32)
        style_cv = np.array(style_img).astype(np.float32)
        h, w = content_cv.shape[:2]
        
        # Resize style to match content
        if style_cv.shape[:2] != (h, w):
            style_cv = cv2.resize(style_cv, (w, h))
        
        # Convert to LAB color space for better style transfer
        content_lab = cv2.cvtColor(content_cv.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        style_lab = cv2.cvtColor(style_cv.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Apply AdaIN only to masked region
        mask_3d = (mask[:, :, np.newaxis] / 255.0)
        
        # Calculate mean and std for each channel in masked region
        styled_lab = content_lab.copy()
        for c in range(3):
            content_channel = content_lab[:, :, c]
            style_channel = style_lab[:, :, c]
            
            # Calculate statistics only in masked region
            masked_content = content_channel[mask > 128]
            masked_style = style_channel[mask > 128]
            
            if len(masked_content) > 0 and len(masked_style) > 0:
                content_mean = np.mean(masked_content)
                content_std = np.std(masked_content) + 1e-5
                style_mean = np.mean(masked_style)
                style_std = np.std(masked_style) + 1e-5
                
                # AdaIN formula: (content - content_mean) / content_std * style_std + style_mean
                normalized = (content_channel - content_mean) / content_std
                styled_channel = normalized * style_std + style_mean
                
                # Blend with original using mask
                styled_lab[:, :, c] = content_channel * (1 - mask_3d[:, :, 0]) + styled_channel * mask_3d[:, :, 0]
        
        # Convert back to RGB
        styled_lab = np.clip(styled_lab, 0, 255).astype(np.uint8)
        styled_rgb = cv2.cvtColor(styled_lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(styled_rgb)
    
    def _composite_curtain(self, room_img: Image.Image, curtain_img: Image.Image, mask: np.ndarray) -> Image.Image:
        """Composite curtain onto room with alpha blending"""
        room_cv = np.array(room_img)
        curtain_cv = np.array(curtain_img)
        
        # Resize curtain if needed
        if curtain_cv.shape[:2] != room_cv.shape[:2]:
            curtain_cv = cv2.resize(curtain_cv, (room_cv.shape[1], room_cv.shape[0]))
        
        # Use provided mask
        mask_3 = cv2.merge([mask, mask, mask]) / 255.0
        
        # Blend
        composite = (room_cv.astype(np.float32) * (1.0 - mask_3) + 
                    curtain_cv.astype(np.float32) * mask_3).astype(np.uint8)
        
        return Image.fromarray(composite)
    
    async def _refine_with_openai(self, composite_img: Image.Image, prompt: str, timestamp: str, window_mask: np.ndarray) -> str:
        """Refine composite with OpenAI image edit"""
        import asyncio
        
        # Save to bytes with proper naming
        img_bytes = BytesIO()
        composite_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        img_bytes.name = 'image.png'
        
        # Use detected window mask for OpenAI
        mask_img = Image.fromarray(window_mask)
        
        # Save mask for debugging
        mask_img.save(self.debug_dir / f"{timestamp}_07_openai_mask.png")
        
        mask_bytes = BytesIO()
        mask_img.save(mask_bytes, format='PNG')
        mask_bytes.seek(0)
        mask_bytes.name = 'mask.png'
        
        # Call OpenAI with shortened prompt
        refine_prompt = f"Refine curtain area: photo-realistic lighting, soft shadows, fabric highlights. Keep fold texture and room unchanged."[:1000]
        
        response = await asyncio.to_thread(
            self.client.images.edit,
            image=img_bytes,
            mask=mask_bytes,
            prompt=refine_prompt,
            size="1024x1024",
            n=1
        )
        
        return response.data[0].url
