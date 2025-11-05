import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from PIL import Image
import boto3
from botocore.exceptions import ClientError
from src.config import config

class GalleryManager:
    def __init__(self):
        self.gallery_file = Path("gallery.json")
        self.gallery_dir = Path("gallery_images")
        self.gallery_dir.mkdir(exist_ok=True)
        self.gallery_data = self._load_gallery()
        self.s3_client = self._init_s3_client()
    
    def _init_s3_client(self) -> Optional[boto3.client]:
        """Initialize S3 client if credentials are available"""
        if config.aws_access_key_id and config.aws_secret_access_key and config.aws_s3_bucket:
            try:
                return boto3.client(
                    's3',
                    aws_access_key_id=config.aws_access_key_id,
                    aws_secret_access_key=config.aws_secret_access_key,
                    region_name=config.aws_s3_region
                )
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
        return None
    
    def _upload_to_s3(self, local_path: str, s3_key: str) -> Optional[str]:
        """Upload file to S3 and return public URL"""
        if not self.s3_client:
            return None
        
        try:
            self.s3_client.upload_file(
                local_path,
                config.aws_s3_bucket,
                s3_key,
                ExtraArgs={'ACL': 'public-read', 'ContentType': self._get_content_type(local_path)}
            )
            url = f"https://{config.aws_s3_bucket}.s3.{config.aws_s3_region}.amazonaws.com/{s3_key}"
            logger.info(f"Uploaded to S3: {url}")
            return url
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return None
    
    def _get_content_type(self, file_path: str) -> str:
        """Get content type based on file extension"""
        ext = Path(file_path).suffix.lower()
        return 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
    
    def _load_gallery(self) -> List[Dict]:
        """Load gallery data from JSON file"""
        if self.gallery_file.exists():
            try:
                with open(self.gallery_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading gallery data: {e}")
        return []
    
    def _save_gallery(self):
        """Save gallery data to JSON file"""
        try:
            with open(self.gallery_file, 'w') as f:
                json.dump(self.gallery_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving gallery data: {e}")
    
    def add_to_gallery(self, room_photo, fabric_photo, result_path: str, user_phone: str = None):
        """Add a new entry to the gallery with original photos"""
        try:
            # Generate unique filename base
            timestamp = Path(result_path).stem.split('_')[-2:] if '_' in Path(result_path).stem else ["demo"]
            base_name = f"gallery_{timestamp[0]}_{timestamp[1]}" if len(timestamp) >= 2 else "gallery_demo"
            
            # Save room photo to gallery
            room_gallery_path = self.gallery_dir / f"{base_name}_room.jpg"
            room_photo.seek(0)  # Reset file pointer
            room_image = Image.open(room_photo).convert("RGB")
            room_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            room_image.save(room_gallery_path, "JPEG", quality=85)
            
            # Save fabric photo to gallery
            fabric_gallery_path = self.gallery_dir / f"{base_name}_fabric.jpg"
            fabric_photo.seek(0)  # Reset file pointer
            fabric_image = Image.open(fabric_photo).convert("RGB")
            fabric_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            fabric_image.save(fabric_gallery_path, "JPEG", quality=85)
            
            # Copy result image to gallery
            result_gallery_path = self.gallery_dir / f"{base_name}_result.png"
            shutil.copy2(result_path, result_gallery_path)
            
            # Upload to S3 if configured
            s3_urls = {}
            if self.s3_client:
                s3_urls['room_url'] = self._upload_to_s3(str(room_gallery_path), f"gallery/{base_name}_room.jpg")
                s3_urls['fabric_url'] = self._upload_to_s3(str(fabric_gallery_path), f"gallery/{base_name}_fabric.jpg")
                s3_urls['result_url'] = self._upload_to_s3(str(result_gallery_path), f"gallery/{base_name}_result.png")
            
            entry = {
                "room_photo_path": str(room_gallery_path),
                "fabric_photo_path": str(fabric_gallery_path),
                "result_path": str(result_gallery_path),
                "user_phone": user_phone[:4] + "****" if user_phone else "Demo",
                "timestamp": timestamp,
                **s3_urls
            }
            
            self.gallery_data.append(entry)
            # Keep only last 20 entries
            if len(self.gallery_data) > 20:
                # Remove old files
                old_entry = self.gallery_data[0]
                for path_key in ["room_photo_path", "fabric_photo_path", "result_path"]:
                    if os.path.exists(old_entry.get(path_key, "")):
                        os.remove(old_entry[path_key])
                
                self.gallery_data = self.gallery_data[-20:]
            
            self._save_gallery()
            logger.info(f"Added entry to gallery: {entry}")
            
        except Exception as e:
            logger.error(f"Error adding to gallery: {e}")
    
    def get_gallery_entries(self) -> List[Dict]:
        """Get all gallery entries, prioritizing S3 URLs"""
        valid_entries = []
        for entry in self.gallery_data:
            # Check if S3 URLs exist, otherwise check local files
            has_s3 = entry.get('room_url') and entry.get('fabric_url') and entry.get('result_url')
            has_local = (os.path.exists(entry.get("room_photo_path", "")) and 
                        os.path.exists(entry.get("fabric_photo_path", "")) and 
                        os.path.exists(entry.get("result_path", "")))
            
            if has_s3 or has_local:
                # Prioritize S3 URLs if available
                if has_s3:
                    entry['display_room'] = entry['room_url']
                    entry['display_fabric'] = entry['fabric_url']
                    entry['display_result'] = entry['result_url']
                else:
                    entry['display_room'] = entry['room_photo_path']
                    entry['display_fabric'] = entry['fabric_photo_path']
                    entry['display_result'] = entry['result_path']
                valid_entries.append(entry)
        
        if len(valid_entries) != len(self.gallery_data):
            self.gallery_data = valid_entries
            self._save_gallery()
        
        return valid_entries[::-1]