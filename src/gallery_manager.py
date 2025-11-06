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
        self._sync_from_s3()
    
    def _init_s3_client(self) -> Optional[boto3.client]:
        """Initialize S3 client if credentials are available"""
        if not config.aws_access_key_id or not config.aws_secret_access_key or not config.aws_s3_bucket:
            logger.warning(f"S3 not configured - missing credentials or bucket. Key: {bool(config.aws_access_key_id)}, Secret: {bool(config.aws_secret_access_key)}, Bucket: {config.aws_s3_bucket}")
            return None
        
        try:
            client = boto3.client(
                's3',
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key,
                region_name=config.aws_s3_region
            )
            logger.info(f"S3 client initialized successfully for bucket: {config.aws_s3_bucket}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            return None
    
    def _upload_to_s3(self, local_path: str, s3_key: str) -> Optional[str]:
        """Upload file to S3 and return public URL"""
        if not self.s3_client:
            logger.warning(f"S3 client not available, skipping upload for {s3_key}")
            return None
        
        try:
            logger.info(f"Uploading {local_path} to s3://{config.aws_s3_bucket}/{s3_key}")
            # Upload without ACL (bucket has ACLs disabled)
            self.s3_client.upload_file(
                local_path,
                config.aws_s3_bucket,
                s3_key,
                ExtraArgs={'ContentType': self._get_content_type(local_path)}
            )
            url = f"https://{config.aws_s3_bucket}.s3.{config.aws_s3_region}.amazonaws.com/{s3_key}"
            logger.info(f"Successfully uploaded to S3: {url}")
            return url
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"S3 upload failed for {s3_key}: {error_code} - {error_msg}")
            return None
        except Exception as e:
            logger.error(f"S3 upload failed for {s3_key}: {str(e)}")
            return None
    
    def _get_content_type(self, file_path: str) -> str:
        """Get content type based on file extension"""
        ext = Path(file_path).suffix.lower()
        return 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
    
    def _download_from_s3(self, s3_url: str, local_path: str) -> bool:
        """Download file from S3 to local path"""
        if not self.s3_client or not s3_url:
            return False
        
        try:
            s3_key = s3_url.split(f"{config.aws_s3_bucket}.s3.{config.aws_s3_region}.amazonaws.com/")[-1]
            logger.info(f"Downloading {s3_key} to {local_path}")
            self.s3_client.download_file(config.aws_s3_bucket, s3_key, local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download {s3_url}: {e}")
            return False
    
    def _sync_from_s3(self):
        """Sync missing local files from S3 on startup"""
        if not self.s3_client:
            return
        
        logger.info("Syncing gallery images from S3...")
        for entry in self.gallery_data:
            # Download missing local files from S3
            if entry.get('room_url') and not os.path.exists(entry.get('room_photo_path', '')):
                self._download_from_s3(entry['room_url'], entry['room_photo_path'])
            if entry.get('fabric_url') and not os.path.exists(entry.get('fabric_photo_path', '')):
                self._download_from_s3(entry['fabric_url'], entry['fabric_photo_path'])
            if entry.get('result_url') and not os.path.exists(entry.get('result_path', '')):
                self._download_from_s3(entry['result_url'], entry['result_path'])
        logger.info("S3 sync complete")
    
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
                logger.info("Uploading gallery images to S3...")
                s3_urls['room_url'] = self._upload_to_s3(str(room_gallery_path), f"gallery/{base_name}_room.jpg")
                s3_urls['fabric_url'] = self._upload_to_s3(str(fabric_gallery_path), f"gallery/{base_name}_fabric.jpg")
                s3_urls['result_url'] = self._upload_to_s3(str(result_gallery_path), f"gallery/{base_name}_result.png")
                logger.info(f"S3 upload complete. URLs: {list(s3_urls.keys())}")
            else:
                logger.warning("S3 client not initialized, skipping S3 upload")
            
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
        """Get all gallery entries using local files"""
        valid_entries = []
        for entry in self.gallery_data:
            has_local = (os.path.exists(entry.get("room_photo_path", "")) and 
                        os.path.exists(entry.get("fabric_photo_path", "")) and 
                        os.path.exists(entry.get("result_path", "")))
            
            if has_local:
                entry['display_room'] = entry['room_photo_path']
                entry['display_fabric'] = entry['fabric_photo_path']
                entry['display_result'] = entry['result_path']
                valid_entries.append(entry)
        
        if len(valid_entries) != len(self.gallery_data):
            self.gallery_data = valid_entries
            self._save_gallery()
        
        return valid_entries[::-1]