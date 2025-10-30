import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

class UserManager:
    def __init__(self):
        self.users_file = Path("users.json")
        self.users_data = self._load_users()
    
    def _load_users(self) -> Dict:
        """Load users data from JSON file"""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading users data: {e}")
        return {}
    
    def _save_users(self):
        """Save users data to JSON file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users data: {e}")
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        # Simple validation for international format
        pattern = r'^\+?[1-9]\d{1,14}$'
        return bool(re.match(pattern, phone.replace(' ', '').replace('-', '')))
    
    def get_user_credits(self, phone: str) -> int:
        """Get remaining credits for user"""
        clean_phone = self._clean_phone(phone)
        return self.users_data.get(clean_phone, {}).get('credits', 0)
    
    def add_credits(self, phone: str, credits: int = 20) -> bool:
        """Add credits to user account"""
        clean_phone = self._clean_phone(phone)
        
        if clean_phone not in self.users_data:
            self.users_data[clean_phone] = {
                'credits': 0,
                'created_at': datetime.now().isoformat(),
                'total_generated': 0
            }
        
        self.users_data[clean_phone]['credits'] += credits
        self.users_data[clean_phone]['last_purchase'] = datetime.now().isoformat()
        self._save_users()
        
        logger.info(f"Added {credits} credits to {clean_phone}")
        return True
    
    def use_credit(self, phone: str) -> bool:
        """Use one credit for image generation"""
        clean_phone = self._clean_phone(phone)
        
        if clean_phone not in self.users_data:
            return False
        
        if self.users_data[clean_phone]['credits'] <= 0:
            return False
        
        self.users_data[clean_phone]['credits'] -= 1
        self.users_data[clean_phone]['total_generated'] += 1
        self.users_data[clean_phone]['last_used'] = datetime.now().isoformat()
        self._save_users()
        
        logger.info(f"Used 1 credit for {clean_phone}, remaining: {self.users_data[clean_phone]['credits']}")
        return True
    
    def _clean_phone(self, phone: str) -> str:
        """Clean and normalize phone number"""
        return re.sub(r'[^\d+]', '', phone)
    
    def get_user_stats(self, phone: str) -> Dict:
        """Get user statistics"""
        clean_phone = self._clean_phone(phone)
        return self.users_data.get(clean_phone, {})