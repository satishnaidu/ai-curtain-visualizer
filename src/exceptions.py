from typing import Optional

class ImageProcessingError(Exception):
    """Base exception for image processing errors"""
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.message = message

class ImageValidationError(ImageProcessingError):
    """Raised when image validation fails"""
    pass

class APIError(Exception):
    """Raised when API calls fail"""
    def __init__(self, message: str, status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.message = message

class ModelError(Exception):
    """Raised when model operations fail"""
    pass

class CacheError(Exception):
    """Raised when cache operations fail"""
    pass
