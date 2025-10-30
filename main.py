import streamlit as st
import asyncio
import os
from pathlib import Path
from loguru import logger
from src.image_processor import ImageProcessor
from src.exceptions import ImageProcessingError, ImageValidationError, APIError, ModelError
from src.config import config, ModelType
from src.logging_config import setup_logging
from src.user_manager import UserManager
from src.payment_simulator import PaymentSimulator

# Setup logging and create logs directory
Path("logs").mkdir(exist_ok=True)
setup_logging()

class CurtainVisualizerApp:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.user_manager = UserManager()
        self.payment_simulator = PaymentSimulator()
        self.setup_page()
        logger.info("CurtainVisualizerApp initialized")

    def setup_page(self):
        """Initialize Streamlit page configuration"""
        st.set_page_config(
            page_title="AI Curtain Visualizer", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("🪟 AI Curtain Visualizer - Production Grade")
        
        # Enhanced sidebar with model info and settings
        with st.sidebar:
            st.header("Configuration")
            if config.test_mode:
                st.success("🧪 **TEST MODE** - No API calls")
            st.write(f"**Model:** {config.effective_model_type.value}")
            st.write(f"**Auto-Optimization:** Images resized to max {config.max_image_dimension}px")
            st.write(f"**Supported Formats:** {', '.join(config.allowed_image_types)}")
            
            # Model selection (if multiple models available)
            if st.button("🔄 Reload Model"):
                st.cache_resource.clear()
                st.rerun()
            
            st.divider()
            st.header("About")
            st.write("Advanced AI-powered curtain visualization using LangChain and production-grade ML frameworks.")

        st.write("Upload a room photo and fabric pattern to generate a professional curtain visualization.")
        
        # Add usage instructions
        with st.expander("📋 Usage Instructions"):
            st.markdown("""
            1. **Room Photo**: Upload a clear photo of your room with visible windows
            2. **Fabric Pattern**: Upload a photo of your desired curtain fabric
            3. **Generate**: Click the button to create your visualization
            
            **Tips for best results:**
            - Use high-quality, well-lit photos
            - Ensure windows are clearly visible in room photos
            - Fabric photos should show texture and color clearly
            """)

    def run(self):
        """Run the Streamlit application with enhanced error handling"""
        # User authentication
        if not self._handle_authentication():
            return
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏠 Room Photo")
            room_photo = st.file_uploader(
                "Upload room photo",
                type=list(config.allowed_image_types),
                help="Upload a clear photo of your room showing windows where curtains will be installed"
            )
            if room_photo:
                st.image(room_photo, caption=f"Room Photo ({room_photo.size} bytes)", use_column_width=True)
                # Show image metadata
                st.caption(f"File: {room_photo.name} | Size: {room_photo.size:,} bytes (will be auto-optimized)")

        with col2:
            st.subheader("🧵 Fabric Pattern")
            fabric_photo = st.file_uploader(
                "Upload fabric pattern",
                type=list(config.allowed_image_types),
                help="Upload a photo of your desired curtain fabric showing color and texture"
            )
            if fabric_photo:
                st.image(fabric_photo, caption=f"Fabric Pattern ({fabric_photo.size} bytes)", use_column_width=True)
                st.caption(f"File: {fabric_photo.name} | Size: {fabric_photo.size:,} bytes (will be auto-optimized)")

        # Generation section
        st.divider()
        
        col_btn, col_status = st.columns([1, 3])
        
        with col_btn:
            # Get current credits for button state
            phone = st.session_state.user_phone
            credits = self.user_manager.get_user_credits(phone)
            
            # Disable button if no credits or missing images
            button_disabled = not (room_photo and fabric_photo) or (credits <= 0 and not config.test_mode)
            
            generate_btn = st.button(
                "🎨 Generate Visualization" if credits > 0 or config.test_mode else "❌ No Credits", 
                type="primary" if credits > 0 or config.test_mode else "secondary",
                disabled=button_disabled,
                use_container_width=True
            )
        
        with col_status:
            if not room_photo or not fabric_photo:
                st.info("Please upload both room photo and fabric pattern to continue.")
            elif credits <= 0 and not config.test_mode:
                st.warning(f"⚠️ No credits remaining. Purchase more to continue.")

        # Check credits for button state
        phone = st.session_state.user_phone
        current_credits = self.user_manager.get_user_credits(phone)
        
        if generate_btn and current_credits > 0:
            # Deduct credit immediately before generation
            if not config.test_mode:
                if not self.user_manager.use_credit(phone):
                    st.error("❌ Failed to deduct credit. Please try again.")
                    return
            
            self._handle_generation(room_photo, fabric_photo)
        elif generate_btn and current_credits <= 0:
            st.error("❌ No credits remaining. Please purchase more credits to continue.")
            if st.button("💳 Buy More Credits"):
                st.session_state.show_payment = True
                st.rerun()
    
    def _handle_generation(self, room_photo, fabric_photo):
        """Handle the image generation process with comprehensive error handling"""
        try:
            logger.info(f"Starting generation for room: {room_photo.name}, fabric: {fabric_photo.name}")
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔄 Processing images and generating visualization..."):
                # Update progress
                progress_bar.progress(20)
                status_text.text("Validating images...")
                
                # Run async processing
                result, saved_path = asyncio.run(self._async_generate(room_photo, fabric_photo, progress_bar, status_text, st.session_state.user_phone))
                
                progress_bar.progress(100)
                status_text.text("✅ Generation complete!")
                
                # Display result
                if config.test_mode:
                    st.success("🧪 Test visualization generated successfully!")
                    st.info("This is a mock result created by combining your images. Enable API mode for AI-generated results.")
                else:
                    st.success("Curtain visualization generated successfully!")
                
                if isinstance(result, str):  # URL from DALL-E
                    st.image(result, caption="Generated Curtain Visualization", use_column_width=True)
                    st.markdown(f"[🔗 Open in new tab]({result})")
                else:  # PIL Image from Test/Stable Diffusion
                    st.image(result, caption="Generated Curtain Visualization", use_column_width=True)
                
                # Add download button
                if os.path.exists(saved_path):
                    with open(saved_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Image",
                            data=file.read(),
                            file_name=os.path.basename(saved_path),
                            mime="image/png",
                            use_container_width=True
                        )
                    st.caption(f"💾 Saved to: {saved_path}")
                
                # Cleanup progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show remaining credits after generation
                if not config.test_mode:
                    remaining = self.user_manager.get_user_credits(st.session_state.user_phone)
                    st.info(f"💳 Credits remaining: {remaining}")
                    if remaining <= 0:
                        st.warning("⚠️ No credits remaining! Purchase more to continue generating images.")
                
                logger.success("Visualization generated and displayed successfully")

        except ImageValidationError as e:
            logger.warning(f"Image validation error: {e.message}")
            st.error(f"❌ Image validation error: {e.message}")
            if e.error_code:
                st.caption(f"Error code: {e.error_code}")
                
        except APIError as e:
            logger.error(f"API error: {e.message}")
            st.error(f"🔌 API error: {e.message}")
            if e.status_code:
                st.caption(f"Status code: {e.status_code}")
            if e.retry_after:
                st.info(f"Please try again in {e.retry_after} seconds.")
                
        except ModelError as e:
            logger.error(f"Model error: {str(e)}")
            st.error(f"🤖 Model error: {str(e)}")
            st.info("This might be a temporary issue. Please try again.")
            
        except ImageProcessingError as e:
            logger.error(f"Processing error: {e.message}")
            st.error(f"⚙️ Processing error: {e.message}")
            if e.error_code:
                st.caption(f"Error code: {e.error_code}")
                
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            st.error(f"💥 An unexpected error occurred: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")
    
    async def _async_generate(self, room_photo, fabric_photo, progress_bar, status_text, user_phone):
        """Async wrapper for image generation with progress updates"""
        progress_bar.progress(40)
        status_text.text("Analyzing images...")
        
        # Process images with user phone
        result, saved_path = await self.image_processor.process_images(room_photo, fabric_photo, user_phone)
        
        progress_bar.progress(80)
        status_text.text("Saving image...")
        
        return result, saved_path
    
    def _handle_authentication(self) -> bool:
        """Handle user authentication and credit management"""
        # Check if payment form should be shown
        if st.session_state.get('show_payment', False):
            if self._handle_payment():
                st.session_state.show_payment = False
                st.rerun()
            return False
        
        # Check if user is already authenticated
        if 'user_phone' in st.session_state:
            self._show_user_dashboard()
            return True
        
        # Show login form
        st.subheader("📱 Enter Your Phone Number")
        st.write("We use your phone number to track your image credits.")
        
        with st.form("phone_form"):
            phone = st.text_input(
                "Phone Number", 
                placeholder="+1234567890",
                help="Enter your phone number with country code"
            )
            
            submitted = st.form_submit_button("🚀 Continue", use_container_width=True)
            
            if submitted:
                if not phone:
                    st.error("Please enter your phone number")
                    return False
                
                if not self.user_manager.validate_phone(phone):
                    st.error("Please enter a valid phone number with country code (e.g., +1234567890)")
                    return False
                
                st.session_state.user_phone = phone
                
                # Check if user has credits
                credits = self.user_manager.get_user_credits(phone)
                if credits <= 0:
                    st.session_state.show_payment = True
                
                st.rerun()
        
        return False
    
    def _show_user_dashboard(self):
        """Show user dashboard with credits and stats"""
        phone = st.session_state.user_phone
        credits = self.user_manager.get_user_credits(phone)
        stats = self.user_manager.get_user_stats(phone)
        
        with st.sidebar:
            st.divider()
            st.header("👤 Your Account")
            st.write(f"**Phone:** {phone}")
            st.write(f"**Credits:** {credits} remaining")
            
            if stats:
                st.write(f"**Generated:** {stats.get('total_generated', 0)} images")
            
            if credits <= 5:
                st.warning(f"⚠️ Low credits: {credits} remaining")
            
            if st.button("💳 Buy More Credits"):
                st.session_state.show_payment = True
                st.rerun()
            
            if st.button("🚪 Logout"):
                del st.session_state.user_phone
                if 'show_payment' in st.session_state:
                    del st.session_state.show_payment
                st.rerun()
    
    def _handle_payment(self) -> bool:
        """Handle payment processing"""
        phone = st.session_state.user_phone
        
        if self.payment_simulator.show_payment_form(phone):
            # Add credits to user account
            self.user_manager.add_credits(phone, 20)
            return True
        
        return False


if __name__ == "__main__":
    try:
        app = CurtainVisualizerApp()
        app.run()
    except Exception as e:
        logger.exception(f"Failed to start application: {str(e)}")
        st.error(f"Failed to start application: {str(e)}")