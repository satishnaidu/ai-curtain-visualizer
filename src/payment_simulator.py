import streamlit as st
import stripe
from typing import Dict
from loguru import logger
from .config import config

class StripePaymentProcessor:
    """Stripe payment processing for credits"""
    
    def __init__(self):
        if config.stripe_secret_key:
            stripe.api_key = config.stripe_secret_key
        self.price_per_credit = 0.50  # $0.50 per credit
        self.credit_packages = {
            "10": {"credits": 10, "price": 500, "description": "10 Credits - $5.00"},
            "20": {"credits": 20, "price": 1000, "description": "20 Credits - $10.00"},
            "50": {"credits": 50, "price": 2000, "description": "50 Credits - $20.00 (Best Value!)"}
        }
    
    def show_payment_form(self, phone: str) -> bool:
        """Show Stripe payment form"""
        st.subheader("💳 Purchase Credits")
        st.write(f"**Phone:** {phone}")
        
        # Package selection
        package_options = list(self.credit_packages.keys())
        package_labels = [self.credit_packages[p]["description"] for p in package_options]
        
        selected_package = st.selectbox(
            "Select Credit Package:",
            options=package_options,
            format_func=lambda x: self.credit_packages[x]["description"]
        )
        
        package_info = self.credit_packages[selected_package]
        

        if not config.stripe_secret_key:
            st.warning("⚠️ Stripe not configured. Using demo mode.")
            return self._show_demo_form(package_info)
        
        # Show selected package info
        st.info(f"Selected: {package_info['description']}")
        
        # Secure Stripe Checkout
        st.write("**Secure Payment with Stripe**")
        st.info("🔒 Your payment will be processed securely by Stripe. We never store your card details.")
        
        if st.button(f"💰 Pay ${package_info['price']/100:.2f} for {package_info['credits']} Credits", use_container_width=True):
            return self._create_stripe_checkout(phone, package_info)
        
        return False
    
    def _create_stripe_checkout(self, phone: str, package_info: Dict) -> bool:
        """Create Stripe Checkout session"""
        try:
            with st.spinner("Creating secure checkout..."):
                # Create Stripe Checkout session
                checkout_session = stripe.checkout.Session.create(
                    payment_method_types=['card'],
                    line_items=[{
                        'price_data': {
                            'currency': 'usd',
                            'product_data': {
                                'name': f'{package_info["credits"]} AI Curtain Credits',
                                'description': f'Credits for AI curtain visualization - Phone: {phone}'
                            },
                            'unit_amount': package_info['price'],
                        },
                        'quantity': 1,
                    }],
                    mode='payment',
                    success_url=f'https://dev-curtainvision.streamlit.app/?session_id={{CHECKOUT_SESSION_ID}}&phone={phone.replace("+", "%2B")}',
                    cancel_url='https://dev-curtainvision.streamlit.app/cancel',
                    metadata={
                        'phone': phone,
                        'credits': package_info['credits']
                    }
                )
                
                # Store session info for verification
                st.session_state.checkout_session_id = checkout_session.id
                st.session_state.pending_credits = package_info['credits']
                st.session_state.checkout_phone = phone
                
                logger.info(f"Stripe checkout created for {phone}: {package_info['credits']} credits, Session ID: {checkout_session.id}")
                
                # Attractive payment button with custom styling
                st.markdown("---")
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 30px; border-radius: 15px; text-align: center; 
                            box-shadow: 0 10px 25px rgba(0,0,0,0.2); margin: 20px 0;">
                    <h2 style="color: white; margin-bottom: 15px;">🔒 Secure Checkout Ready</h2>
                    <p style="color: #f0f0f0; font-size: 18px; margin-bottom: 25px;">
                        Your payment will be processed securely through Stripe<br/>
                        <small>🛡️ Bank-level encryption • No card details stored</small>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Large prominent button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.link_button(
                        f"💳 Complete Payment - ${package_info['price']/100:.2f}",
                        checkout_session.url,
                        type="primary",
                        use_container_width=True
                    )
                
                st.markdown("""
                <div style="text-align: center; margin-top: 15px; color: #666;">
                    <small>✓ Instant credit activation • ✓ Secure payment • ✓ Money-back guarantee</small>
                </div>
                """, unsafe_allow_html=True)
                
                # For demo purposes, show manual verification option
                st.divider()
                st.caption("**For Demo: Manual Payment Verification**")
                if st.button("✅ Mark Payment as Completed (Demo)", type="secondary"):
                    st.success(f"✅ Payment verified! {package_info['credits']} credits added to your account.")
                    st.balloons()
                    return True
                
                return False
                
        except stripe.error.StripeError as e:
            st.error(f"❌ Checkout creation failed: {str(e)}")
            logger.error(f"Stripe checkout error: {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Error creating checkout: {str(e)}")
            logger.error(f"Checkout creation error: {str(e)}")
            return False
    
    def _show_demo_form(self, package_info: Dict) -> bool:
        """Show demo payment form when Stripe is not configured"""
        st.write("**Demo Payment Mode**")
        st.info("🧪 Stripe not configured. Using demo mode - no real charges.")
        
        if st.button(f"💰 Demo Pay ${package_info['price']/100:.2f} for {package_info['credits']} Credits", use_container_width=True):
            with st.spinner("Processing demo payment..."):
                import time
                time.sleep(1)
            
            st.success(f"✅ Demo payment successful! {package_info['credits']} credits added.")
            st.balloons()
            return True
        
        return False

# Backward compatibility
PaymentSimulator = StripePaymentProcessor