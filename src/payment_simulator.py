import streamlit as st
from typing import Dict
from loguru import logger

class PaymentSimulator:
    """Simulates payment processing for MVP"""
    
    @staticmethod
    def show_payment_form(phone: str) -> bool:
        """Show payment form and simulate payment processing"""
        st.subheader("💳 Purchase Credits")
        st.write(f"**Phone:** {phone}")
        st.write("**Package:** 20 Images for $10")
        
        with st.form("payment_form"):
            st.write("**Payment Details** (Demo - No real charges)")
            
            col1, col2 = st.columns(2)
            with col1:
                card_number = st.text_input("Card Number", value="4111111111111111", disabled=True)
                expiry = st.text_input("MM/YY", value="12/25", disabled=True)
            with col2:
                cvv = st.text_input("CVV", value="123", type="password", disabled=True)
                name = st.text_input("Cardholder Name", value="Demo User", disabled=True)
            
            st.info("🧪 This is a demo payment form. No real charges will be made.")
            
            submitted = st.form_submit_button("💰 Pay $10 for 20 Credits", use_container_width=True)
            
            if submitted:
                # Simulate payment processing
                with st.spinner("Processing payment..."):
                    import time
                    time.sleep(2)  # Simulate processing delay
                
                st.success("✅ Payment successful! 20 credits added to your account.")
                st.balloons()
                return True
        
        return False