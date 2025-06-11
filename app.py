import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dashboard components
from dashboard.overview_tab import render_overview_tab
from dashboard.recommendation_tab import render_recommendation_tab
from dashboard.pricing_tab import render_pricing_tab
from dashboard.sentiment_tab import render_sentiment_tab
from dashboard.comparison_tab import render_comparison_tab

# Import utilities
from simulation.data_generator import generate_all_data
from config.airport_profiles import AIRPORT_PROFILES
from utils.styling import apply_custom_css, get_domain_icon, create_domain_header

def main():
    # Configure page
    st.set_page_config(
        page_title="AeroNexus AI - Airport Revenue Optimization",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom dark mode styling
    apply_custom_css()
    
    # Enhanced Header with Airport Icon
    airport_icon = get_domain_icon('airport')
    header_html = f"""
    <div style="text-align: center; padding: 30px 0; background: linear-gradient(135deg, #0E1117 0%, #262730 100%); border-radius: 15px; margin-bottom: 30px; border: 2px solid #FF6B35;">
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
            {airport_icon}
            <h1 style="margin-left: 20px; font-family: 'Orbitron', monospace; font-size: 3rem; background: linear-gradient(45deg, #FF6B35, #F7931E, #FFD700); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(255, 107, 53, 0.5);">
                AeroNexus AI
            </h1>
        </div>
        <p style="font-family: 'Rajdhani', sans-serif; font-size: 1.4rem; color: #FAFAFA; opacity: 0.9; font-weight: 500;">
            AI-Driven Airport Revenue Optimization Platform
        </p>
        <div style="background: linear-gradient(90deg, #FF6B35, #F7931E, #FFD700); height: 3px; width: 200px; margin: 20px auto; border-radius: 2px;"></div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Initialize data if not exists
    if not os.path.exists('data/delhi_passengers.csv'):
        with st.spinner('Initializing simulation data...'):
            generate_all_data()
    
    # Enhanced Sidebar with Logo
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <h2 style="color: white; margin: 0; font-family: 'Orbitron', monospace;">Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Airport selection with icon
        st.markdown(create_domain_header('airport', 'Airport Selection'), unsafe_allow_html=True)
        selected_airport = st.selectbox(
            "Choose Airport",
            options=list(AIRPORT_PROFILES.keys()),
            format_func=lambda x: f"{x} - {AIRPORT_PROFILES[x]['name']}"
        )
    
    # Store selected airport in session state
    if 'selected_airport' not in st.session_state:
        st.session_state.selected_airport = selected_airport
    
    st.session_state.selected_airport = selected_airport
    
    # Date range selector
    st.sidebar.subheader("📅 Analysis Period")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=30),
        max_value=datetime.now()
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    # Store dates in session state
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    
    # Refresh data button
    if st.sidebar.button("🔄 Refresh Data", help="Generate new simulation data"):
        with st.spinner('Refreshing simulation data...'):
            generate_all_data()
        st.success("Data refreshed successfully!")
        st.rerun()
    
    # Enhanced Tab Navigation with Icons
    overview_icon = get_domain_icon('analytics')
    rec_icon = get_domain_icon('retail') 
    pricing_icon = get_domain_icon('pricing')
    sentiment_icon = get_domain_icon('f&b')
    comparison_icon = get_domain_icon('airport')
    
    # Create custom tab headers
    tab_html = f"""
    <div style="background: linear-gradient(135deg, #262730 0%, #1A1D29 100%); padding: 10px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #FF6B35;">
        <div style="display: flex; justify-content: space-around; align-items: center;">
            <div style="text-align: center; font-family: 'Orbitron', monospace; color: #FAFAFA;">
                {overview_icon}<br><span style="font-size: 12px;">Overview</span>
            </div>
            <div style="text-align: center; font-family: 'Orbitron', monospace; color: #FAFAFA;">
                {rec_icon}<br><span style="font-size: 12px;">AI Recommendations</span>
            </div>
            <div style="text-align: center; font-family: 'Orbitron', monospace; color: #FAFAFA;">
                {pricing_icon}<br><span style="font-size: 12px;">Dynamic Pricing</span>
            </div>
            <div style="text-align: center; font-family: 'Orbitron', monospace; color: #FAFAFA;">
                {sentiment_icon}<br><span style="font-size: 12px;">Sentiment & NPS</span>
            </div>
            <div style="text-align: center; font-family: 'Orbitron', monospace; color: #FAFAFA;">
                {comparison_icon}<br><span style="font-size: 12px;">Comparison</span>
            </div>
        </div>
    </div>
    """
    st.markdown(tab_html, unsafe_allow_html=True)
    
    # Main tabs with simplified labels
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "AI Recommendations", 
        "Dynamic Pricing",
        "Sentiment & NPS",
        "DEL vs JAI"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_recommendation_tab()
    
    with tab3:
        render_pricing_tab()
    
    with tab4:
        render_sentiment_tab()
    
    with tab5:
        render_comparison_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 AeroNext Phase 5 Pilot | Powered by AI & IoT | 
        Target: ₹312 → ₹390 revenue uplift | NPS +10 points</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
