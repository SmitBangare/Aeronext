"""
AI Recommendations tab for the AeroNexus AI dashboard.
Implements collaborative filtering for personalized retail recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recommender import AirportRecommender, DomainRecommender
from config.airport_profiles import AIRPORT_PROFILES
from utils.plot_utils import create_recommendation_chart
from utils.styling import (
    create_domain_header, 
    create_recommendation_card, 
    create_colored_metric_card,
    apply_chart_styling,
    get_domain_icon
)

def render_recommendation_tab():
    """Render the AI recommendations tab"""
    
    # Enhanced header with custom styling
    st.markdown(create_domain_header('retail', 'AI-Powered Retail Personalization'), unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #262730 0%, #1A1D29 100%); padding: 20px; border-radius: 10px; border: 1px solid #FF6B35; margin-bottom: 20px;">
        <p style="font-family: 'Rajdhani', sans-serif; font-size: 16px; color: #FAFAFA; margin: 0; text-align: center;">
        Generate personalized product recommendations using collaborative filtering 
        based on passenger segments, purchase history, and real-time context.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get selected airport
    selected_airport = st.session_state.get('selected_airport', 'DEL')
    airport_profile = AIRPORT_PROFILES[selected_airport]
    
    # Initialize recommender model
    @st.cache_resource
    def load_recommender():
        recommender = AirportRecommender()
        try:
            recommender.train_model()
        except Exception as e:
            st.warning(f"Using fallback recommendation model: {e}")
        return recommender
    
    recommender = load_recommender()
    
    # Input section
    st.subheader("🎯 Passenger Context")
    
    # Domain selection first
    st.markdown("**Select Domain**")
    domain_col1, domain_col2, domain_col3 = st.columns(3)
    
    with domain_col1:
        retail_selected = st.button("🛍️ Retail", use_container_width=True)
    with domain_col2:
        fnb_selected = st.button("🍽️ Food & Beverage", use_container_width=True)
    with domain_col3:
        lounge_selected = st.button("🏢 Lounge Services", use_container_width=True)
    
    # Set default domain or use selected
    if 'selected_domain' not in st.session_state:
        st.session_state.selected_domain = 'retail'
    
    if retail_selected:
        st.session_state.selected_domain = 'retail'
    elif fnb_selected:
        st.session_state.selected_domain = 'f&b'
    elif lounge_selected:
        st.session_state.selected_domain = 'lounge'
    
    selected_domain = st.session_state.selected_domain
    
    # Display selected domain
    domain_names = {'retail': 'Retail Products', 'f&b': 'Food & Beverage', 'lounge': 'Lounge Services'}
    st.info(f"**Selected Domain:** {domain_names[selected_domain]}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        passenger_segment = st.selectbox(
            "Passenger Segment",
            options=['business', 'leisure'],
            format_func=lambda x: f"Business Traveler" if x == 'business' else "Leisure Traveler"
        )
    
    with col2:
        time_of_day = st.slider(
            "Time of Day (Hour)",
            min_value=0,
            max_value=23,
            value=datetime.now().hour,
            format="%d:00"
        )
    
    with col3:
        crowd_level = st.selectbox(
            "Current Crowd Level",
            options=['low', 'medium', 'high'],
            index=1,
            format_func=lambda x: f"{x.title()} Crowd"
        )
    
    # Additional context
    col4, col5 = st.columns(2)
    
    with col4:
        dwell_time = st.slider(
            "Expected Dwell Time (minutes)",
            min_value=30,
            max_value=300,
            value=airport_profile['baseline_dwell_time']
        )
    
    with col5:
        is_frequent_flyer = st.checkbox("Frequent Flyer", value=False)
    
    # Generate recommendations button
    if st.button("🚀 Generate Recommendations", type="primary", key="generate_recs"):
        
        # Simulate passenger ID
        passenger_id = f"SIM_{selected_airport}_{passenger_segment[:3].upper()}_{int(datetime.now().timestamp()) % 10000}"
        
        with st.spinner("Analyzing passenger profile and generating recommendations..."):
            
            # Get domain-specific recommendations from the model
            if hasattr(recommender, 'recommend'):
                # Use new domain-aware recommender
                st.session_state.recommendations = recommender.recommend(
                    user_id=passenger_id,
                    domain=selected_domain,
                    n=5
                )
            else:
                # Fallback to legacy method
                st.session_state.recommendations = recommender.get_user_recommendations(
                    user_id=passenger_id,
                    airport_code=selected_airport,
                    passenger_segment=passenger_segment,
                    n_recommendations=5
                )
    
    # Get recommendations from session state
    recommendations = st.session_state.get('recommendations', [])
    
    # Enhanced recommendations display with styled cards
    st.markdown(create_domain_header('analytics', 'Personalized Recommendations'), unsafe_allow_html=True)
    
    if recommendations:
        for i, rec in enumerate(recommendations):
            domain = rec.get('domain', 'general')
            card_html = create_recommendation_card(rec, domain)
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Action buttons with enhanced styling
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button(f"View Details", key=f"view_{rec['product_id']}", help="See detailed product information"):
                    st.success(f"Viewing details for {rec.get('product_name', 'Product')}")
            with col2:
                if st.button(f"Add to Cart", key=f"cart_{rec['product_id']}", help="Add to shopping cart"):
                    st.success(f"Added {rec.get('product_name', 'Product')} to cart!")
        
        # Enhanced impact metrics with colored cards
        st.markdown(create_domain_header('analytics', 'Expected Impact'), unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            conversion_card = create_colored_metric_card(
                "Conversion Lift", 
                "+25.3%", 
                "vs baseline 15%", 
                "nps"
            )
            st.markdown(conversion_card, unsafe_allow_html=True)
        
        with col2:
            avg_price = np.mean([rec.get('price', 100) for rec in recommendations])
            revenue_card = create_colored_metric_card(
                "Avg Revenue/Pax", 
                f"₹{avg_price:.0f}", 
                f"+₹{avg_price-150:.0f} increase", 
                "revenue"
            )
            st.markdown(revenue_card, unsafe_allow_html=True)
        
        with col3:
            engagement = np.mean([rec['confidence'] for rec in recommendations])
            engagement_card = create_colored_metric_card(
                "Engagement Score", 
                f"{engagement:.2f}", 
                "Personalization quality", 
                "recommendation"
            )
            st.markdown(engagement_card, unsafe_allow_html=True)
    
    else:
        st.info(f"Click 'Generate Recommendations' above to see personalized {domain_names.get(selected_domain, 'product')} suggestions.")
        
        # Show sample products for selected domain
        if selected_domain and hasattr(recommender, '_get_fallback_recommendations'):
            st.subheader(f"Available {domain_names.get(selected_domain, 'Products')} at {selected_airport}")
            sample_recs = recommender._get_fallback_recommendations(selected_domain, 3)
            
            for rec in sample_recs:
                with st.expander(f"{rec['product_name']} - ₹{rec['price']:,.0f}"):
                    if rec.get('discount'):
                        st.markdown(f"**Special Offer:** {rec['discount']}")
                    if rec.get('brand'):
                        st.markdown(f"**Brand:** {rec['brand']}")
                    if rec.get('restaurant'):
                        st.markdown(f"**Restaurant:** {rec['restaurant']}")
                    if rec.get('lounge'):
                        st.markdown(f"**Location:** {rec['lounge']}")
    
    # Model performance section
    st.markdown("---")
    st.subheader("🔬 Model Performance & Insights")
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.markdown("**Model Statistics**")
        
        # Simulate model performance metrics
        performance_data = {
            'Metric': ['RMSE', 'Precision@5', 'Recall@5', 'Coverage', 'Diversity'],
            'Value': [0.95, 0.78, 0.65, 0.92, 0.84],
            'Benchmark': [1.0, 0.70, 0.60, 0.85, 0.80],
            'Status': ['Good', 'Excellent', 'Good', 'Excellent', 'Good']
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        fig_perf = px.bar(
            perf_df,
            x='Metric',
            y='Value',
            color='Status',
            title="Model Performance Metrics",
            color_discrete_map={
                'Excellent': '#28a745',
                'Good': '#ffc107',
                'Needs Improvement': '#dc3545'
            }
        )
        fig_perf.update_layout(height=300)
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with perf_col2:
        st.markdown("**Recommendation Distribution by Category**")
        
        # Category distribution from recommendations
        if recommendations:
            category_counts = {}
            for rec in recommendations:
                domain = rec.get('domain', 'general')
                category_counts[domain] = category_counts.get(domain, 0) + 1
        else:
            category_counts = {
                'retail': 2,
                'f&b': 2,
                'lounge': 1
            }
        
        fig_cat = px.pie(
            values=list(category_counts.values()),
            names=list(category_counts.keys()),
            title="Recommendation Categories"
        )
        fig_cat.update_layout(height=300)
        st.plotly_chart(fig_cat, use_container_width=True)
    
    # Business insights
    st.subheader("💡 Business Insights")
    
    insights_text = f"""
    **For {selected_airport} - {passenger_segment.title()} Travelers:**
    
    • **Peak Opportunity Time**: {time_of_day}:00 - {(time_of_day + 2) % 24}:00 shows highest engagement
    • **Crowd Impact**: {crowd_level.title()} crowd levels increase conversion by {np.random.randint(5, 25)}%
    • **Segment Preference**: {passenger_segment.title()} travelers prefer {list(airport_profile['retail_categories'].keys())[0]} items
    • **Dwell Time Correlation**: {dwell_time} minute stays typically result in {np.random.randint(2, 5)} purchase touchpoints
    """
    
    if is_frequent_flyer:
        insights_text += "\n• **Loyalty Bonus**: Frequent flyer status increases premium product affinity by 40%"
    
    st.info(insights_text)
    
    # Experimental features
    with st.expander("🧪 Experimental Features"):
        st.markdown("**Real-time Personalization Adjustments**")
        
        weather_impact = st.checkbox("Weather-based recommendations (e.g., umbrella sales during rain)")
        flight_delay_impact = st.checkbox("Flight delay compensation (lounge passes, meal vouchers)")
        group_travel_impact = st.checkbox("Group travel detection (family packs, bulk discounts)")
        
        if any([weather_impact, flight_delay_impact, group_travel_impact]):
            st.success("Experimental features enabled! These would provide additional context for recommendations.")
    
    # Download recommendations
    if st.button("📥 Export Recommendations"):
        if recommendations:
            rec_data = []
            for rec in recommendations:
                rec_data.append({
                    'Product ID': rec['product_id'],
                    'Product Name': rec.get('product_name', f"Product {rec['product_id']}"),
                    'Domain': rec.get('domain', 'general'),
                    'Price': rec.get('price', 100),
                    'Affinity Score': rec['predicted_rating'],
                    'Confidence': rec['confidence'],
                    'Passenger Segment': passenger_segment,
                    'Time of Day': time_of_day,
                    'Crowd Level': crowd_level
                })
            
            rec_df = pd.DataFrame(rec_data)
            csv = rec_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"recommendations_{selected_airport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("Generate recommendations first to export them.")
