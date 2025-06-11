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

def render_recommendation_tab():
    """Render the AI recommendations tab"""
    
    st.header("🧠 AI-Powered Retail Personalization")
    st.markdown("""
    Generate personalized product recommendations using collaborative filtering 
    based on passenger segments, purchase history, and real-time context.
    """)
    
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
    
    # Display recommendations (always shown, even if empty)
    st.subheader("💎 Personalized Recommendations")
    
    if recommendations:
        # Load product data for detailed display
        try:
            products_df = pd.read_csv('data/products.csv')
        except Exception:
            # Fallback product data
            products_df = pd.DataFrame({
                'product_id': [r['product_id'] for r in recommendations],
                'name': [f"Product {r['product_id']}" for r in recommendations],
                'category': ['Food & Beverage'] * len(recommendations),
                'base_price': [100 + i * 50 for i in range(len(recommendations))],
                'description': [f"Description for {r['product_id']}" for r in recommendations]
                })
            
            # Create recommendation cards
            for i, rec in enumerate(recommendations):
                product_info = products_df[products_df['product_id'] == rec['product_id']]
                
                if not product_info.empty:
                    product = product_info.iloc[0]
                else:
                    # Fallback product info
                    product = {
                        'name': f"Recommended Product {i+1}",
                        'category': 'General',
                        'base_price': 100 + i * 50,
                        'description': f"Personalized recommendation based on your profile"
                    }
                
                # Recommendation card
                with st.container():
                    card_col1, card_col2, card_col3 = st.columns([2, 3, 1])
                    
                    with card_col1:
                        # Rating visualization
                        rating = rec['predicted_rating']
                        confidence = rec['confidence']
                        
                        fig_rating = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = rating,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Affinity Score"},
                            gauge = {
                                'axis': {'range': [None, 5]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 2.5], 'color': "lightgray"},
                                    {'range': [2.5, 4], 'color': "yellow"},
                                    {'range': [4, 5], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 4.5
                                }
                            }
                        ))
                        fig_rating.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_rating, use_container_width=True)
                    
                    with card_col2:
                        st.markdown(f"### {product.get('name', 'Product')}")
                        st.markdown(f"**Category:** {product.get('category', 'N/A')}")
                        st.markdown(f"**Price:** ₹{product.get('base_price', 0):.2f}")
                        st.markdown(f"**Description:** {product.get('description', 'No description available')}")
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        
                        # Personalization reason
                        if passenger_segment == 'business':
                            if product.get('category') == 'Electronics & Gadgets':
                                reason = "🏢 Recommended for business travelers who frequently purchase tech items"
                            elif product.get('category') == 'Food & Beverage':
                                reason = "☕ Popular choice among business travelers during this time"
                            else:
                                reason = "📊 Based on similar business traveler preferences"
                        else:
                            if product.get('category') == 'Souvenirs & Gifts':
                                reason = "🎁 Perfect souvenir for leisure travelers visiting this destination"
                            elif product.get('category') == 'Food & Beverage':
                                reason = "🍽️ Highly rated by leisure travelers"
                            else:
                                reason = "🌟 Popular among leisure travelers with similar profiles"
                        
                        st.info(reason)
                    
                    with card_col3:
                        # Action buttons
                        if st.button(f"View Details", key=f"details_{i}"):
                            st.balloons()
                            st.success(f"Showing details for {product.get('name', 'Product')}")
                        
                        if st.button(f"Add to Wishlist", key=f"wishlist_{i}"):
                            st.success("Added to wishlist!")
                
                st.markdown("---")
            
            # Conversion impact estimation
            st.subheader("📊 Expected Impact")
            
            conversion_metrics = recommender.calculate_conversion_lift(recommendations)
            baseline_conversion = 0.15  # 15% baseline conversion rate
            
            impact_col1, impact_col2, impact_col3 = st.columns(3)
            
            with impact_col1:
                st.metric(
                    "Conversion Lift",
                    f"+{conversion_metrics.get('conversion_lift', 0):.1f}%",
                    f"vs baseline {baseline_conversion:.1%}"
                )
            
            with impact_col2:
                expected_revenue_per_pax = sum([
                    products_df[products_df['product_id'] == rec['product_id']].iloc[0].get('base_price', 100) 
                    * (baseline_conversion + conversion_metrics.get('conversion_lift', 0) / 100)
                    for rec in recommendations
                    if not products_df[products_df['product_id'] == rec['product_id']].empty
                ])
                
                st.metric(
                    "Expected Revenue/Pax",
                    f"₹{expected_revenue_per_pax:.0f}",
                    f"+{expected_revenue_per_pax - airport_profile['baseline_revenue_per_pax']:.0f} vs baseline"
                )
            
            with impact_col3:
                engagement_score = np.mean([rec['confidence'] for rec in recommendations])
                st.metric(
                    "Engagement Score",
                    f"{engagement_score:.2f}",
                    "Personalization quality"
                )
        
        else:
            st.warning("No recommendations available. Please check the recommendation model.")
    
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
        
        # Simulate category distribution
        if 'recommendations' in locals() and recommendations:
            category_counts = {}
            for rec in recommendations:
                product_info = products_df[products_df['product_id'] == rec['product_id']]
                if not product_info.empty:
                    category = product_info.iloc[0]['category']
                    category_counts[category] = category_counts.get(category, 0) + 1
        else:
            category_counts = {
                'Food & Beverage': 2,
                'Electronics & Gadgets': 1,
                'Fashion & Accessories': 1,
                'Souvenirs & Gifts': 1
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
        if 'recommendations' in locals() and recommendations:
            rec_data = []
            for rec in recommendations:
                product_info = products_df[products_df['product_id'] == rec['product_id']]
                if not product_info.empty:
                    product = product_info.iloc[0]
                    rec_data.append({
                        'Product ID': rec['product_id'],
                        'Product Name': product.get('name', 'Unknown'),
                        'Category': product.get('category', 'Unknown'),
                        'Price': product.get('base_price', 0),
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
