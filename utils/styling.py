"""
Advanced styling utilities for AeroNexus AI platform.
Provides dark mode themes, custom CSS, and unique domain icons.
"""

import streamlit as st

def apply_custom_css():
    """Apply comprehensive dark mode styling with enhanced visual effects"""
    
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
        
        /* Global Dark Mode Styling */
        .stApp {
            background: linear-gradient(135deg, #0E1117 0%, #1A1D29 50%, #262730 100%);
            color: #FAFAFA;
        }
        
        /* Custom Headers with Gradient */
        h1, h2, h3 {
            font-family: 'Orbitron', monospace !important;
            background: linear-gradient(45deg, #FF6B35, #F7931E, #FFD700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            text-shadow: 0 0 20px rgba(255, 107, 53, 0.3);
        }
        
        /* Body Text */
        p, div, span, label {
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 400;
        }
        
        /* Sidebar Enhancement */
        .css-1d391kg {
            background: linear-gradient(180deg, #1A1D29 0%, #262730 100%);
            border-right: 2px solid #FF6B35;
        }
        
        /* Metric Cards with Neon Glow */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #262730 0%, #1A1D29 100%);
            border: 1px solid #FF6B35;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 0 20px rgba(255, 107, 53, 0.2);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(255, 107, 53, 0.4);
        }
        
        /* Revenue Cards - Gold Theme */
        .revenue-card {
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            color: #000 !important;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 0 25px rgba(255, 215, 0, 0.3);
        }
        
        /* NPS Cards - Green Theme */
        .nps-card {
            background: linear-gradient(135deg, #00FF87 0%, #60EFFF 100%);
            color: #000 !important;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 0 25px rgba(0, 255, 135, 0.3);
        }
        
        /* Recommendation Cards - Purple Theme */
        .recommendation-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #FFF !important;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 0 25px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        
        .recommendation-card:hover {
            transform: scale(1.02);
            box-shadow: 0 0 35px rgba(102, 126, 234, 0.5);
        }
        
        /* Retail Cards - Orange Theme */
        .retail-card {
            background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
            color: #000 !important;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 0 25px rgba(255, 126, 95, 0.3);
        }
        
        /* F&B Cards - Red Theme */
        .fb-card {
            background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
            color: #000 !important;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 0 25px rgba(255, 107, 107, 0.3);
        }
        
        /* Lounge Cards - Blue Theme */
        .lounge-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: #000 !important;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 0 25px rgba(79, 172, 254, 0.3);
        }
        
        /* Button Enhancements */
        .stButton > button {
            background: linear-gradient(45deg, #FF6B35, #F7931E);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 107, 53, 0.5);
        }
        
        /* Selectbox and Input Styling */
        .stSelectbox > div > div {
            background: linear-gradient(135deg, #262730 0%, #1A1D29 100%);
            border: 1px solid #FF6B35;
            border-radius: 10px;
        }
        
        /* Progress Bars */
        .stProgress > div > div {
            background: linear-gradient(90deg, #FF6B35, #F7931E, #FFD700);
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #262730 0%, #1A1D29 100%);
            border: 1px solid #FF6B35;
            border-radius: 10px;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(135deg, #262730 0%, #1A1D29 100%);
            color: #FAFAFA;
            border-radius: 10px 10px 0 0;
            font-family: 'Orbitron', monospace;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(45deg, #FF6B35, #F7931E);
            color: white;
        }
        
        /* Chart Container Styling */
        .js-plotly-plot {
            background: rgba(38, 39, 48, 0.8) !important;
            border-radius: 15px;
            border: 1px solid #FF6B35;
        }
        
        /* Success/Info/Warning Messages */
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid #FF6B35;
            background: rgba(38, 39, 48, 0.9);
        }
        
        /* Sidebar Logo Area */
        .sidebar-logo {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
            border-radius: 15px;
            margin-bottom: 20px;
        }
        
        /* Animation for Loading */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 2s infinite;
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1A1D29;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #FF6B35, #F7931E);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #F7931E, #FFD700);
        }
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)

def get_domain_icon(domain):
    """Get unique SVG icons for each domain"""
    
    icons = {
        'retail': """
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M7 4V2C7 1.45 7.45 1 8 1H16C16.55 1 17 1.45 17 2V4H20C20.55 4 21 4.45 21 5S20.55 6 20 6H19V19C19 20.1 18.1 21 17 21H7C5.9 21 5 20.1 5 19V6H4C3.45 6 3 5.55 3 5S3.45 4 4 4H7ZM9 3V4H15V3H9ZM7 6V19H17V6H7Z" fill="#FF6B35"/>
            <circle cx="10" cy="12" r="1.5" fill="#FFD700"/>
            <circle cx="14" cy="12" r="1.5" fill="#FFD700"/>
            <path d="M8 15H16V16H8V15Z" fill="#F7931E"/>
        </svg>
        """,
        'f&b': """
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8.1 13.34L12 17.23L15.9 13.34C17.28 11.96 17.28 9.73 15.9 8.35C14.52 6.97 12.29 6.97 10.91 8.35C9.53 9.73 9.53 11.96 8.1 13.34Z" fill="#FF6B35"/>
            <circle cx="12" cy="11" r="2" fill="#FFD700"/>
            <path d="M12 2L13.5 6.5H18L14.5 9.5L16 14L12 11L8 14L9.5 9.5L6 6.5H10.5L12 2Z" fill="#F7931E"/>
            <path d="M4 20H20V22H4V20Z" fill="#FF6B35"/>
        </svg>
        """,
        'lounge': """
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M21 9V7L17 3H7L3 7V9C3 9.55 3.45 10 4 10V16C4 16.55 4.45 17 5 17H6C6.55 17 7 16.55 7 16V15H17V16C17 16.55 17.45 17 18 17H19C19.55 17 20 16.55 20 16V10C20.55 10 21 9.55 21 9Z" fill="#FF6B35"/>
            <circle cx="8" cy="11.5" r="1.5" fill="#FFD700"/>
            <circle cx="16" cy="11.5" r="1.5" fill="#FFD700"/>
            <path d="M9 20H15V22H9V20Z" fill="#F7931E"/>
        </svg>
        """,
        'airport': """
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M21 16V14L13.5 7V3.5C13.5 2.67 12.83 2 12 2S10.5 2.67 10.5 3.5V7L3 14V16L10.5 13.5V19L8.5 20.5V22L12 21L15.5 22V20.5L13.5 19V13.5L21 16Z" fill="#FF6B35"/>
            <circle cx="12" cy="4" r="1" fill="#FFD700"/>
            <path d="M12 8L16 12H8L12 8Z" fill="#F7931E"/>
        </svg>
        """,
        'analytics': """
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 13H5V21H3V13ZM7 9H9V21H7V9ZM11 5H13V21H11V5ZM15 1H17V21H15V1ZM19 9H21V21H19V9Z" fill="#FF6B35"/>
            <circle cx="4" cy="12" r="1" fill="#FFD700"/>
            <circle cx="8" cy="8" r="1" fill="#FFD700"/>
            <circle cx="12" cy="4" r="1" fill="#FFD700"/>
            <circle cx="16" cy="2" r="1" fill="#FFD700"/>
            <circle cx="20" cy="8" r="1" fill="#FFD700"/>
        </svg>
        """,
        'pricing': """
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L14.5 9.5H22L16.5 14L19 21.5L12 17L5 21.5L7.5 14L2 9.5H9.5L12 2Z" fill="#FF6B35"/>
            <circle cx="12" cy="12" r="3" fill="#FFD700"/>
            <text x="12" y="15" text-anchor="middle" fill="#000" font-size="8" font-weight="bold">₹</text>
        </svg>
        """
    }
    
    return icons.get(domain, icons['airport'])

def create_colored_metric_card(title, value, delta, color_theme="default"):
    """Create colored metric cards with custom themes"""
    
    theme_classes = {
        'revenue': 'revenue-card',
        'nps': 'nps-card', 
        'recommendation': 'recommendation-card',
        'retail': 'retail-card',
        'fb': 'fb-card',
        'lounge': 'lounge-card',
        'default': 'metric-container'
    }
    
    card_class = theme_classes.get(color_theme, 'metric-container')
    
    card_html = f"""
    <div class="{card_class}" style="margin: 10px 0;">
        <h3 style="margin: 0 0 10px 0; font-size: 18px;">{title}</h3>
        <h2 style="margin: 0; font-size: 28px; font-weight: 700;">{value}</h2>
        <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.8;">{delta}</p>
    </div>
    """
    
    return card_html

def create_domain_header(domain, title):
    """Create stylized headers with domain icons"""
    
    icon = get_domain_icon(domain)
    
    header_html = f"""
    <div style="display: flex; align-items: center; margin: 20px 0;">
        {icon}
        <h2 style="margin-left: 15px; font-family: 'Orbitron', monospace;">{title}</h2>
    </div>
    """
    
    return header_html

def create_recommendation_card(recommendation, domain):
    """Create styled recommendation cards with domain-specific theming"""
    
    theme_map = {
        'retail': 'retail-card',
        'f&b': 'fb-card', 
        'lounge': 'lounge-card'
    }
    
    card_class = theme_map.get(domain, 'recommendation-card')
    icon = get_domain_icon(domain)
    
    product_name = recommendation.get('product_name', f"Product {recommendation['product_id']}")
    price = recommendation.get('price', 100)
    discount = recommendation.get('discount', '')
    brand = recommendation.get('brand', '')
    restaurant = recommendation.get('restaurant', '')
    lounge = recommendation.get('lounge', '')
    
    card_html = f"""
    <div class="{card_class}" style="margin: 15px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            {icon}
            <h3 style="margin-left: 15px; font-size: 20px;">{product_name}</h3>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <p><strong>Rating:</strong> {recommendation['predicted_rating']:.1f}/5.0</p>
                <p><strong>Match:</strong> {recommendation['confidence']:.0%}</p>
                <p><strong>Price:</strong> ₹{price:,.0f}</p>
            </div>
            <div>
                {f'<p><strong>Offer:</strong> {discount}</p>' if discount else ''}
                {f'<p><strong>Brand:</strong> {brand}</p>' if brand else ''}
                {f'<p><strong>Restaurant:</strong> {restaurant}</p>' if restaurant else ''}
                {f'<p><strong>Location:</strong> {lounge}</p>' if lounge else ''}
            </div>
        </div>
    </div>
    """
    
    return card_html

def apply_chart_styling():
    """Apply consistent styling to plotly charts"""
    
    return {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(38,39,48,0.8)',
        'font': {'color': '#FAFAFA', 'family': 'Rajdhani'},
        'title': {'font': {'color': '#FF6B35', 'size': 18, 'family': 'Orbitron'}},
        'xaxis': {'gridcolor': 'rgba(255,107,53,0.2)', 'color': '#FAFAFA'},
        'yaxis': {'gridcolor': 'rgba(255,107,53,0.2)', 'color': '#FAFAFA'}
    }