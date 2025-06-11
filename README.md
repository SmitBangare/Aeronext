# 🛫 AeroNexus AI - Airport Revenue Optimization Platform

An AI-driven simulation platform for non-aeronautical revenue uplift and passenger experience optimization, specifically designed for the **AeroNext Phase 5 Pilot Proposal** targeting Delhi (DEL) and Jaipur (JAI) airports.

## 📋 Project Overview

This Streamlit-based dashboard demonstrates how advanced AI and IoT solutions can unlock new revenue streams and deliver seamless, engaging experiences for airport travelers. The platform implements real AI models for personalization, dynamic pricing, and sentiment analysis to achieve measurable business outcomes.

### 🎯 Pilot Objectives (90-Day PoC)

| **Objective** | **Baseline** | **Target** | **Impact** |
|---------------|--------------|------------|------------|
| **Revenue per Passenger** | ₹312 → ₹390 | 25% uplift | ₹78 additional revenue/pax |
| **Net Promoter Score** | 52 → 65+ | +10 points | Enhanced passenger satisfaction |
| **Dwell Time Variance** | ±22 min → ±15 min | 30% reduction | Improved operational efficiency |

## 🏢 Target Airports

### 🛫 Delhi Indira Gandhi International (DEL)
- **Profile**: Tier-1 Gateway, 70M+ passengers annually
- **Segment**: 55% business, 45% leisure travelers
- **Focus**: Advanced AI personalization, premium service optimization
- **Technology**: High-complexity implementation with full IoT integration

### ✈️ Jaipur International (JAI)
- **Profile**: Tier-2 Regional Hub, 6.8M+ passengers annually  
- **Segment**: 30% business, 70% leisure travelers
- **Focus**: Cost-effective scaling, tourism-oriented offerings
- **Technology**: Streamlined implementation, regional connectivity focus

## 🧠 AI Models & Technology Stack

### 1. **Recommendation Engine** 🎯
- **Technology**: Collaborative filtering using Surprise library
- **Purpose**: Personalized retail and F&B recommendations
- **Business Impact**: 15% conversion lift, enhanced passenger engagement
- **Implementation**: Matrix factorization with segment-based business rules

### 2. **Dynamic Pricing Engine** 💰
- **Technology**: Q-Learning reinforcement learning
- **Purpose**: Real-time price optimization based on demand and context
- **Business Impact**: 12-18% revenue uplift vs static pricing
- **Implementation**: State-action optimization with business constraints

### 3. **Sentiment Analysis & NPS** 💬
- **Technology**: TextBlob NLP with domain-specific enhancements
- **Purpose**: Real-time feedback analysis and NPS calculation
- **Business Impact**: +10 point NPS improvement, proactive issue resolution
- **Implementation**: Aspect-based sentiment with predictive NPS modeling

## 📊 Dashboard Features

### 🏁 **Overview Tab**
- Airport profile comparison (DEL vs JAI)
- Baseline metrics and target progress tracking
- Passenger segment analysis and retail category mix
- Commercial zone utilization and peak hour patterns

### 🧠 **AI Recommendations Tab**
- Interactive passenger profiling (segment, time, crowd level)
- Real-time personalized product recommendations
- Conversion impact estimation and engagement scoring
- Model performance metrics and business insights

### 💸 **Dynamic Pricing Tab**
- Market condition inputs (demand, time, crowd level)
- Real-time pricing recommendations with AI reasoning
- 24-hour pricing simulation and revenue impact analysis
- Category-specific elasticity modeling and optimization windows

### 💬 **Sentiment & NPS Tab**
- Real-time feedback analysis with sentiment scoring
- NPS calculation and trend visualization
- Aspect-based feedback analysis (service, shopping, facilities)
- 30-day NPS prediction and actionable insights

### ⚖️ **DEL vs JAI Comparison Tab**
- Side-by-side airport performance comparison
- AI model effectiveness across different airport tiers
- ROI analysis and implementation complexity assessment
- Strategic recommendations for scaling and optimization

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Streamlit
- Required ML libraries (pandas, numpy, scikit-surprise, textblob, plotly)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd aeronexus-ai
