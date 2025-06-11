"""
Domain-aware collaborative filtering recommendation engine for personalized retail offers.
Uses sklearn-based matrix factorization with domain filtering (retail, f&b, lounge).
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.airport_profiles import AIRPORT_PROFILES

class DomainRecommender:
    def __init__(self, no_components: int = 32, learning_rate: float = 0.05, loss: str = 'warp'):
        """Initialize domain-aware recommender system"""
        self.model = NMF(n_components=no_components, random_state=42, max_iter=200)
        self.user_item_matrix = None
        self.products_df = None
        self.scaler = MinMaxScaler()
        self.item_similarity = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.domains = ['retail', 'f&b', 'lounge']
        
    def fit(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame):
        """
        Train the domain-aware recommendation model
        
        Args:
            interactions_df: DataFrame with columns [user_id, item_id, rating]
            products_df: DataFrame with columns [item_id, domain, name, price, category]
        """
        # Store products with domain information
        self.products_df = products_df.copy()
        
        # Ensure domain column exists and normalize values
        if 'domain' not in products_df.columns:
            # Map categories to domains
            domain_mapping = {
                'Food & Beverage': 'f&b',
                'Coffee': 'f&b',
                'Restaurant': 'f&b',
                'Snacks': 'f&b',
                'Retail': 'retail',
                'Electronics': 'retail',
                'Books': 'retail',
                'Souvenirs': 'retail',
                'Lounge': 'lounge',
                'Premium Lounge': 'lounge'
            }
            self.products_df['domain'] = products_df['category'].map(
                lambda x: domain_mapping.get(x, 'retail')
            )
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # Create mappings
        self.user_mapping = {user: idx for idx, user in enumerate(self.user_item_matrix.index)}
        self.item_mapping = {item: idx for idx, item in enumerate(self.user_item_matrix.columns)}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Scale the ratings
        user_item_scaled = self.scaler.fit_transform(self.user_item_matrix.values)
        
        # Train NMF model
        self.model.fit(user_item_scaled)
        
        # Calculate item similarity matrix for domain filtering
        item_features = self.model.components_.T
        self.item_similarity = cosine_similarity(item_features)
        
        print(f"Model trained with {len(self.user_item_matrix)} users and {len(self.user_item_matrix.columns)} products")
        return self
    
    def recommend(self, user_id: str, domain: str, n: int = 5) -> List[Dict]:
        """
        Get domain-filtered recommendations for a user
        
        Args:
            user_id: External user ID
            domain: Target domain ('retail', 'f&b', 'lounge')
            n: Number of recommendations
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            if self.model is None or self.user_item_matrix is None:
                return self._get_fallback_recommendations(domain, n)
            
            # Get or create user profile
            if user_id in self.user_mapping:
                user_idx = self.user_mapping[user_id]
                user_ratings = self.user_item_matrix.iloc[user_idx].values
            else:
                # New user - use average ratings
                user_ratings = np.mean(self.user_item_matrix.values, axis=0)
            
            # Transform user ratings
            user_scaled = self.scaler.transform([user_ratings])
            
            # Get user factors
            user_factors = self.model.transform(user_scaled)
            
            # Predict ratings for all items
            predicted_ratings = np.dot(user_factors, self.model.components_)[0]
            
            # Filter items by domain
            domain_items = self.products_df[
                self.products_df['domain'].str.lower() == domain.lower()
            ]['item_id'].tolist()
            
            # Get recommendations for domain items only
            recommendations = []
            item_scores = []
            
            for item_id in domain_items:
                if item_id in self.item_mapping:
                    item_idx = self.item_mapping[item_id]
                    score = predicted_ratings[item_idx]
                    
                    # Skip if user already rated highly
                    if user_ratings[item_idx] < 3.0:
                        item_scores.append((item_id, score, item_idx))
            
            # Sort by predicted rating and take top n
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (item_id, score, item_idx) in enumerate(item_scores[:n]):
                # Get product details
                product_info = self.products_df[
                    self.products_df['item_id'] == item_id
                ].iloc[0]
                
                recommendations.append({
                    'product_id': item_id,
                    'predicted_rating': float(score),
                    'confidence': min(1.0, abs(score) / 5.0),
                    'product_name': product_info.get('name', f'Product {item_id}'),
                    'category': product_info.get('category', domain.title()),
                    'domain': domain,
                    'price': float(product_info.get('price', 100)),
                    'rank': i + 1
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return self._get_fallback_recommendations(domain, n)
    
    def _get_fallback_recommendations(self, domain: str, n: int) -> List[Dict]:
        """Fallback recommendations when model fails"""
        fallback_by_domain = {
            'retail': [
                {'product_id': 'RET001', 'name': 'Travel Accessories', 'price': 800},
                {'product_id': 'RET002', 'name': 'Electronics Bundle', 'price': 1200},
                {'product_id': 'RET003', 'name': 'Local Souvenirs', 'price': 500},
                {'product_id': 'RET004', 'name': 'Books & Magazines', 'price': 300},
                {'product_id': 'RET005', 'name': 'Fashion Accessories', 'price': 600}
            ],
            'f&b': [
                {'product_id': 'FB001', 'name': 'Premium Coffee', 'price': 150},
                {'product_id': 'FB002', 'name': 'Gourmet Sandwich', 'price': 250},
                {'product_id': 'FB003', 'name': 'Local Cuisine', 'price': 400},
                {'product_id': 'FB004', 'name': 'Fresh Juice', 'price': 120},
                {'product_id': 'FB005', 'name': 'Artisan Pastries', 'price': 180}
            ],
            'lounge': [
                {'product_id': 'LNG001', 'name': 'Premium Lounge Access', 'price': 2000},
                {'product_id': 'LNG002', 'name': 'Spa Services', 'price': 1500},
                {'product_id': 'LNG003', 'name': 'Business Center', 'price': 500},
                {'product_id': 'LNG004', 'name': 'Private Meeting Room', 'price': 1000},
                {'product_id': 'LNG005', 'name': 'Shower Facilities', 'price': 300}
            ]
        }
        
        domain_products = fallback_by_domain.get(domain.lower(), fallback_by_domain['retail'])
        recommendations = []
        
        for i, product in enumerate(domain_products[:n]):
            recommendations.append({
                'product_id': product['product_id'],
                'predicted_rating': 4.0 - (i * 0.2),
                'confidence': 0.8 - (i * 0.1),
                'product_name': product['name'],
                'category': domain.title(),
                'domain': domain,
                'price': product['price'],
                'rank': i + 1
            })
        
        return recommendations
    
    def get_domain_products(self, domain: str) -> pd.DataFrame:
        """Get all products in a specific domain"""
        if self.products_df is not None:
            return self.products_df[
                self.products_df['domain'].str.lower() == domain.lower()
            ].copy()
        return pd.DataFrame()
    
    def calculate_conversion_lift(self, recommendations: List[Dict], baseline_conversion: float = 0.15) -> Dict:
        """Calculate expected conversion lift from recommendations"""
        if not recommendations:
            return {
                'baseline_conversion': baseline_conversion,
                'predicted_conversion': baseline_conversion,
                'conversion_lift': 0.0,
                'expected_revenue_uplift': 0.0,
                'recommendation_count': 0
            }
        
        # Calculate weighted conversion based on confidence scores
        avg_confidence = sum(r['confidence'] for r in recommendations) / len(recommendations)
        predicted_conversion = baseline_conversion * (1 + avg_confidence)
        
        conversion_lift = (predicted_conversion - baseline_conversion) / baseline_conversion * 100
        
        # Calculate expected revenue uplift
        avg_price = sum(r['price'] for r in recommendations) / len(recommendations)
        expected_revenue_uplift = (predicted_conversion - baseline_conversion) * avg_price
        
        return {
            'baseline_conversion': baseline_conversion,
            'predicted_conversion': round(predicted_conversion, 3),
            'conversion_lift': round(conversion_lift, 1),
            'expected_revenue_uplift': round(expected_revenue_uplift, 2),
            'avg_recommendation_price': round(avg_price, 2),
            'recommendation_count': len(recommendations)
        }
    
    def save(self, path: str):
        """Save trained model and dataset"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model_data = {
                'model': self.model,
                'user_item_matrix': self.user_item_matrix,
                'products_df': self.products_df,
                'scaler': self.scaler,
                'item_similarity': self.item_similarity,
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'reverse_item_mapping': self.reverse_item_mapping
            }
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    @classmethod
    def load(cls, path: str):
        """Load trained model and dataset"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            instance = cls()
            instance.model = model_data['model']
            instance.user_item_matrix = model_data['user_item_matrix']
            instance.products_df = model_data['products_df']
            instance.scaler = model_data['scaler']
            instance.item_similarity = model_data['item_similarity']
            instance.user_mapping = model_data['user_mapping']
            instance.item_mapping = model_data['item_mapping']
            instance.reverse_item_mapping = model_data['reverse_item_mapping']
            
            print(f"Model loaded from {path}")
            return instance
        except Exception as e:
            print(f"Error loading model: {e}")
            return cls()

# Legacy compatibility class
class AirportRecommender(DomainRecommender):
    """Legacy wrapper for backward compatibility"""
    
    def __init__(self):
        super().__init__()
    
    def train_model(self):
        """Legacy method for training"""
        # Generate sample data if no real data exists
        interactions_df = self._generate_sample_interactions()
        products_df = self._generate_sample_products()
        return self.fit(interactions_df, products_df)
    
    def get_user_recommendations(self, user_id: str, airport_code: str, passenger_segment: str, n_recommendations: int = 5):
        """Legacy method for getting recommendations"""
        # Default to retail domain for legacy compatibility
        return self.recommend(user_id, 'retail', n_recommendations)
    
    def _generate_sample_interactions(self):
        """Generate sample interaction data"""
        np.random.seed(42)
        
        passengers = [f'P{i:04d}' for i in range(1, 501)]
        products = [f'PROD_{i:03d}' for i in range(1, 51)]
        
        data = []
        for passenger in passengers:
            n_purchases = np.random.randint(1, 6)
            purchased_products = np.random.choice(products, n_purchases, replace=False)
            
            for product in purchased_products:
                rating = np.random.uniform(1, 5)
                data.append({
                    'user_id': passenger,
                    'item_id': product,
                    'rating': rating
                })
        
        return pd.DataFrame(data)
    
    def _generate_sample_products(self):
        """Generate sample product data with domains"""
        np.random.seed(42)
        
        products = [f'PROD_{i:03d}' for i in range(1, 51)]
        categories = ['Food & Beverage', 'Retail', 'Electronics', 'Books', 'Souvenirs', 'Lounge']
        domains = ['f&b', 'retail', 'retail', 'retail', 'retail', 'lounge']
        
        data = []
        for i, product_id in enumerate(products):
            category_idx = i % len(categories)
            data.append({
                'item_id': product_id,
                'name': f'Product {i+1}',
                'category': categories[category_idx],
                'domain': domains[category_idx],
                'price': np.random.uniform(50, 500)
            })
        
        return pd.DataFrame(data)