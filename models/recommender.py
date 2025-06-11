"""
Collaborative filtering recommendation engine for personalized retail offers.
Uses sklearn-based matrix factorization and similarity-based recommendations.
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.airport_profiles import AIRPORT_PROFILES

class AirportRecommender:
    def __init__(self):
        self.model = None
        self.user_item_matrix = None
        self.products_df = None
        self.scaler = MinMaxScaler()
        self.item_similarity = None
        self.n_components = 10
        
    def load_data(self):
        """Load transaction and product data"""
        try:
            # Load transaction data
            transactions_df = pd.read_csv('data/transactions.csv')
            products_df = pd.read_csv('data/products.csv')
            
            # Create user-item matrix with implicit ratings
            # Use transaction frequency and amount as rating signal
            user_item = transactions_df.groupby(['passenger_id', 'product_id']).agg({
                'quantity': 'sum',
                'total_amount': 'sum'
            }).reset_index()
            
            # Normalize ratings to 1-5 scale
            user_item['rating'] = np.clip(
                (user_item['quantity'] * user_item['total_amount'] / 1000), 1, 5
            )
            
            self.products_df = products_df
            return user_item[['passenger_id', 'product_id', 'rating']]
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample data if files don't exist"""
        np.random.seed(42)
        
        # Generate sample transactions
        passengers = [f'P{i:04d}' for i in range(1, 501)]
        products = [f'PROD_{i:03d}' for i in range(1, 51)]
        
        data = []
        for passenger in passengers:
            # Each passenger buys 1-5 products
            n_purchases = np.random.randint(1, 6)
            purchased_products = np.random.choice(products, n_purchases, replace=False)
            
            for product in purchased_products:
                rating = np.random.uniform(1, 5)
                data.append({
                    'passenger_id': passenger,
                    'product_id': product,
                    'rating': rating
                })
        
        # Generate sample products
        categories = ['Food & Beverage', 'Retail', 'Electronics', 'Books', 'Souvenirs']
        self.products_df = pd.DataFrame({
            'product_id': products,
            'name': [f'Product {i}' for i in range(1, 51)],
            'category': np.random.choice(categories, len(products)),
            'price': np.random.uniform(50, 500, len(products))
        })
        
        return pd.DataFrame(data)
    
    def train_model(self):
        """Train the recommendation model"""
        # Load data
        ratings_df = self.load_data()
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot(
            index='passenger_id', 
            columns='product_id', 
            values='rating'
        ).fillna(0)
        
        # Scale the ratings
        user_item_scaled = self.scaler.fit_transform(self.user_item_matrix)
        
        # Train NMF model
        self.model = NMF(n_components=self.n_components, random_state=42)
        self.model.fit(user_item_scaled)
        
        # Calculate item similarity matrix
        item_features = self.model.components_.T
        self.item_similarity = cosine_similarity(item_features)
        
        print(f"Model trained with {len(self.user_item_matrix)} users and {len(self.user_item_matrix.columns)} products")
        return True
    
    def get_user_recommendations(self, user_id, airport_code, passenger_segment, n_recommendations=5):
        """Get personalized recommendations for a user"""
        try:
            if self.model is None:
                self.train_model()
            
            # If user doesn't exist, create new user profile
            if user_id not in self.user_item_matrix.index:
                user_ratings = np.zeros(len(self.user_item_matrix.columns))
            else:
                user_ratings = self.user_item_matrix.loc[user_id].values
            
            # Transform user ratings
            user_scaled = self.scaler.transform([user_ratings])
            
            # Get user factors
            user_factors = self.model.transform(user_scaled)
            
            # Predict ratings for all items
            predicted_ratings = np.dot(user_factors, self.model.components_)[0]
            
            # Get top recommendations
            product_ids = self.user_item_matrix.columns.tolist()
            recommendations = []
            
            # Sort by predicted rating
            sorted_indices = np.argsort(predicted_ratings)[::-1]
            
            count = 0
            for idx in sorted_indices:
                if count >= n_recommendations:
                    break
                    
                # Skip items user has already rated highly
                if user_ratings[idx] < 3.0:  # Only recommend if not already purchased/rated
                    product_id = product_ids[idx]
                    predicted_rating = predicted_ratings[idx]
                    
                    # Get product details
                    product_info = self.products_df[
                        self.products_df['product_id'] == product_id
                    ].iloc[0] if len(self.products_df) > 0 else {'name': f'Product {product_id}', 'category': 'General'}
                    
                    recommendations.append({
                        'product_id': product_id,
                        'predicted_rating': float(predicted_rating),
                        'confidence': min(1.0, abs(predicted_rating) / 5.0),
                        'product_name': product_info.get('name', f'Product {product_id}'),
                        'category': product_info.get('category', 'General'),
                        'price': product_info.get('price', 100),
                        'airport_code': airport_code,
                        'passenger_segment': passenger_segment
                    })
                    count += 1
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return self._get_fallback_recommendations(airport_code, passenger_segment, n_recommendations)
    
    def _get_fallback_recommendations(self, airport_code, passenger_segment, n_recommendations):
        """Fallback recommendations when model fails"""
        fallback_products = [
            {'product_id': 'COFFEE001', 'name': 'Premium Coffee', 'category': 'Food & Beverage', 'price': 150},
            {'product_id': 'SNACK001', 'name': 'Local Snacks', 'category': 'Food & Beverage', 'price': 200},
            {'product_id': 'BOOK001', 'name': 'Travel Guide', 'category': 'Books', 'price': 300},
            {'product_id': 'SOUVENIR001', 'name': 'Local Souvenirs', 'category': 'Souvenirs', 'price': 500},
            {'product_id': 'ELECTRONICS001', 'name': 'Travel Accessories', 'category': 'Electronics', 'price': 800}
        ]
        
        recommendations = []
        for i, product in enumerate(fallback_products[:n_recommendations]):
            recommendations.append({
                'product_id': product['product_id'],
                'predicted_rating': 4.0 - (i * 0.2),
                'confidence': 0.8 - (i * 0.1),
                'product_name': product['name'],
                'category': product['category'],
                'price': product['price'],
                'airport_code': airport_code,
                'passenger_segment': passenger_segment
            })
        
        return recommendations
    
    def get_product_similarity(self, product_id, n_similar=5):
        """Get similar products using item-based collaborative filtering"""
        try:
            if self.item_similarity is None:
                return []
            
            product_ids = self.user_item_matrix.columns.tolist()
            if product_id not in product_ids:
                return []
            
            product_idx = product_ids.index(product_id)
            similarities = self.item_similarity[product_idx]
            
            # Get most similar products
            similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]  # Exclude self
            
            similar_products = []
            for idx in similar_indices:
                similar_products.append({
                    'product_id': product_ids[idx],
                    'similarity_score': float(similarities[idx])
                })
            
            return similar_products
            
        except Exception as e:
            print(f"Error getting similar products: {e}")
            return []
    
    def calculate_conversion_lift(self, recommendations, baseline_conversion=0.15):
        """Calculate expected conversion lift from recommendations"""
        if not recommendations:
            return {
                'baseline_conversion': baseline_conversion,
                'predicted_conversion': baseline_conversion,
                'conversion_lift': 0.0,
                'expected_revenue_uplift': 0.0
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
    
    def save_model(self, filepath='models/recommender_model.pkl'):
        """Save trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                'model': self.model,
                'user_item_matrix': self.user_item_matrix,
                'scaler': self.scaler,
                'item_similarity': self.item_similarity,
                'products_df': self.products_df
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath='models/recommender_model.pkl'):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.user_item_matrix = model_data['user_item_matrix']
            self.scaler = model_data['scaler']
            self.item_similarity = model_data['item_similarity']
            self.products_df = model_data['products_df']
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False