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
        users = [f"USER_{i:03d}" for i in range(1, 101)]
        products = [f"PROD_{i:03d}" for i in range(1, 31)]
        
        data = []
        for user in users:
            # Each user rates 5-15 products
            n_ratings = np.random.randint(5, 16)
            user_products = np.random.choice(products, n_ratings, replace=False)
            for product in user_products:
                rating = np.random.randint(1, 6)
                data.append([user, product, rating])
        
        return pd.DataFrame(data, columns=['passenger_id', 'product_id', 'rating'])
    
    def train_model(self):
        """Train the recommendation model"""
        # Load data
        ratings_df = self.load_data()
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df, reader)
        
        # Split data
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Train SVD model for matrix factorization
        self.model = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, random_state=42)
        self.model.fit(self.trainset)
        
        # Train KNN model for similarity-based recommendations
        self.knn_model = KNNBasic(k=10, sim_options={'name': 'cosine', 'user_based': True})
        self.knn_model.fit(self.trainset)
        
        return self
    
    def get_user_recommendations(self, user_id, airport_code, passenger_segment, n_recommendations=5):
        """Get personalized recommendations for a user"""
        if not self.model:
            self.train_model()
        
        try:
            # Get all products for the airport
            if self.products_df is not None:
                airport_products = self.products_df[
                    self.products_df['airport_code'] == airport_code
                ]['product_id'].tolist()
            else:
                # Fallback to generic products
                airport_products = [f"PROD_{i:03d}" for i in range(1, 21)]
            
            # Get predictions for all products
            predictions = []
            for product_id in airport_products:
                pred = self.model.predict(user_id, product_id)
                predictions.append({
                    'product_id': product_id,
                    'predicted_rating': pred.est,
                    'confidence': 1 - abs(pred.est - 3) / 2  # Simple confidence metric
                })
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
            
            # Apply business rules based on passenger segment
            if passenger_segment == 'business':
                # Prefer electronics and premium F&B
                category_weights = {
                    'Electronics & Gadgets': 1.2,
                    'Food & Beverage': 1.1,
                    'Fashion & Accessories': 1.0,
                    'Books & Magazines': 0.9,
                    'Souvenirs & Gifts': 0.8
                }
            else:  # leisure
                # Prefer souvenirs and casual items
                category_weights = {
                    'Souvenirs & Gifts': 1.2,
                    'Food & Beverage': 1.1,
                    'Fashion & Accessories': 1.0,
                    'Books & Magazines': 0.9,
                    'Electronics & Gadgets': 0.8
                }
            
            # Adjust predictions based on segment
            for pred in predictions:
                if self.products_df is not None:
                    product_info = self.products_df[
                        self.products_df['product_id'] == pred['product_id']
                    ]
                    if not product_info.empty:
                        category = product_info.iloc[0]['category']
                        weight = category_weights.get(category, 1.0)
                        pred['predicted_rating'] *= weight
            
            # Re-sort and return top N
            predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
            return predictions[:n_recommendations]
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return self._get_fallback_recommendations(airport_code, passenger_segment, n_recommendations)
    
    def _get_fallback_recommendations(self, airport_code, passenger_segment, n_recommendations):
        """Fallback recommendations when model fails"""
        if passenger_segment == 'business':
            products = [
                {'product_id': 'PROD_003', 'predicted_rating': 4.5, 'confidence': 0.8},
                {'product_id': 'PROD_001', 'predicted_rating': 4.2, 'confidence': 0.7},
                {'product_id': 'PROD_004', 'predicted_rating': 4.0, 'confidence': 0.6},
            ]
        else:
            products = [
                {'product_id': 'PROD_006', 'predicted_rating': 4.3, 'confidence': 0.8},
                {'product_id': 'PROD_002', 'predicted_rating': 4.1, 'confidence': 0.7},
                {'product_id': 'PROD_005', 'predicted_rating': 3.9, 'confidence': 0.6},
            ]
        return products[:n_recommendations]
    
    def get_product_similarity(self, product_id, n_similar=5):
        """Get similar products using item-based collaborative filtering"""
        if not self.knn_model:
            self.train_model()
        
        try:
            # Get inner product ID
            inner_product_id = self.trainset.to_inner_iid(product_id)
            
            # Get similar items
            similar_items = self.knn_model.get_neighbors(inner_product_id, k=n_similar)
            
            # Convert back to raw IDs
            similar_products = []
            for inner_id in similar_items:
                raw_id = self.trainset.to_raw_iid(inner_id)
                similar_products.append(raw_id)
            
            return similar_products
            
        except Exception as e:
            print(f"Error getting similar products: {e}")
            return []
    
    def calculate_conversion_lift(self, recommendations, baseline_conversion=0.15):
        """Calculate expected conversion lift from recommendations"""
        if not recommendations:
            return 0.0
        
        # Higher predicted ratings should lead to higher conversion
        avg_rating = np.mean([r['predicted_rating'] for r in recommendations])
        avg_confidence = np.mean([r['confidence'] for r in recommendations])
        
        # Simple model: conversion lift based on rating and confidence
        lift = (avg_rating - 3.0) * 0.1 * avg_confidence
        return max(0.0, lift)
    
    def save_model(self, filepath='models/recommender_model.pkl'):
        """Save trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'knn_model': self.knn_model,
                    'trainset': self.trainset
                }, f)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath='models/recommender_model.pkl'):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.knn_model = data['knn_model']
                self.trainset = data['trainset']
        except Exception as e:
            print(f"Error loading model: {e}")
            self.train_model()  # Train new model if loading fails
