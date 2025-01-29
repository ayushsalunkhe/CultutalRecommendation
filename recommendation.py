from data import CULTURAL_DATA
import logging
import requests
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CulturalRecommender:
    def __init__(self):
        self.categories = ["dance_forms", "music", "cuisines"]
        self.items = self._prepare_items()
        self.API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        logger.debug(f"Initialized recommender with {len(self.items)} items")

    def _prepare_items(self) -> List[Dict[str, Any]]:
        """Prepare all items for recommendation"""
        items = []
        for category in self.categories:
            if category in CULTURAL_DATA:
                for item in CULTURAL_DATA[category]:
                    item_copy = item.copy()
                    item_copy['category_name'] = category
                    items.append(item_copy)
            else:
                logger.warning(f"Category {category} not found in cultural data")
        return items

    def _get_item_description(self, item: Dict[str, Any]) -> str:
        """Create a rich description for matching"""
        keywords = item.get('keywords', [])
        festivals = item.get('festivals', [])
        traditions = item.get('traditions', [])
        
        description = f"{item['name']} {item['description']} {item['region']}"
        if keywords:
            description += f" {' '.join(keywords)}"
        if festivals:
            description += f" {' '.join(festivals)}"
        if traditions:
            description += f" {' '.join(traditions)}"
        
        return description

    def _calculate_similarity_scores(self, query: str, descriptions: List[str]) -> List[float]:
        """Calculate semantic similarity scores using Hugging Face API"""
        try:
            # Enhance query with category context if detected
            category = self._get_category_from_query(query)
            if category:
                query = f"{query} {category.replace('_', ' ')}"
            
            payload = {
                "inputs": {
                    "source_sentence": query,
                    "sentences": descriptions
                }
            }
            
            logger.debug(f"Sending request to Hugging Face API with query: {query}")
            response = requests.post(self.API_URL, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                scores = response.json()
                logger.debug(f"Received similarity scores from API: {scores}")
                return scores
            else:
                logger.error(f"API Error: {response.text}")
                return self._fallback_similarity_scores(query, descriptions)
                
        except Exception as e:
            logger.error(f"Error calculating similarity scores: {str(e)}")
            return self._fallback_similarity_scores(query, descriptions)

    def _fallback_similarity_scores(self, query: str, descriptions: List[str]) -> List[float]:
        """Fallback method for calculating similarity when API fails"""
        scores = []
        query_words = set(query.lower().split())
        for desc in descriptions:
            desc_words = set(desc.lower().split())
            common_words = query_words.intersection(desc_words)
            score = len(common_words) / len(query_words) if query_words else 0.5
            scores.append(score)
        return scores

    def _apply_preference_boost(self, score: float, item: Dict[str, Any], preferences: Dict[str, str]) -> float:
        """Apply preference-based score boosting"""
        if not preferences:
            return score

        final_score = score

        # Region preference 
        if 'region' in preferences and preferences['region']:
            if item['region'].lower() == preferences['region'].lower():
                # boost for exact region match
                final_score *= 50.0
            else:
                #  penalty for non-matching region
                final_score *= 0.02

        # Category preference 
        if 'category' in preferences and preferences['category']:
            if item['category_name'].lower() == preferences['category'].lower():
                final_score *= 2.0
            else:
                final_score *= 0.1

        logger.debug(f"Item: {item['name']}, Region: {item['region']}, "
                    f"Preferences: {preferences}, "
                    f"Original Score: {score:.4f}, Final Score: {final_score:.4f}")
        
        return final_score

    def _get_category_from_query(self, query: str) -> str:
        """Determine category based on query keywords"""
        query = query.lower()
        if any(word in query for word in ['dance', 'dancing', 'classical dance']):
            return 'dance_forms'
        elif any(word in query for word in ['music', 'song', 'singing', 'musical']):
            return 'music'
        elif any(word in query for word in ['food', 'cuisine', 'dish', 'cooking']):
            return 'cuisines'
        return ''

    def get_recommendations(self, query: str, preferences: Dict[str, str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get personalized cultural recommendations"""
        try:
            # Initialize preferences if None
            preferences = preferences or {}
            
            # Detect category from query 
            if 'category' not in preferences or not preferences['category']:
                detected_category = self._get_category_from_query(query)
                if detected_category:
                    preferences['category'] = detected_category
                    logger.debug(f"Detected category from query: {detected_category}")

            # Log the preferences being used
            logger.debug(f"Getting recommendations with query: {query}, preferences: {preferences}")
            
            # Get descriptions for all items
            descriptions = [self._get_item_description(item) for item in self.items]
            
            # Calculate similarity scores
            if query:
                similarity_scores = self._calculate_similarity_scores(query, descriptions)
            else:
                similarity_scores = [1.0] * len(self.items)  # Equal scores if no query
            
            # Apply preference boosts and create scored items
            scored_items = []
            for i, (score, item) in enumerate(zip(similarity_scores, self.items)):
                # Skip items that don't match the category 
                if preferences.get('category') and item['category_name'] != preferences['category']:
                    continue
                
                final_score = self._apply_preference_boost(score, item, preferences)
                scored_items.append((final_score, i))
                logger.debug(f"Scored item: {item['name']}, Region: {item['region']}, Category: {item['category_name']}, "
                           f"Base Score: {score:.4f}, Final Score: {final_score:.4f}")
            
            # Sort by score in descending order
            scored_items.sort(reverse=True, key=lambda x: x[0])
            
            # Create recommendation objects
            recommendations = []
            for score, idx in scored_items[:top_k]:
                item = self.items[idx]
                recommendation = {
                    'name': item['name'],
                    'region': item['region'],
                    'description': item['description'],
                    'category': item.get('category', ''),
                    'category_name': item['category_name'],
                    'festivals': item.get('festivals', []),
                    'traditions': item.get('traditions', []),
                    'videos': item.get('videos', []),
                    'similarity_score': score
                }
                recommendations.append(recommendation)
            
            # final recommendations
            logger.debug("Final recommendations:")
            for r in recommendations:
                logger.debug(f"- {r['name']} (Region: {r['region']}, Category: {r['category_name']}, Score: {r['similarity_score']:.4f})")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
            raise

    def get_recommendations_by_region(self, region: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get recommendations specific to a region"""
        preferences = {"region": region}
        return self.get_recommendations("", preferences, top_k)

    def get_recommendations_by_festival(self, festival_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get recommendations related to a specific festival"""
        return self.get_recommendations(festival_name, top_k=top_k)

    def get_similar_items(self, item_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find items similar to a given item"""
        try:
            # Find the target 
            target_item = None
            for item in self.items:
                if item['name'].lower() == item_name.lower():
                    target_item = item
                    break
            
            if not target_item:
                logger.warning(f"Item not found: {item_name}")
                return []
            
            # Create preferences 
            preferences = {
                "region": target_item["region"],
                "category": target_item["category_name"]
            }
            
            # Get recommendations using user interest
            query = self._get_item_description(target_item)
            recommendations = self.get_recommendations(query, preferences, top_k=top_k+1)
            
            # Filter 
            filtered_recommendations = [
                rec for rec in recommendations 
                if rec['name'].lower() != item_name.lower()
            ]
            
            return filtered_recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar items: {str(e)}", exc_info=True)
            raise
