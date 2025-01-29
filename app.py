from flask import Flask, request, jsonify, render_template, send_from_directory
from recommendation import CulturalRecommender
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuring logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static')

recommender = CulturalRecommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bharatanatyam')
def bharatanatyam():
    return render_template('bharatanatyam.html')

@app.route('/kathak')
def kathak():
    return render_template('kathak.html')

@app.route('/hyderabadibiryani')
def hyderabadibiryani():
    return render_template('hyderabadibiryani.html')

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        logger.debug(f"Received request data: {data}")
        
        query = data.get('query', '')
        preferences = data.get('preferences', {})
        top_k = data.get('top_k', 3)
        
        logger.debug(f"Processing request with query: {query}, preferences: {preferences}, top_k: {top_k}")
        
        recommendations = recommender.get_recommendations(query, preferences, top_k)
        logger.debug(f"Generated recommendations: {[r['name'] for r in recommendations]}")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommend/region/<region>')
def get_recommendations_by_region(region):
    try:
        logger.debug(f"Getting recommendations for region: {region}")
        top_k = request.args.get('top_k', 3, type=int)
        recommendations = recommender.get_recommendations_by_region(region, top_k)
        logger.debug(f"Generated recommendations for region: {[r['name'] for r in recommendations]}")
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Error in get_recommendations_by_region: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommend/festival/<festival>')
def get_recommendations_by_festival(festival):
    try:
        logger.debug(f"Getting recommendations for festival: {festival}")
        top_k = request.args.get('top_k', 3, type=int)
        recommendations = recommender.get_recommendations_by_festival(festival, top_k)
        logger.debug(f"Generated recommendations for festival: {[r['name'] for r in recommendations]}")
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Error in get_recommendations_by_festival: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/similar/<item_name>')
def get_similar_items(item_name):
    try:
        logger.debug(f"Getting similar items for: {item_name}")
        top_k = request.args.get('top_k', 3, type=int)
        similar_items = recommender.get_similar_items(item_name, top_k)
        logger.debug(f"Generated similar items: {[r['name'] for r in similar_items]}")
        return jsonify({
            'success': True,
            'recommendations': similar_items
        })
    except Exception as e:
        logger.error(f"Error in get_similar_items: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
