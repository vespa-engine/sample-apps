# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from flask import Flask, render_template, request, jsonify, session
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Vespa API settings
VESPA_SEARCH_URL = os.getenv("VESPA_API_URL") + "/search/"
CERT_PATH = os.getenv("VESPA_CERT_PATH")
KEY_PATH = os.getenv("VESPA_KEY_PATH")

# Image cache to avoid redundant Wikipedia lookups
IMAGE_CACHE = {
    # Pre-populate with hardcoded fallbacks for common car makes and models
    'toyota_camry': 'https://upload.wikimedia.org/wikipedia/commons/a/ac/2018_Toyota_Camry_%28ASV70R%29_Ascent_sedan_%282018-08-27%29_01.jpg',
    'honda_accord': 'https://upload.wikimedia.org/wikipedia/commons/7/76/2018_Honda_Accord_front_4.1.18.jpg',
    'tesla_model_3': 'https://upload.wikimedia.org/wikipedia/commons/9/91/2019_Tesla_Model_3_Performance_AWD_Front.jpg',
    'ford_f-150': 'https://upload.wikimedia.org/wikipedia/commons/a/a8/2018_Ford_F-150_XLT_Crew_Cab.jpg',
    'chevrolet_cruze': 'https://upload.wikimedia.org/wikipedia/commons/6/6d/2016_Chevrolet_Cruze_front_5.24.18.jpg',
    'bmw_5_series': 'https://upload.wikimedia.org/wikipedia/commons/9/9b/2018_BMW_530d_M_Sport_Automatic_3.0.jpg',
    'mercedes_e_class': 'https://upload.wikimedia.org/wikipedia/commons/8/8c/Mercedes-Benz_W213_E_350_by_RudolfSimon_cropped.jpg',
    'audi_a4': 'https://upload.wikimedia.org/wikipedia/commons/d/d5/Audi_A4_B9_sedans_%28FL%29_IMG_3699.jpg',
    'opel_corsa': 'https://upload.wikimedia.org/wikipedia/commons/6/69/2012_Opel_Corsa_%28CO%29_Enjoy_5-door_hatchback_%282015-11-11%29_01.jpg',
}

def load_system_prompt():
    """Load the system prompt from file"""
    with open(os.path.join(os.path.dirname(__file__), "system_prompt.txt"), "r") as f:
        return f.read()

def extract_json_from_response(response):
    """Extract JSON data from the OpenAI response"""
    try:
        json_str = response.split("===")[1].strip()
        if json_str.startswith("JSON"):
            json_str = json_str[4:].strip()
        return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None

def get_wikipedia_image(make, model):
    """Fetch a relevant image URL from Wikipedia for a given make and model"""
    try:
        # Create a cache key
        cache_key = f"{make}_{model}".lower().replace(" ", "_")
        
        # Check if we already have this in the cache
        if cache_key in IMAGE_CACHE:
            print(f"Using cached image for {make} {model}")
            return IMAGE_CACHE[cache_key]
            
        print(f"Fetching image for {make} {model}")
        # Try direct search for car model first
        search_query = f"{make} {model} car"
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "format": "json",
            "origin": "*"
        }
        
        print(f"Searching Wikipedia for: {search_query}")
        search_response = requests.get(search_url, params=search_params, timeout=5)
        search_data = search_response.json()
        
        # If no search results, try a simpler query
        if not search_data.get("query", {}).get("search"):
            print(f"No results for {search_query}, trying just {make}")
            search_query = f"{make} car"
            search_params["srsearch"] = search_query
            search_response = requests.get(search_url, params=search_params, timeout=5)
            search_data = search_response.json()
            
            if not search_data.get("query", {}).get("search"):
                print(f"No results for {search_query} either")
                return None
            
        # Get the first search result's page ID
        page_id = search_data["query"]["search"][0]["pageid"]
        print(f"Found Wikipedia page ID: {page_id}")
        
        # Get images from the page
        image_params = {
            "action": "query",
            "prop": "images",
            "pageids": page_id,
            "format": "json",
            "origin": "*"
        }
        
        image_response = requests.get(search_url, params=image_params, timeout=5)
        image_data = image_response.json()
        
        # Check if we have any images
        images = image_data.get("query", {}).get("pages", {}).get(str(page_id), {}).get("images", [])
        if not images:
            print(f"No images found on page {page_id}")
            return None
            
        print(f"Found {len(images)} images on page")
        
        # Direct check for car images
        car_image_keywords = ["car", "vehicle", "automobile", "sedan", "suv", "truck", make.lower(), model.lower()]
        for image in images:
            title = image.get("title", "").lower()
            if any(keyword in title for keyword in car_image_keywords):
                # Get the actual image URL
                image_url_params = {
                    "action": "query",
                    "prop": "imageinfo",
                    "titles": image.get("title"),
                    "iiprop": "url",
                    "format": "json",
                    "origin": "*"
                }
                
                url_response = requests.get(search_url, params=image_url_params, timeout=5)
                url_data = url_response.json()
                
                # Extract the actual image URL
                pages = url_data.get("query", {}).get("pages", {})
                for page in pages.values():
                    if "imageinfo" in page and page["imageinfo"]:
                        image_url = page["imageinfo"][0].get("url")
                        if image_url:
                            print(f"Found image URL: {image_url}")
                            # Store in cache
                            IMAGE_CACHE[cache_key] = image_url
                            return image_url
        
        # If we didn't find a specific car image, try any image
        print("No car-specific images found, trying any image")
        for image in images:
            if not image.get("title", "").lower().endswith(('.svg', '.png', '.gif')):
                # Prefer JPG/JPEG images as they're more likely to be photos
                image_url_params = {
                    "action": "query",
                    "prop": "imageinfo",
                    "titles": image.get("title"),
                    "iiprop": "url",
                    "format": "json",
                    "origin": "*"
                }
                
                url_response = requests.get(search_url, params=image_url_params, timeout=5)
                url_data = url_response.json()
                
                # Extract the actual image URL
                pages = url_data.get("query", {}).get("pages", {})
                for page in pages.values():
                    if "imageinfo" in page and page["imageinfo"]:
                        image_url = page["imageinfo"][0].get("url")
                        if image_url and not image_url.lower().endswith(('.svg', '.png', '.gif')):
                            print(f"Found fallback image URL: {image_url}")
                            # Store in cache
                            IMAGE_CACHE[cache_key] = image_url
                            return image_url
        
        # Check if we have a generic fallback for this make
        make_key = make.lower().replace(" ", "_")
        for cached_key in IMAGE_CACHE:
            if cached_key.startswith(make_key + "_"):
                print(f"Using similar make image for {make} {model}")
                # Store this in the cache for this specific model too
                IMAGE_CACHE[cache_key] = IMAGE_CACHE[cached_key]
                return IMAGE_CACHE[cached_key]
        
        print(f"No images found for {make} {model}")
        return None
    except requests.exceptions.Timeout:
        print(f"Timeout while fetching image for {make} {model}")
        return None
    except Exception as e:
        print(f"Error fetching Wikipedia image for {make} {model}: {e}")
        return None

def query_vespa(preferences, facet_filters=None):
    """Query Vespa with user preferences and facet filters for search results only"""
    # Convert preferences to Vespa feature format
    features = []
    for key, value in preferences.items():
        if key in ['make', 'model']:
            # Handle make/model specially
            features.append(f"{{features:{key.lower()}}}:{value}")
        else:
            features.append(f"{{features:{key}}}:{value}")
    
    # Create the where clause based on selected facets
    if not facet_filters or all(len(filters) == 0 for filters in facet_filters.values()):
        # No filters, just use "true"
        where_clause = "true"
    else:
        # Start building an AND clause
        and_conditions = []
        
        # Add make filter if any
        if facet_filters.get('make') and len(facet_filters['make']) > 0:
            make_conditions = []
            for make in facet_filters['make']:
                make_conditions.append({"contains": ["make", make]})
            and_conditions.append({"or": make_conditions})
        
        # Add model filter if any
        if facet_filters.get('model') and len(facet_filters['model']) > 0:
            model_conditions = []
            for model in facet_filters['model']:
                model_conditions.append({"contains": ["model", model]})
            and_conditions.append({"or": model_conditions})
        
        # Add transmission filter if any
        if facet_filters.get('transmission') and len(facet_filters['transmission']) > 0:
            transmission_conditions = []
            for transmission in facet_filters['transmission']:
                transmission_conditions.append({"contains": ["transmission", transmission]})
            and_conditions.append({"or": transmission_conditions})
        
        # Add fuelType filter if any
        if facet_filters.get('fuelType') and len(facet_filters['fuelType']) > 0:
            fuelType_conditions = []
            for fuelType in facet_filters['fuelType']:
                fuelType_conditions.append({"contains": ["fuelType", fuelType]})
            and_conditions.append({"or": fuelType_conditions})
        
        # Final where clause
        where_clause = {"and": and_conditions}
    
    # Results-only query without grouping
    query = {
        "select": {
            "where": where_clause
        },
        "hits": 10,
        "ranking": "rank_cars",
        "presentation.summary": "attributes",
        "ranking.features.query(user_preferences)": '{' + ','.join(features) + '}',
        "trace.level": 1
    }
    
    print(f"Sending results query to Vespa: {json.dumps(query, indent=2)}")
    
    try:
        # Use mTLS certificates for authentication
        response = requests.post(
            VESPA_SEARCH_URL, 
            json=query,
            cert=(CERT_PATH, KEY_PATH)
        )
        
        response.raise_for_status()
        results = response.json()
        
        # Don't wait for image loading - just return the results
        # Each car result will have a "car_id" that the frontend can use to request images
        if results and "root" in results and "children" in results["root"]:
            for item in results["root"]["children"]:
                if "fields" in item:
                    # Add a unique car ID based on make and model
                    make = item["fields"].get("make", "")
                    model = item["fields"].get("model", "")
                    if make and model:
                        item["fields"]["car_id"] = f"{make}_{model}".replace(" ", "_")
        
        return results
    except requests.exceptions.RequestException as e:
        print(f"Error querying Vespa for results: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response body: {e.response.text}")
        return None

def query_vespa_facets(preferences, facet_filters=None):
    """Query Vespa for facets only. Special case: if exactly one make is selected, filter by it and show models."""
    # Convert preferences to Vespa feature format
    features = []
    for key, value in preferences.items():
        if key in ['make', 'model']:
            # Handle make/model specially
            features.append(f"{{features:{key.lower()}}}:{value}")
        else:
            features.append(f"{{features:{key}}}:{value}")
    
    # Check if we have exactly one make selected
    single_make_selected = (
        facet_filters and 
        facet_filters.get('make') and 
        len(facet_filters['make']) == 1
    )
    
    # Facets-only query with grouping
    # If single make selected, filter by it and add model facet
    query = {
        "select": {
            "where": "true" if not single_make_selected else 
                    {"contains": ["make", facet_filters['make'][0]]},
            "grouping": [
                {
                    "all": {
                        "group": "make",
                        "order": "-count()",
                        "max": 10,
                        "each": {
                            "output": "count()"
                        }
                    }
                },
                {
                    "all": {
                        "group": "transmission",
                        "order": "-count()",
                        "max": 10,
                        "each": {
                            "output": "count()"
                        }
                    }
                },
                {
                    "all": {
                        "group": "fuelType",
                        "order": "-count()",
                        "max": 10,
                        "each": {
                            "output": "count()"
                        }
                    }
                }
            ]
        },
        "hits": 0,  # We don't need actual hits for facets
        "ranking": "rank_cars",
        "presentation.summary": "attributes",
        "ranking.features.query(user_preferences)": '{' + ','.join(features) + '}',
        "trace.level": 1
    }
    
    # Add model facet if single make is selected
    if single_make_selected:
        query["select"]["grouping"].append({
            "all": {
                "group": "model",
                "order": "-count()",
                "max": 20,  # Show more model options
                "each": {
                    "output": "count()"
                }
            }
        })
    
    print(f"Sending facets query to Vespa: {json.dumps(query, indent=2)}")
    
    try:
        # Use mTLS certificates for authentication
        response = requests.post(
            VESPA_SEARCH_URL, 
            json=query,
            cert=(CERT_PATH, KEY_PATH)
        )
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Vespa for facets: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response body: {e.response.text}")
        return None

@app.route('/')
def index():
    """Render the main page"""
    # Initialize a new session if needed
    if 'messages' not in session:
        session['messages'] = []
    if 'preferences' not in session:
        session['preferences'] = {}
    return render_template('index.html', messages=session['messages'])

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.json
    user_message = data.get('message', '')
    manual_preferences = data.get('manual_preferences', {})
    facet_filters = data.get('facet_filters', {})
    is_preference_adjustment = data.get('is_preference_adjustment', False)
    is_preference_removal = data.get('is_preference_removal', False)
    adjustment = data.get('adjustment', {})
    removal = data.get('removal', {})
    
    # Initialize or get conversation history
    if 'messages' not in session:
        session['messages'] = []
        
    # Add user message to history
    session['messages'].append({"role": "user", "content": user_message})
    
    # Prepare messages for OpenAI - use the standard system prompt without modifications
    system_prompt = load_system_prompt()
    
    # Add current facet filters to the system prompt if any are active
    if facet_filters and any(len(filters) > 0 for filters in facet_filters.values()):
        filter_context = "\nBelow are the filters I have already selected." + \
             " Keep them in mind as implicit preferences when suggesting other options.\n"
        for facet_type, filters in facet_filters.items():
            if filters:
                filter_context += f"- {facet_type.title()}: {', '.join(filters)}\n"
        system_prompt += filter_context
        print(f"System prompt: {system_prompt}")
    
    openai_messages = [
        {"role": "system", "content": system_prompt}
    ] + session['messages']
    
    try:
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            temperature=0.7,
        )
        
        # Extract assistant's response
        assistant_response = response.choices[0].message.content
        
        # Add assistant response to session history
        session['messages'].append({"role": "assistant", "content": assistant_response})
        session.modified = True  # Ensure session is saved
        
        # Extract JSON preferences if present
        json_data = extract_json_from_response(assistant_response)
        
        # If it's a preference adjustment
        if is_preference_adjustment:
            # Get existing preferences from session or create new
            existing_prefs = session.get('preferences', {}).copy()
            
            if not json_data:
                json_data = existing_prefs.copy()
                
            # If this is a removal, remove the preference
            if is_preference_removal and 'key' in removal:
                key_to_remove = removal['key']
                if key_to_remove in json_data:
                    del json_data[key_to_remove]
            # Otherwise, if it's an adjustment, set the preference
            elif 'key' in adjustment and 'value' in adjustment:
                json_data[adjustment['key']] = adjustment['value']
        
        # Store preferences in session
        if json_data:
            session['preferences'] = json_data
            session.modified = True
        
        # Store facet filters in session
        session['facet_filters'] = facet_filters
        session.modified = True
        
        # Query Vespa if we have preferences
        search_results = None
        facet_results = None
        if json_data:
            # Get filtered results
            search_results = query_vespa(json_data, facet_filters)
            # Get complete facets
            facet_results = query_vespa_facets(json_data, facet_filters)
        
        return jsonify({
            'response': assistant_response,
            'preferences': json_data,
            'search_results': search_results,
            'facet_results': facet_results
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_preferences', methods=['POST'])
def update_preferences():
    """Update preferences and return new search results with separate facets"""
    data = request.json
    preferences = data.get('preferences', {})
    facet_filters = data.get('facet_filters', {})
    
    if not preferences:
        return jsonify({'error': 'No preferences provided'}), 400
    
    # Store preferences in session
    session['preferences'] = preferences
    session.modified = True
    
    # Store facet filters in session
    session['facet_filters'] = facet_filters
    session.modified = True
    
    # Query Vespa with the updated preferences and facet filters (for results)
    search_results = query_vespa(preferences, facet_filters)
    
    # Query Vespa for facets, passing the filters for make/model handling
    facet_results = query_vespa_facets(preferences, facet_filters)
    
    return jsonify({
        'preferences': preferences,
        'search_results': search_results,
        'facet_results': facet_results
    })

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation history"""
    session['messages'] = []
    if 'preferences' in session:
        del session['preferences']
    if 'facet_filters' in session:
        del session['facet_filters']
    return jsonify({'status': 'success'})

@app.route('/api/car_image/<make>/<model>', methods=['GET'])
def get_car_image(make, model):
    """Endpoint to get a car image asynchronously, with caching"""
    try:
        # Create a cache key from make and model
        cache_key = f"{make}_{model}".lower().replace(" ", "_")
        
        # Check if image is already in cache
        if cache_key in IMAGE_CACHE:
            print(f"Cache hit for {make} {model}")
            return jsonify({"image_url": IMAGE_CACHE[cache_key], "from_cache": True})
        
        # Not in cache, try to get an image from Wikipedia
        print(f"Cache miss for {make} {model}, fetching from Wikipedia")
        image_url = get_wikipedia_image(make, model)
        
        if image_url:
            # Store in cache for future use
            IMAGE_CACHE[cache_key] = image_url
            return jsonify({"image_url": image_url, "from_cache": False})
        else:
            # No image found
            return jsonify({"error": "No image found"}), 404
    except Exception as e:
        print(f"Error getting car image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 