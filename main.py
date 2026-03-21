import os
import json
import requests
import logging
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import Any

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Setup the Brain
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
# --- ADD THIS LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# ------------------------------

# TODO: Replace with your actual BAMPFA Blacklight JSON endpoint
BAMPFA_API_URL = "https://collection.bampfa.berkeley.edu/catalog.json" 
HEADERS = {
    "User-Agent": "BAMPFA-AI-Bridge/1.0",
    "Accept": "application/json"
}

from museum_fields import materials

def parse_results(json_data: dict):
    artworks = []
    data = json_data.get('data', [])
    seen_urls = set()
    
    for record in data:
        doc = record.get('attributes')
        if not doc: 
            continue

        # Grab the self link first
        item_url = record.get('links', {}).get('self')
        
        # If we've already added this exact URL, skip it to prevent duplicates!
        if item_url:
            if item_url in seen_urls:
                continue
            else:
                seen_urls.add(item_url)
            
        def get_val(key, default=""):
            field = doc.get(key)
            if isinstance(field, dict):
                return field.get('attributes', {}).get('value', default)
            elif isinstance(field, str):
                return field
            return default
        

        artworks.append({
            "title": get_val('title', 'Untitled'),
            "artist": get_val('artistcalc_s', 'Unknown Artist'),
            "classification": get_val('itemclass_s', 'Unknown'),
            "material": get_val('materials_s', 'Unknown'),
            "url": item_url # Pass the URL to the frontend
        })
        
    return artworks

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search_bampfa(request: Request, query: str = Form(...)):
    
    # Our masterpiece prompt from earlier
    system_prompt = f"""
    You are an intelligent search routing assistant for the Berkeley Art Museum.
    Your job is to take a user's natural language request and convert it into a strict JSON object that will be sent to our Blacklight/Solr search API.

    Extract the following fields if they are mentioned. If a field is not mentioned, return null for that field.

    1. "query": Any general keyword, subject, or title.
    2. "artist": The specific name of the creator.
    3. "country": The origin country (e.g., "Japan", "France", "United States").
    4. "material": The physical material.
    5. "classification": The type of item.

    CRITICAL RULE FOR "query":
    - Strip out all conversational filler ("show me", "I want to see", "do you have").
    - Strip out subjective or relative adjectives ("old", "beautiful", "big", "recent").
    - ONLY extract the core, essential keywords.

    CRITICAL RULE FOR CLASSIFICATION:
    - ALWAYS establish the classification first based on the core noun of the user's request (e.g., "documents", "drawings", "paintings").
    - If the user implies a classification, you MUST map it to one of the following exact terms. Do not invent your own terms. 
    - Permitted Classifications: ["Photograph", "Print", "Work on paper", "Painting", "Drawing", "Sculpture", "Documentation", "Ephemera", "Artist's book", "Mixed media", "Textile", "Ceramic", "Decorative Arts", "Video", "Multiple", "Audio", "Artifact", "Installation", "Film", "Multi-Media", "Furniture"]

    Examples:
    - "Show me some pics" -> "Photograph"
    - "I want to see statues" -> "Sculpture"
    - "Movies" -> "Film"
    - "Documents" -> "Documentation"

    CRITICAL RULE FOR MATERIALS (THE ALIAS ENGINE):
    We have hundreds of highly specific material strings in our database. You must semantically map the user's requested material to the SINGLE closest exact match from the permitted list below. 
    - "material" MUST be a single string or null. NEVER an array.
    - STRICT MATCHING: If you cannot find a highly logical, confident match for the EXACT material requested, you MUST set "material" to null. 
    - DO NOT guess a material based on a single overlapping word (e.g., do NOT map "black ink" to "black-and-white photograph").
    - CRITICAL: You MUST use the EXACT capitalization provided in the list. Return the EXACT string provided in the list.
    - If you set "material" to null, put their core material keywords into the "query" string instead.
    - If you successfully map a material to the Permitted List, do NOT put those material keywords into the "query" field.

    Permitted Materials:
    [{materials}]

    User Request: "{query}"
    Output ONLY valid JSON.
"""
    
    try:
        artworks_data = []
        error_msg = None
        
        # 1. Ask the AI to parse the intent
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=system_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0 # Lowest temperature for strict rule-following
            )
        )

        # 1. THE PROMPT
        logger.info("\n" + "="*40 + " NEW SEARCH " + "="*40)
        logger.info(f"1. USER PROMPT: {query}")

        # ... your existing Gemini generation code ...

        # 2. THE RETURN FROM GEMINI
        logger.info(f"2. GEMINI RAW RESPONSE:\n{response.text}")
        
        
        if not response or not response.text:
            raise ValueError("The AI returned an empty response (likely due to a safety filter or API timeout). Please try rephrasing your query.")

        search_params = json.loads(response.text)
       # 2. Map the JSON to Solr/Blacklight GET parameters
        api_params: dict[str, Any] = {"rows": 12} # Get up to 12 results
        
        # --- COMBINE QUERY AND ARTIST INTO KEYWORD SEARCH ---
        query_parts = []
        if search_params.get("query"):
            query_parts.append(search_params["query"])
        if search_params.get("artist"):
            query_parts.append(search_params["artist"])
            
        if query_parts:
            # Join them with a space and send to the 'q' parameter
            api_params["q"] = " ".join(query_parts)
        
        # 3. THE EXTRACTED SEARCH ATTRIBUTES
        logger.info(f"3. EXTRACTED ATTRIBUTES: {json.dumps(search_params)}")

            
        # Blacklight handles facet filters using f[field_name][]=value
        
        if search_params.get("country"):
            api_params["f[artistorigin_s][]"] = search_params["country"]
        if search_params.get("material"):
            api_params["f[materials_s][]"] = search_params["material"]
        if search_params.get("classification"):
            api_params["f[itemclass_s][]"] = search_params["classification"]

        # 4. THE EXTRACTED SEARCH ATTRIBUTES
        logger.info(f"4. SEARCH PARAMETERS TO SOLR: {json.dumps(api_params)}")
        # 3. Call the API
        bampfa_res = requests.get(BAMPFA_API_URL, params=api_params, headers=HEADERS)
        bampfa_res.raise_for_status() # Throw error if we get a 404/500
        
        # 4. Parse the clean JSON
        artworks_data = parse_results(bampfa_res.json())
        logged_records = [{"title": a["title"], "url": a["url"]} for a in artworks_data]
        logger.info(f"5. RETURNED RECORDS (Count: {len(artworks_data)}): {json.dumps(logged_records, indent=2)}")
        logger.info("="*92 + "\n")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(error_msg)

    

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "query": query, 
        "artworks": artworks_data,
        "error": error_msg
    })