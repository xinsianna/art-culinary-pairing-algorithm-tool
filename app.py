from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
import base64 
import os

app = Flask(__name__)

# Paths to CSV dataset files
FOOD_CSV_PATH = '/Users/sianna/Downloads/cpsc490/fooddataset490.csv'
ART_CSV_PATH = '/Users/sianna/Downloads/cpsc490/artdataset490.csv'

# HTML 
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Art and Culinary Pairing</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=Montserrat:wght@300;400;500;600&family=Poppins:wght@300;400;500;600&display=swap');

        :root {
            --modern-bg: #1a1a1a;
            --modern-text: #e5e5e5;
            --modern-accent: #C0A080;
            --modern-card-bg: rgba(255, 255, 255, 0.03);
            --modern-border: rgba(255, 255, 255, 0.1);
            
            --classic-bg: #f7f7f7;
            --classic-text: #333333;
            --classic-accent: #3498db;
            --classic-card-bg: #ffffff;
            --classic-border: #ecf0f1;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            transition: background-color 0.3s, color 0.3s;
        }

        body {
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }

        body.modern-theme {
            background-color: var(--modern-bg);
            color: var(--modern-text);
            font-family: 'Montserrat', sans-serif;
        }

        body.classic-theme {
            background-color: var(--classic-bg);
            color: var(--classic-text);
            font-family: 'Poppins', sans-serif;
        }

        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }

        .switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }

        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            bbackground-color: var(--modern-accent);
            transition: .4s;
            border-radius: 34px;
        }

        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }

        .nav-arrow {
            padding: 0.5rem 1rem;
            font-size: 1.5rem;
            line-height: 1;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: opacity 0.3s;
        }

        .nav-arrow:hover {
            opacity: 0.8;
        }

        .modern-theme .nav-arrow {
            background: var(--modern-accent);
            color: var(--modern-bg);
        }

        .classic-theme .nav-arrow {
            background: var(--classic-accent);
            color: white;
        }

        input:checked + .slider {
            background-color: var(--classic-accent); 
        }

        input:checked + .slider:before {
            transform: translateX(26px);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            text-align: center;
            padding: 2rem 0;
        }

        .modern-theme header h1 {
            font-family: 'Cormorant Garamond', serif;
            font-size: 3.5rem;
            font-weight: 300;
            letter-spacing: 2px;
            color: var(--modern-text);
        }

        .classic-theme header h1 {
            font-family: 'Poppins', sans-serif;
            font-size: 2.5rem;
            font-weight: 600;
            color: var(--classic-text);
        }

        .subtitle {
            margin-top: 1rem;
        }

        .modern-theme .subtitle {
            font-family: 'Montserrat', sans-serif;
            color: var(--modern-accent);
            text-transform: uppercase;
            letter-spacing: 3px;
        }

        .classic-theme .subtitle {
            font-family: 'Poppins', sans-serif;
            color: var(--classic-accent);
        }

        .content-wrapper {
            display: grid;
            gap: 3rem;
            transition: all 0.3s ease;
        }

        .content-wrapper.centered {
            grid-template-columns: 1fr;
            max-width: 600px;
            margin: 0 auto;
        }

        .content-wrapper.split {
            grid-template-columns: 1fr 1fr;
            max-width: 100%;
        }

        .input-section {
            padding: 2.5rem;
            border-radius: 12px;
        }

        .modern-theme .input-section {
            background: var(--modern-card-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--modern-border);
        }

        .classic-theme .input-section {
            background: var(--classic-card-bg);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        textarea {
            width: 100%;
            min-height: 150px;
            padding: 1.2rem;
            border-radius: 8px;
            font-size: 0.95rem;
            resize: vertical;
            margin: 1rem 0;
        }

        .modern-theme textarea {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--modern-border);
            color: var(--modern-text);
            font-family: 'Montserrat', sans-serif;
        }

        .classic-theme textarea {
            background: white;
            border: 1px solid var(--classic-border);
            color: var(--classic-text);
            font-family: 'Poppins', sans-serif;
        }

        button {
            padding: 1rem 2rem;
            border-radius: 6px;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .modern-theme button {
            background: var(--modern-accent);
            color: var(--modern-bg);
            border: none;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .classic-theme button {
            background: var(--classic-accent);
            color: white;
            border: none;
        }

        .output-section {
            display: none;
            padding: 2.5rem;
            border-radius: 12px;
        }

        .modern-theme .output-section {
            background: var(--modern-card-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--modern-border);
        }

        .classic-theme .output-section {
            background: var(--classic-card-bg);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        /* Slideshow specific styles */
        .art-matches {
            position: relative;
            min-height: 500px;
        }

        .art-match {
            display: none;
            padding: 1.5rem;
            border-radius: 8px;
        }

        .art-match.active {
            display: block;
        }

        .modern-theme .art-match {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid var(--modern-border);
        }

        .classic-theme .art-match {
            background: var(--classic-card-bg);
            border: 1px solid var(--classic-border);
        }

        .art-display {
            margin: 2rem 0;
            border-radius: 8px;
            overflow: hidden;
            height: 400px;
        }

        .art-display img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 8px;
        }

        .slideshow-nav {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            margin-top: 1rem;
        }

        .nav-dots {
            display: flex;
            gap: 0.5rem;
        }

        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #ccc;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        .modern-theme .dot.active {
            background-color: var(--modern-accent);
        }

        .classic-theme .dot.active {
            background-color: var(--classic-accent);
        }

        .about-link {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1000;
            text-decoration: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .modern-theme .about-link {
           background: var(--modern-accent);
            color: var(--modern-bg);
            font-family: 'Montserrat', sans-serif;
            text-transform: uppercase;
            letter-spacing: 1px;
            border: none;
        }

        .classic-theme .about-link {
            background: var(--classic-accent);
            color: white;
            font-family: 'Poppins', sans-serif;
            border: none;
        }       

        .about-link:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }


        /* Update the about overlay styles */
        .about-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 2000;
            overflow-y: auto;
        }

        .modern-theme .about-overlay {
            background: var(--modern-bg);
            color: var(--modern-text);
        }

        .classic-theme .about-overlay {
            background: var(--classic-bg);
            color: var(--classic-text);
        }

        .about-content {
            max-width: 800px;
            margin: 0 auto;
            padding: 4rem 2rem;
            position: relative;
        }

        .modern-theme .about-content h2 {
            font-family: 'Cormorant Garamond', serif;
            font-size: 2.5rem;
            font-weight: 300;
            letter-spacing: 2px;
            margin-bottom: 2rem;
        }

        .classic-theme .about-content h2 {
            font-family: 'Poppins', sans-serif;
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 2rem;
        }

        .about-content p {
            line-height: 1.8;
            margin-bottom: 1.5rem;
        }

        .modern-theme .about-content p {
            font-family: 'Montserrat', sans-serif;
        }

        .classic-theme .about-content p {
            font-family: 'Poppins', sans-serif;
        }

        .about-content ul {
            margin: 1.5rem 0 2rem 1.5rem;
            line-height: 1.8;
        }

        .about-content li {
            margin-bottom: 0.5rem;
        }

        .close-about {
            position: fixed;
            top: 20px;
            right: 20px;
            font-size: 1.5rem;
            cursor: pointer;
            border: none;
            background: none;
            padding: 0.5rem;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }

        .modern-theme .close-about {
            color: var(--modern-text);
        }

        .classic-theme .close-about {
            color: var(--classic-text);
        }

        .close-about:hover {
            transform: scale(1.1);
        }

        @media (max-width: 768px) {
            .about-content {
                padding: 4rem 1.5rem;
            }
            
            .about-content h2 {
                font-size: 1.75rem;
            }
        }
    </style>
</head>
<body class="modern-theme">
<a class="about-link" onclick="showAbout()">About</a>
    <div class="theme-toggle">
        <label class="switch">
            <input type="checkbox" onchange="toggleTheme()">
            <span class="slider"></span>
        </label>
    </div>

    <div class="container">
        <header>
            <h1>Art and Culinary Pairing</h1>
            <div class="subtitle">Experience Art With Food</div>
        </header>

        <div class="content-wrapper centered">
            <div class="input-section">
                <h2>Enter Food Description</h2>
                <textarea id="food-input" placeholder="Describe your dish or food..."></textarea>
                <button onclick="generatePairing()">Generate Art Pairing</button>
            </div>

            <div id="output-section" class="output-section">
                <h2>Recommended Art Pairings</h2>
                <div class="art-matches"></div>
            </div>
        </div>

        <div id="loading"></div>
    </div>

    <script>
        let currentSlide = 0;
        let totalSlides = 0;

        function toggleTheme() {
            document.body.classList.toggle('classic-theme');
            document.body.classList.toggle('modern-theme');
        }

        function showSlide(index) {
            const slides = document.querySelectorAll('.art-match');
            const dots = document.querySelectorAll('.dot');
            
            slides.forEach(slide => slide.classList.remove('active'));
            dots.forEach(dot => dot.classList.remove('active'));
            
            slides[index].classList.add('active');
            dots[index].classList.add('active');
            
            currentSlide = index;
        }

        function createDots(count) {
            const navDots = document.createElement('div');
            navDots.className = 'nav-dots';
            
            for (let i = 0; i < count; i++) {
                const dot = document.createElement('div');
                dot.className = 'dot' + (i === 0 ? ' active' : '');
                dot.onclick = () => showSlide(i);
                navDots.appendChild(dot);
            }
            
            return navDots;
        }

        async function generatePairing() {
            const loadingElement = document.getElementById('loading');
            const outputSection = document.getElementById('output-section');
            const contentWrapper = document.querySelector('.content-wrapper');
            
            try {
                const userInput = document.getElementById('food-input').value;
                
                if (!userInput.trim()) {
                    alert('Please enter a food description');
                    return;
                }

                loadingElement.style.display = 'block';
                
                const response = await fetch('/generate-pairing', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ input: userInput })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Failed to generate pairing');
                }

                if (data.success && data.art_matches) {
                    contentWrapper.classList.remove('centered');
                    contentWrapper.classList.add('split');
                    
                    const artMatchesContainer = outputSection.querySelector('.art-matches');
                    artMatchesContainer.innerHTML = '';
                    
                    data.art_matches.forEach((match, index) => {
                        const artMatch = document.createElement('div');
                        artMatch.className = `art-match${index === 0 ? ' active' : ''}`;
                        artMatch.innerHTML = `
                            <div class="art-display">
                                <img src="${match.match['Image URL']}" alt="${match.match.Title}">
                            </div>
                            <div class="art-info">
                                <p><strong>Title:</strong> ${match.match.Title}</p>
                                <p><strong>Artist:</strong> ${match.match.Artist}</p>
                                <p><strong>Style:</strong> ${match.match.Style}</p>
                                <p><strong>Category:</strong> ${match.match.Category}</p>
                            </div>
                        `;
                        artMatchesContainer.appendChild(artMatch);
                    });
                    
                    // Add navigation
                        const slideNav = document.createElement('div');
                        slideNav.className = 'slideshow-nav';

                        const prevButton = document.createElement('button');
                        prevButton.textContent = '❮';
                        prevButton.className = 'nav-arrow';
                        prevButton.onclick = () => {
                            currentSlide = (currentSlide - 1 + data.art_matches.length) % data.art_matches.length;
                            showSlide(currentSlide);
                        };

                        const nextButton = document.createElement('button');
                        nextButton.textContent = '❯';
                        nextButton.className = 'nav-arrow';
                        nextButton.onclick = () => {
                            currentSlide = (currentSlide + 1) % data.art_matches.length;
                            showSlide(currentSlide);
                        };

                        const dots = createDots(data.art_matches.length);

                        slideNav.appendChild(prevButton);
                        slideNav.appendChild(dots);
                        slideNav.appendChild(nextButton);
                        artMatchesContainer.appendChild(slideNav);
                    
                    totalSlides = data.art_matches.length;
                    currentSlide = 0;
                    
                    outputSection.style.display = 'block';
                }
            } catch (error) {
                console.error('Error:', error);
                alert(error.message);
            } finally {
                loadingElement.style.display = 'none';
            }
        }

        //functions for about page
        function showAbout() {
            document.getElementById('aboutOverlay').style.display = 'block';
            document.body.style.overflow = 'hidden';
        }

        function hideAbout() {
            document.getElementById('aboutOverlay').style.display = 'none';
            document.body.style.overflow = 'auto';
        }

        // close about overlay when click outside the content
        document.addEventListener('click', function(event) {
            const overlay = document.getElementById('aboutOverlay');
            const aboutContent = overlay.querySelector('.about-content');
            const aboutLink = document.querySelector('.about-link');
            
            if (event.target === overlay && !aboutContent.contains(event.target) && !aboutLink.contains(event.target)) {
                hideAbout();
            }
        });

        // close about overlay when press esc key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                hideAbout();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (totalSlides === 0) return;
            
            if (e.key === 'ArrowLeft') {
                currentSlide = (currentSlide - 1); 
            } else if (e.key === 'ArrowRight') {
                changeSlide(1);
            }
        });

        async function generateAIArt() {
            console.log("Generate AI Art button clicked"); // Debug log
            
            const loadingElement = document.getElementById('loading');
            const outputSection = document.getElementById('output-section');
            const contentWrapper = document.querySelector('.content-wrapper');
            const userInput = document.getElementById('food-input').value;
            
            console.log("User input:", userInput); // Debug log
            
            if (!userInput.trim()) {
                alert('Please enter a food description');
                return;
            }

            try {
                console.log("Starting AI art generation..."); // Debug log
                loadingElement.style.display = 'block';
                
                const response = await fetch('/generate-ai-art', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ input: userInput })
                });

                console.log("Response received"); // Debug log
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Failed to generate AI art');
                }

                console.log("Processing response data"); // Debug log
                if (data.success && data.image_data) {
                    contentWrapper.classList.remove('centered');
                    contentWrapper.classList.add('split');
                    
                    const artMatchesContainer = outputSection.querySelector('.art-matches');
                    artMatchesContainer.innerHTML = '';
                    
                    const artMatch = document.createElement('div');
                    artMatch.className = 'art-match active';
                    artMatch.innerHTML = `
                        <div class="art-display">
                            <img src="${data.image_data}" alt="AI Generated Art">
                        </div>
                        <div class="art-info">
                            <p><strong>Title:</strong> AI Interpretation</p>
                            <p><strong>Generated by:</strong> Stable Diffusion XL</p>
                            <p><strong>Based on:</strong> "${userInput}"</p>
                        </div>
                    `;
                    
                    artMatchesContainer.appendChild(artMatch);
                    outputSection.style.display = 'block';
                    console.log("AI art displayed successfully"); // Debug log
                }
            } catch (error) {
                console.error("Error in generateAIArt:", error); // Debug log
                alert(error.message);
            } finally {
                loadingElement.style.display = 'none';
            }
        }

        // update button creation func + debuggingg 
        function addGenerateOriginalArtButton() {
            console.log("Adding AI Art button"); // Debug log
            const inputSection = document.querySelector('.input-section');
            const generateButton = inputSection.querySelector('button');
            
            const buttonContainer = document.createElement('div');
            buttonContainer.style.display = 'flex';
            buttonContainer.style.gap = '1rem';
            buttonContainer.style.marginTop = '1rem';
            
            generateButton.parentNode.insertBefore(buttonContainer, generateButton.nextSibling);
            buttonContainer.appendChild(generateButton);
            
            const aiArtButton = document.createElement('button');
            aiArtButton.textContent = 'Generate AI Art';
            aiArtButton.onclick = generateAIArt;
            aiArtButton.className = generateButton.className;
            buttonContainer.appendChild(aiArtButton);
            console.log("AI Art button added successfully"); // Debug log
        }

        document.addEventListener('DOMContentLoaded', addGenerateOriginalArtButton);
    </script>

    <div class="about-overlay" id="aboutOverlay">
    <div class="about-content">
        <button class="close-about" onclick="hideAbout()">×</button>
        <h2>About Art and Culinary Pairing</h2>
        <div style="margin-top: 1.5rem;">
            <p>Welcome to Art and Culinary Pairing, a platform that bridges culinary experiences and visual arts. This application uses advanced natural language processing to analyze your food descriptions and match them with complementary artworks that enhance your dining experience.</p>
            
            <p style="margin-top: 1rem;">How it works:</p>
            <ul style="margin-left: 1.5rem; margin-top: 0.5rem;">
                <li>Enter a description of your dish or food experience</li>
                <li>The algorithm analyzes the flavors, textures, and emotions in your description</li>
                <li>We match your food with artworks that complement or enhance your culinary experience</li>
                <li>Browse through the suggested artworks using the navigation arrows</li>
            </ul>
            
            <p style="margin-top: 1rem;">Whether you're a chef looking to create an immersive dining experience, a food enthusiast seeking to enhance your meal, or simply curious about the intersection of culinary arts and visual aesthetics, this tool offers a way to explore these connections.</p>
            
            <p style="margin-top: 1rem;">Use the theme toggle in the top right corner to switch between classic and modern viewing modes.</p>

            <p style="margin-top: 1rem;">_________________________________________________________________________________________</p>

            <p style="margin-top: 1rem;">This product was made in partial fulfillment for my B.A. degree in Computing and the Arts at Yale University. See my senior thesis here. It aligns with my broader interest in the intersection of art, technology, and the culinary space—see more of my project <a href="https://fatart.cargo.site/">here.</a>.</p>
        </div>
    </div>
</div>

</body>

</html>
'''

# load datasets
def load_datasets():
    """Load and prepare the food and art datasets"""
    global food_df, art_df
    try:
        food_df = pd.read_csv(FOOD_CSV_PATH)
        art_df = pd.read_csv(ART_CSV_PATH)
        print(f"Loaded {len(food_df)} food entries and {len(art_df)} art entries")
        return True
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return False

def find_matching_food(input_text):
    """Find the most similar food item to input text"""
    # intialize and fit TF-IDF vectorizer on food descriptions
    food_vectorizer = TfidfVectorizer(stop_words='english')
    food_tfidf_matrix = food_vectorizer.fit_transform(food_df['description'].fillna(''))
    
    # trabsform input text
    input_vector = food_vectorizer.transform([input_text])
    
    # calculate similarities
    similarities = cosine_similarity(input_vector, food_tfidf_matrix).flatten()
    
    # best match
    best_match_idx = similarities.argmax()
    best_match = food_df.iloc[best_match_idx]
    
    return {
        'match': {
            'name': best_match['name'],
            'description': best_match['description']
        },
        'similarity': float(similarities[best_match_idx])
    }

def find_matching_art(food_description, num_matches=3):
    """Find top 3 most similar artworks based on the food description"""
    art_vectorizer = TfidfVectorizer(stop_words='english')
    art_titles = art_df['Title'].fillna('')
    art_tfidf_matrix = art_vectorizer.fit_transform(art_titles)
    
    food_vector = art_vectorizer.transform([food_description])
    similarities = cosine_similarity(food_vector, art_tfidf_matrix).flatten()
    
    # Get indices of top 3 matches
    top_indices = similarities.argsort()[-num_matches:][::-1]
    
    matches = []
    for idx in top_indices:
        art_piece = art_df.iloc[idx]
        matches.append({
            'match': {
                'Title': art_piece['Title'],
                'Artist': art_piece['Artist'],
                'Style': art_piece['Style'],
                'Category': art_piece['Category'],
                'Image URL': art_piece['Image URL']
            },
            'similarity': float(similarities[idx])
        })
    
    return matches

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate-pairing', methods=['POST'])
def generate_pairing():
    try:
        data = request.get_json()
        user_input = data.get('input', '')
        
        if not user_input.strip():
            return jsonify({'error': 'Please enter a food description'}), 400
        
        food_match = find_matching_food(user_input)
        
        if food_match['match']:
            art_matches = find_matching_art(food_match['match']['description'])
            
            if art_matches:
                return jsonify({
                    'success': True,
                    'art_matches': art_matches
                })
            else:
                return jsonify({'error': 'No matching artwork found'}), 404
        else:
            return jsonify({'error': 'No matching recipe found'}), 404
            
    except Exception as e:
        print(f"Error: {str(e)}")  # debugging
        return jsonify({'error': str(e)}), 500
    
@app.route('/generate-ai-art', methods=['POST'])
def generate_ai_art():
    print("Received AI art generation request")
    try:
        data = request.get_json()
        food_description = data.get('input', '')
        print(f"Processing request for: {food_description}")
        
        if not food_description.strip():
            return jsonify({'error': 'Please enter a food description'}), 400

        print("Initializing Stable Diffusion pipeline")
        from diffusers import StableDiffusionPipeline  # regular SD pipeline
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",  # lighter model
            torch_dtype=torch.float32
        )
        
        # force CPU usage
        pipe = pipe.to("cpu")
        
        # Generate image
        print("Generating image...")
        prompt = f"A beautiful photograph of {food_description}, food photography"
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=15,  # fewer steps for faster generation
            guidance_scale=7.0,
        ).images[0]
        
        print("Image generated successfully")
        
        # save and convert image
        temp_path = "temp_generated_art.png"
        image.save(temp_path)
        
        with open(temp_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'image_data': f"data:image/png;base64,{encoded_string}"
        })

    except Exception as e:
        print(f"Error generating AI art: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
print("Loading datasets...")
if not load_datasets():
    print("Failed to load datasets. Please check the file paths and data format.")

if __name__ == '__main__':
    app.run(debug=True)