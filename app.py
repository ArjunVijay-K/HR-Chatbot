from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import chardet
from datetime import datetime

app = Flask(__name__)

class HRChatbot:
    def __init__(self, csv_file_path):
        """Initialize the HR chatbot with FAQ data"""
        self.df = None
        self.vectorizer = None
        self.questions = []
        self.answers = []
        
        try:
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV file not found at {os.path.abspath(csv_file_path)}")
            
            print(f"Found CSV file at: {os.path.abspath(csv_file_path)}")

            with open(csv_file_path, 'rb') as f:
                rawdata = f.read(10000)  
                encoding = chardet.detect(rawdata)['encoding']
                print(f"Detected encoding: {encoding}")

            encodings_to_try = [encoding, 'utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    print(f"Attempting to read with {encoding} encoding...")
                    self.df = pd.read_csv(csv_file_path, encoding=encoding)
                    print(f"Successfully read CSV with {encoding} encoding")
                    
                    if 'question' not in self.df.columns or 'answer' not in self.df.columns:
                        raise ValueError("CSV must contain both 'question' and 'answer' columns")
                    
                    self.df = self.df.dropna(subset=['question', 'answer'])
                    self.df['question'] = self.df['question'].astype(str).str.strip()
                    self.df['answer'] = self.df['answer'].astype(str).str.strip()
                    self.df = self.df[(self.df['question'] != '') & (self.df['answer'] != '')]
                    
                    if len(self.df) == 0:
                        raise ValueError("No valid question-answer pairs found after cleaning")
                    
                    print(f"Loaded {len(self.df)} valid FAQ entries")
                    print("Sample questions:", self.df['question'].head(3).tolist())
                    
                    self.vectorizer = TfidfVectorizer(
                        stop_words='english',
                        lowercase=True,
                        ngram_range=(1, 2),
                        max_features=1000
                    )
                    
                    self.questions = self.df['question'].tolist()
                    self.answers = self.df['answer'].tolist()
                    
                   
                    self.question_vectors = self.vectorizer.fit_transform(self.questions)
                    print("Vectorizer trained successfully")
                    break
                    
                except UnicodeDecodeError:
                    print(f"Failed to decode with {encoding} encoding")
                    continue
                except Exception as e:
                    print(f"Error with {encoding} encoding: {str(e)}")
                    continue
            
            if self.df is None:
                raise ValueError("Failed to read CSV with all attempted encodings")
                
        except Exception as e:
            print(f"\nCRITICAL ERROR IN CHATBOT INITIALIZATION:")
            print(f"Type: {type(e).__name__}")
            print(f"Error: {str(e)}")
            print(f"CSV Path: {os.path.abspath(csv_file_path) if 'csv_file_path' in locals() else 'Unknown'}")
            raise

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        try:
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return ""

    def find_best_match(self, user_query, threshold=0.1):
        """Find the best matching FAQ for user query"""
        try:
            if not user_query or not user_query.strip():
                return None, 0.0
            
            processed_query = self.preprocess_text(user_query)
            if not processed_query:
                return None, 0.0
                
            query_vector = self.vectorizer.transform([processed_query])
            similarity_scores = cosine_similarity(query_vector, self.question_vectors).flatten()
            best_match_idx = np.argmax(similarity_scores)
            best_score = float(similarity_scores[best_match_idx]) 
            
            if best_score >= threshold:
                return {
                    'question': self.questions[best_match_idx],
                    'answer': self.answers[best_match_idx],
                    'confidence': best_score
                }, best_score
            
            return None, best_score
        except Exception as e:
            print(f"Error in find_best_match: {str(e)}")
            return None, 0.0

try:
    print("\n" + "="*50)
    print("Initializing HR Chatbot...")
    print("="*50)
    
    chatbot = HRChatbot('faqs.csv')
    
    print("\nChatbot initialized successfully!")
    print(f"Total FAQs loaded: {len(chatbot.questions)}")
    print("="*50 + "\n")
    
except Exception as e:
    print("\n" + "!"*50)
    print("FATAL ERROR: Chatbot initialization failed")
    print(f"Error: {str(e)}")
    print("!"*50 + "\n")
    chatbot = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not chatbot:
            return jsonify({
                'success': False,
                'error': 'Chatbot service is currently unavailable. Please try again later.'
            })
        
        data = request.get_json()
        user_query = data.get('message', '').strip()
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Please enter a question'
            })
        
        print(f"\nProcessing query: '{user_query}'")
        
        match_result, confidence = chatbot.find_best_match(user_query)
        confidence_float = float(confidence) if confidence is not None else 0.0
        
        if match_result:
            print(f"Found match (confidence: {confidence_float:.2f}): {match_result['question']}")
        else:
            print(f"No good match found (best confidence: {confidence_float:.2f})")
        
        response = {
            'success': True,
            'answer': match_result['answer'] if match_result else 
                     "I couldn't find a specific answer. Would you like me to escalate this to HR?",
            'matched_question': match_result['question'] if match_result else None,
            'confidence': confidence_float,
            'timestamp': datetime.now().strftime('%H:%M'),
            'needs_escalation': bool(not match_result or confidence_float < 0.3)
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"\nError in chat endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Sorry, I encountered an error processing your request. Please try again.'
        })

@app.route('/escalate', methods=['POST'])
def escalate():
    try:
        data = request.get_json()
        user_query = data.get('message', 'No details provided')
        contact_info = data.get('contact_info', 'Not provided')
        
        print(f"\nEscalating query: '{user_query}'")
        print(f"Contact info: {contact_info}")
        
        return jsonify({
            'success': True,
            'message': f'Your query has been escalated to HR. You will receive a response within 24 hours at {contact_info if contact_info else "your registered email"}.',
            'ticket_id': f'HR-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        })
        
    except Exception as e:
        print(f"\nError in escalation: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to escalate your query. Please contact HR directly.'
        })

@app.route('/debug/faqs')
def debug_faqs():
    """Debug endpoint to view loaded FAQs"""
    if not chatbot:
        return jsonify({
            'status': 'error',
            'message': 'Chatbot not initialized'
        }), 500
    
    return jsonify({
        'status': 'success',
        'faq_count': len(chatbot.questions),
        'sample_questions': chatbot.questions[:5],
        'sample_answers': chatbot.answers[:5]
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'running',
        'chatbot_initialized': bool(chatbot),
        'faqs_loaded': len(chatbot.questions) if chatbot else 0,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\nStarting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)