import os
import numpy as np
import pandas as pd
import spacy
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math



nlp = spacy.load('en_core_web_sm')


def preprocess(text):
    if not isinstance(text,str):
        return ""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def loaddata():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'location.csv')
    data = pd.read_csv(filepath)
    data.fillna('', inplace=True)
    data['processed_country'] = data['Country'].apply(preprocess)
    data['processed_state'] = data['State/Province'].apply(preprocess)
    data['processed_place_name'] = data['PlaceName'].apply(preprocess)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)) 
    tfidf_matrix = vectorizer.fit_transform(data['processed_place_name']) 
    data['combined'] = data['processed_place_name'] + " " + data['processed_state'] + " " + data['processed_country']
    return data, vectorizer, tfidf_matrix

data, vectorizer, tfidf_matrix = loaddata()

def calculate_similarity(query, target):
  if not query or not target:
        return 0
  query_tokens = query.split()
  target_tokens = target.split()
  scores = []
  for t in target_tokens:
        bestmatch = max([fuzz.ratio(t, q) for q in query_tokens]) if query_tokens else 0
        scores.append(bestmatch)
    
  return sum(scores) / len(scores) if scores else 0


def location(query, data):
    if not query:
        return None
    processed = preprocess(query)

    def fuzzyscore(row):
        place = calculate_similarity(processed, row['processed_place_name']) * 0.70
        state = calculate_similarity(processed, row['processed_state']) * 0.20
        country = calculate_similarity(processed, row['processed_country']) * 0.10
        return place + state + country

    data['fuzz_score'] = data.apply(fuzzyscore, axis=1)
    query_vec = vectorizer.transform([processed])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten() * 100
    data['tfidf_score'] = cosine_scores
    data['final_score'] = data['fuzz_score'] * 0.7 + data['tfidf_score'] * 0.3
    best = data.sort_values('final_score', ascending=False).iloc[0] 
    country=best['Country']
    state=best['State/Province']
    place=best['PlaceName']
    lat1 = best['Longitude']
    lon1 = best['Latitude']
    state_name = state
    result = data[data['State/Province'] == state_name]
    longitude_list = result['Longitude'].tolist()
    latitude_list = result['Latitude'].tolist()
    def calculate_distance(lat1, lon1, latitude_list, longitude_list, unit="km"):
        R = 6371
        lat1_rad, lon1_rad = map(math.radians, [lat1, lon1])
        
        distances = []
        seen = set()
        
        for lat2, lon2 in zip(latitude_list, longitude_list):
            if (lat2, lon2) in seen:
                distances.append(None)  
                continue
            
            seen.add((lat2, lon2))
            
            lat2_rad, lon2_rad = map(math.radians, [lat2, lon2])
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance = R * c
            
            if unit == "miles":
                distance *= 0.621371
            
            distances.append(round(distance, 2))
        
        return distances

    distances = calculate_distance(lat1, lon1, latitude_list, longitude_list)
    result['DistanceFromBest'] = distances
    print(f"Best match: {best['PlaceName']} in {best['State/Province']}, {best['Country']} with a score of {best['final_score']:.2f}")
    print(result[['PlaceName', 'State/Province', 'Country', 'DistanceFromBest']].sort_values('DistanceFromBest'))
query = input("Enter your question: ")
location(query, data)




