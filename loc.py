import os
import numpy as np
import pandas as pd
import spacy
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math


nlp = spacy.load('en_core_web_sm')

LABELS = ['A', 'B', 'C', 'D', 'E']


def preprocess(text):
    if not isinstance(text, str):
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


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2)**2
    return round(R * 2 * math.asin(math.sqrt(a)), 2)


def next_closest(lat, lon, nearby, visited_coords):
    col = 'dist_tmp'
    distances = []
    for _, row in nearby.iterrows():
        coord = (row['Latitude'], row['Longitude'])
        if coord in visited_coords:
            distances.append(None)
        else:
            distances.append(haversine(lat, lon, row['Latitude'], row['Longitude']))
    nearby = nearby.copy()
    nearby[col] = distances
    valid = nearby.dropna(subset=[col])
    valid = valid[valid[col] > 0]
    if valid.empty:
        return None, None
    best_idx = valid[col].idxmin()
    return valid.loc[best_idx], valid.loc[best_idx, col]


def score_and_rank(processed, data):
    def fuzzyscore(row):
        place = calculate_similarity(processed, row['processed_place_name']) * 0.70
        state  = calculate_similarity(processed, row['processed_state'])      * 0.20
        country = calculate_similarity(processed, row['processed_country'])   * 0.10
        return place + state + country

    scored = data.copy()
    scored['fuzz_score'] = scored.apply(fuzzyscore, axis=1)
    query_vec = vectorizer.transform([processed])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten() * 100
    scored['tfidf_score'] = cosine_scores
    scored['final_score'] = scored['fuzz_score'] * 0.7 + scored['tfidf_score'] * 0.3
    return scored.sort_values('final_score', ascending=False)


def build_chain(best_row, data, max_stops=None):
    if max_stops is None:
        max_stops = len(LABELS)
    state_name = best_row['State/Province']
    nearby = data[data['State/Province'] == state_name].copy()
    chain = [best_row]
    visited = {(best_row['Latitude'], best_row['Longitude'])}

    for _ in range(max_stops - 1):
        prev = chain[-1]
        row, _ = next_closest(prev['Latitude'], prev['Longitude'], nearby, visited)
        if row is None:
            break
        chain.append(row)
        visited.add((row['Latitude'], row['Longitude']))
    return chain


def location(query, data):
    if not query:
        return None
    processed = preprocess(query)
    ranked = score_and_rank(processed, data)
    best = ranked.iloc[0]
    chain = build_chain(best, data)

    print()
    print("=" * 52)
    for i, row in enumerate(chain):
        label = LABELS[i]
        print(f"  Location {label}: {row['PlaceName']}")
        print(f"             {row['State/Province']}, {row['Country']}")
        print(f"             Lat {row['Latitude']}  |  Lon {row['Longitude']}")
        if i > 0:
            prev = chain[i - 1]
            dist = haversine(prev['Latitude'], prev['Longitude'], row['Latitude'], row['Longitude'])
            print(f"             Distance {LABELS[i-1]} → {label}: {dist} km")
        if i < len(chain) - 1:
            print(f"             ↓")
    print("=" * 52)


query = input("Enter your question: ")
location(query, data)

