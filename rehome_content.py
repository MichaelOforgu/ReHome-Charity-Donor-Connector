#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import math
import argparse

# Load the datasets
D = pd.read_csv("donor_dataset.csv")
C = pd.read_csv("charity_data.csv")

# Get a list of the possible preferences
prefs = []
for col in [x for x in D.columns if "Pref_" in x]:
    prefs = prefs + D[col].unique().tolist()
prefs = list(set(prefs))

# Get a list of all the possible Charity Categories
tmp = C.Causes.str.split('-', expand=True)
causes = []
for col in tmp:
    causes = causes + tmp[col].str.strip().str.title().unique().tolist()
causes = list(set(causes))
causes = [x for x in causes if x is not None]

# Clean specific erroneous data entry
C.loc[416, 'Causes'] = C.loc[416, 'Causes'].replace(' Youth - Elderly  TRENDLEWOOD CHURCH Vowles Cl, Wraxall, Bristol BS48 1PP, UK  Religious - Children /', '')

# Ensure 'causes_list' column is created with the correct dtype
C['causes_list'] = [[] for _ in range(len(C))]

# Populate the 'causes_list' column
for x in C.index:
    tmp = []
    for cause in causes:
        if cause.lower() in C.loc[x, 'Causes'].lower():
            tmp.append(cause)
    C.at[x, 'causes_list'] = tmp

# Delete the "Causes" column as it is a string and hard to find causes there
C = C.drop('Causes', axis=1)

# Combine the Causes and the preferences.
pref_causes = {
    'Human Rights': ['Human Rights', 'Ethnic / Racial', 'Disability', 'Economic / Community'],
    'Disaster Relief': ['Overseas Aid / Famine', 'Housing'],
    'Religious': ['Religious'],
    'Women': ['Human Rights', 'Children / Youth', 'Health / Medical', 'Economic / Community'],
    'Poverty': ['Poverty'],
    'Elderly Care': ['Elderly'],
    'Healthcare': ['Health / Medical'],
    'Environment': ['Environment'],
    'Education': ['Education / Training'],
    'Animals': ['Animals']
}

def charities_by_cause(pref):
    tmp = pd.DataFrame()
    for p in range(len(pref_causes[pref])):
        tmp = pd.concat([tmp, C[C.causes_list.apply(lambda x: True if pref_causes[pref][p] in x else False)]])
    return tmp

def haversine_distance(coord1, coord2):
    R = 3958.8  # Radius of Earth in miles
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def get_nearby_charities(location, radius=5):
    def distance(row):
        return haversine_distance(location, (row['Latitude'], row['Longitude']))
    
    C['distance'] = C.apply(distance, axis=1)
    nearby_charities = C[C['distance'] <= radius]
    return nearby_charities

def encode_donation_types(donation_types, preferences):
    return [1 if t in preferences else 0 for t in donation_types]

def sort_by_distance(user, tmp):
    Distance = tmp.drop(['Name', 'causes_list'], axis=1).copy()
    Distance.Latitude = np.abs(Distance.Latitude - user.Latitude)
    Distance.Longitude = np.abs(Distance.Longitude - user.Longitude)
    Distance['Distance (miles)'] = np.sqrt(Distance.Latitude ** 2 + Distance.Longitude ** 2) * 97.55 # convert to miles 
    result = pd.merge(Distance.sort_values(by='Distance (miles)'), C, on='Charity ID')
    return result.drop_duplicates(subset=['Charity ID'], keep='first')

def finder_scenario_1(user):
    tmp = pd.DataFrame()
    for col in [x for x in D.columns if "Pref_" in x]:
        subset = charities_by_cause(user[col])
        subset['Preference'] = user[col]
        tmp = pd.concat([tmp, subset])
    return sort_by_distance(user, tmp)[['Name', 'Distance (miles)', 'Preference']]

def finder_scenario_2(user):
    result = pd.DataFrame()
    for col in [x for x in D.columns if "Pref_" in x]:
        subset = sort_by_distance(user, charities_by_cause(user[col])).head(10)
        subset['Preference'] = user[col]
        result = pd.concat([result, subset])
    return result[['Name', 'Charity ID', 'Distance (miles)', 'Preference']].drop_duplicates(keep='first').reset_index(drop=True)

def finder_scenario_3(user):
    result = pd.DataFrame()
    max_dist = sort_by_distance(user, C)['Distance (miles)'].max()
    for i, col in enumerate([x for x in D.columns if "Pref_" in x]):
        partial_result = sort_by_distance(user, charities_by_cause(user[col]))
        partial_result['tmp_dist'] = partial_result['Distance (miles)'] / max_dist  # Normalize Distances
        partial_result['tmp_pref'] = i / 5  # Create an index to account for preferences, the lower the better
        partial_result['Preference'] = user[col]
        result = pd.concat([result, partial_result])
    result['metric'] = np.sqrt(result.tmp_dist ** 2 + result.tmp_pref ** 2)
    return result.sort_values(by='metric').drop_duplicates(subset=['Charity ID'], keep='last').reset_index(drop=True)[['Name', 'Distance (miles)', 'Preference']]

def content_based_recommendation(donor_df, charity_df, n_recommendations=5):
    charity_df['causes_list'] = charity_df['Causes'].str.split('-').apply(lambda x: [cause.strip().title() for cause in x])
    mlb = MultiLabelBinarizer()
    charity_features = mlb.fit_transform(charity_df['causes_list'])
    charity_feature_df = pd.DataFrame(charity_features, columns=mlb.classes_)
    charity_df = charity_df.join(charity_feature_df).drop('causes_list', axis=1)

    donor_df['preferences'] = donor_df.apply(lambda row: [row[f"Pref_{i}"].strip().title() for i in range(1, 6)], axis=1)
    donor_preferences = mlb.transform(donor_df['preferences'])
    donor_preference_df = pd.DataFrame(donor_preferences, columns=mlb.classes_)
    donor_df = donor_df.join(donor_preference_df).drop('preferences', axis=1)

    similarity_matrix = cosine_similarity(donor_preference_df.values, charity_feature_df.values)
    similarity_df = pd.DataFrame(similarity_matrix, index=donor_df.index, columns=charity_df.index)

    def get_top_n_recommendations(donor_id, n=n_recommendations):
        donor_similarities = similarity_df.loc[donor_id]
        top_n_charities = donor_similarities.sort_values(ascending=False).head(n).index
        return charity_df.loc[top_n_charities]

    recommendations = {}
    for donor_id in donor_df.index:
        recommendations[donor_id] = get_top_n_recommendations(donor_id, n=n_recommendations)

    return recommendations

def main(donor_id, scenario=1, radius=5):
    user = D.loc[donor_id, :]
    if scenario == 1:
        result = finder_scenario_1(user)
    elif scenario == 2:
        result = finder_scenario_2(user)
    elif scenario == 3:
        result = finder_scenario_3(user)
    elif scenario == 4:
        result = content_based_recommendation(D, C, n_recommendations=5)[donor_id]
    else:
        raise ValueError("Invalid scenario selected. Choose between 1, 2, 3, or 4.")
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Charity Recommendation System')
    parser.add_argument('donor_id', type=int, help='ID of the donor')
    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2, 3, 4], help='Scenario number to run (1, 2, 3, or 4)')
    parser.add_argument('--radius', type=float, default=5, help='Radius for nearby charity search in miles (only for scenario 1)')
    args = parser.parse_args()

    main(args.donor_id, args.scenario, args.radius)

def main(donor_id, scenario=1, radius=5):
    user = D.loc[donor_id, :]
    if scenario == 1:
        result = finder_scenario_1(user)
    elif scenario == 2:
        result = finder_scenario_2(user)
    elif scenario == 3:
        result = finder_scenario_3(user)
    elif scenario == 4:
        result = content_based_recommendation(D, C, n_recommendations=5)[donor_id]
    else:
        raise ValueError("Invalid scenario selected. Choose between 1, 2, 3, or 4.")
    return result
