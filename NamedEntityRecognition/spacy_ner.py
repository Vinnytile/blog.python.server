import pandas as pd
import numpy as np 
import spacy 
import en_core_web_sm

nlp = en_core_web_sm.load()

cities = pd.read_csv('us_cities_states_counties.csv')
cities['City alias'] = cities['City alias'].apply(lambda x: str(x))

# GPE = Countries, cities, states.
count = 0
passed = 0
for i, city in enumerate(cities['City alias'].values):
    try:
        doc = nlp(city)
        for X in doc.ents: 
            if X.label_=='GPE': 
                count+=1
    except:
        passed +=1
        pass
    if i% 5000 == 0: print (i, count, passed)
print(f'Spacy knows {count} out of {cities.shape[0]}')
print('couldnt process:', passed)
