from http.server import BaseHTTPRequestHandler
from urllib import parse
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        s = self.path
        dic = dict(parse.parse_qsl(parse.urlsplit(s).query))
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()


        nlp = spacy.load("en_core_web_sm")
        existing_data = pd.read_csv(r'api/ipc_sections.csv')
        def extract_entities(text):
            doc = nlp(text)
            return [ent.text.lower() for ent in doc.ents]
        existing_data['Entities'] = existing_data['Offense'].apply(extract_entities).dropna()
        flattened_data = existing_data[['Section', 'Punishment', 'Entities']].explode('Entities')
        label_encoder = LabelEncoder()
        flattened_data['Section_Label'] = label_encoder.fit_transform(flattened_data['Section'])
        def extract_entities(text):
            doc = nlp(text)
            return ' '.join([ent.text.lower() for ent in doc.ents])
        existing_data['Entities'] = existing_data['Offense'].apply(extract_entities)
        existing_data.count()
        existing_data['Entities'].sample(10)
        flattened_data['Entities'] = flattened_data['Entities'].fillna('')
        tfidf_vectorizer = TfidfVectorizer(lowercase=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform(flattened_data['Entities'])
        classifier = SVC(kernel='linear')
        classifier.fit(tfidf_matrix, flattened_data['Section_Label'])
        def predict_section_and_punishment(user_input):
            user_entities = extract_entities(user_input)    
            if isinstance(user_entities, str):
                user_entities = [user_entities]    
            user_tfidf = tfidf_vectorizer.transform(user_entities)
            predicted_label = classifier.predict(user_tfidf)    
            predicted_section = label_encoder.inverse_transform(predicted_label)    
            punishment = existing_data[existing_data['Section'] == predicted_section[0]]['Punishment'].iloc[0]    
            return predicted_section[0], punishment
            # return predicted_section[0]
            # return {'User Input':'predicted_section[0]'}
        if "userInput" in dic: 
            user_input=dic["userInput"]
        else:
            user_input = '''I am writing to report an attempted murder incident that occurred on December 10, 2023, at 11:00 PM, in [Location]. The victim, Mr. Ram Kumar, narrowly escaped harm during this event. I urgently seek your attention to this matter for a swift investigation.'''
        section, punishment = predict_section_and_punishment(user_input)
        # if section and punishment:
        #     print(f"User Input: {user_input}")
        #     print(f"Predicted Section: {section}")
        #     print(f"Predicted Punishment: {punishment}")
        # else:
        #     print("No match found for the input.")





        self.wfile.write(section.encode())
        return