import spacy
import pandas as pd


#Test the models with sample data
med = pd.read_csv('data/mtsamples.csv', index_col=0)
nlp = spacy.load("en_ner_bc5cdr_md")

transcription = med['transcription'].iloc[1]
doc= nlp(transcription)

# extract and print disease entity
struct_data = []
for ent in doc.ents:
    struct_data.append({ent.label_: ent.text})
print(struct_data)