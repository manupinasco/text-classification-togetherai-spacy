import spacy
from spacy.tokens import Doc
from transformers import pipeline
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the labels
labels = ["world","sports","business","sci/tech"]

# Create a blank English spaCy model
nlp = spacy.blank("en")

# Define the custom component
@spacy.Language.factory("zero_shot_text_categorizer")
def create_zero_shot_text_categorizer(nlp, name):
    def zero_shot_text_categorizer(doc):
        # Get the classification results
        result = classifier(doc.text, candidate_labels=labels)
        # Assign the classification results to the custom extension attribute
        doc._.cats = {label: score for label, score in zip(result["labels"], result["scores"])}
        return doc

    # Register the custom extension attribute on the Doc class
    Doc.set_extension("cats", default={}, force=True)
    return zero_shot_text_categorizer

# Add the custom component to the pipeline
nlp.add_pipe("zero_shot_text_categorizer")

# Load the dataset and randomly select the texts
ds = load_dataset("fancyzhx/ag_news")["test"]
ds=[x for x in ds if type(x["label"])!=Ellipsis]
df = pd.DataFrame(ds)
texts,_=train_test_split(ds, train_size=100, stratify=df["label"], random_state=42)
for text in texts:
   text["label"]=labels[text["label"]] 
texts = pd.DataFrame(texts)
# Test the pipeline with the texts
docs = nlp.pipe(texts["text"])
POINTS=0
i=0
for doc in docs:
    if max(doc._.cats,key=doc._.cats.get)==texts["label"][i]:
        POINTS+=1
    i+=1

print("SCORE: "+str(POINTS/100))