import torch
from transformers import BertTokenizerFast
from model import BiLSTM_CRF  # Import your BiLSTM_CRF model class

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

import streamlit as st
from typing import List, Dict

# Initialize the tokenizer and model parameters
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
vocab_size = tokenizer.vocab_size

# Define the label list, which should be consistent with ENTITY_COLORS in gui.py
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
tagset_size = len(label_list)

# Initialize the BiLSTM-CRF model and load its state
model = BiLSTM_CRF(vocab_size, tagset_size)
model.load_state_dict(torch.load('model_backup.pth'))
model.eval()

# Map nltk.ne_chunk labels to your ENTITY_COLORS labels
nltk_to_entity_colors_map = {
    "PERSON": "B-PER",
    "ORGANIZATION": "B-ORG",
    "GPE": "B-LOC",
    "LOCATION": "B-LOC",
    "FACILITY": "B-MISC",
    "PRODUCT": "B-MISC",
    # Add other mappings as needed
}

# Define colors for different entity types
ENTITY_COLORS = {
    "B-PER": "#17a2b8",  # Person (blue)
    "I-PER": "#17a2b8",  # Continuation of Person (blue)
    "B-ORG": "#fd7e14",  # Organization (orange)
    "I-ORG": "#fd7e14",  # Continuation of Organization (orange)
    "B-LOC": "#ffc107",  # Location (yellow)
    "I-LOC": "#ffc107",  # Continuation of Location (yellow)
    "B-MISC": "#6c757d",  # Miscellaneous (gray)
    "I-MISC": "#6c757d"   # Continuation of Miscellaneous (gray)
}

def predict(text: str) -> List[Dict]:
    """Perform NER prediction on the input text."""
    results = []
    tokens = nltk.sent_tokenize(text)

    # Process each sentence
    for token in tokens:
        # Use NLTK's named entity chunker
        chunks = ne_chunk(pos_tag(word_tokenize(token)))
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                # Get entity text and map to our label format
                entity = ' '.join(c[0] for c in chunk)
                start = token.index(entity)
                end = start + len(entity)
                label = nltk_to_entity_colors_map.get(chunk.label(), "O")  # Default to "O" if not mapped
                
                # Append result with mapped label
                results.append({"entity": entity, "start": start, "end": end, "tag": label})
            else:
                # Non-entity word (we'll skip these in highlighting if the tag is "O")
                word = chunk[0]
                start = token.index(word)
                end = start + len(word)
                results.append({"entity": word, "start": start, "end": end, "tag": "O"})
    
    return results

def highlight_entities(text: str, entities: List[Dict]) -> str:
    """Highlight entities in text using HTML with labeled color boxes."""
    highlighted_text = ""
    last_idx = 0
    
    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        label = entity["tag"]
        
        # Skip non-entity tokens
        if label == "O" or label == "":
            continue
        
        color = ENTITY_COLORS.get(label, "#ff0000")  # Use red for unmatched labels
        
        # Get the entity type (e.g., "PER", "ORG") if the label has a hyphen
        label_text = label.split("-")[1] if "-" in label else label

        # Add non-entity text
        highlighted_text += text[last_idx:start]
        
        # Add entity text with highlighted color and label
        highlighted_text += (
            f'<span style="background-color: {color}; color: white; padding: 4px 6px; border-radius: 4px; '
            f'margin: 0 3px; display: inline-block; font-weight: bold;">'
            f'{text[start:end]} '
            f'<span style="background-color: #fff; color: {color}; padding: 2px 4px; font-size: 0.8em; '
            f'border-radius: 3px; margin-left: 4px;">{label_text}</span>'
            f'</span>'
        )
        
        # Update last index
        last_idx = end

    # Add remaining text after the last entity
    highlighted_text += text[last_idx:]
    
    return highlighted_text


# Streamlit interface
st.title("Enhanced Named Entity Recognition (NER) Web App")
st.write("Enter text and see the named entities highlighted by the BiLSTM-CRF model.")

# Text input from the user
input_text = st.text_area("Enter text to analyze", "Type some text here...")

if st.button("Get NER Predictions"):
    if input_text.strip():
        # Perform prediction
        predictions = predict(input_text)

        # Highlight entities in the text
        highlighted_text = highlight_entities(input_text, predictions)
        
        # Display highlighted text using Streamlit's Markdown with unsafe_allow_html
        st.write("NER Predictions (highlighted):")
        st.markdown(
            f'<div style="line-height: 1.6; font-size: 1.2em; font-family: Arial, sans-serif;">{highlighted_text}</div>', 
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter some text for NER analysis.")
