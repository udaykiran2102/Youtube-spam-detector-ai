import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

# Define paths
DATASET_PATH = os.path.join('dataset', 'Youtube-Spam-Dataset.csv')
MODEL_SAVE_PATH = 'spam_detector_model'

# Load and preprocess the dataset
def load_and_preprocess_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please ensure the file exists.")
    
    data = pd.read_csv(csv_path)
    
    # Clean the comments: remove special characters, extra spaces, and lowercase
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = text.lower().strip()  # Lowercase and remove extra spaces
        return text
    
    data['CONTENT'] = data['CONTENT'].apply(clean_text)
    data = data[data['CONTENT'] != '']
    
    texts = data['CONTENT'].values
    labels = data['CLASS'].values
    
    return texts, labels

# Custom Dataset class for BERT
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Train the BERT model
def train_model(model, train_loader, val_loader, device, epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Validation
        evaluate_model(model, val_loader, device)

# Evaluate the model
def evaluate_model(model, val_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Not Spam', 'Spam']))
    model.train()

# Load the trained model and tokenizer
def load_model_and_tokenizer(model_path='spam_detector_model'):
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
    else:
        print("Training new model...")
        # Load and preprocess data
        texts, labels = load_and_preprocess_data(DATASET_PATH)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Initialize tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        
        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Create datasets and dataloaders
        train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
        val_dataset = SpamDataset(val_texts, val_labels, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # Train the model
        train_model(model, train_loader, val_loader, device)
        
        # Save the model
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("Model training complete and saved!")
    
    return model, tokenizer

# Preprocess and predict spam
def predict_spam(comment, model, tokenizer, device, max_len=128):
    # Clean the comment
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        return text

    cleaned_comment = clean_text(comment)
    if cleaned_comment == "":
        return "Empty comment. Please provide a valid comment."

    # Tokenize the comment
    encoding = tokenizer.encode_plus(
        cleaned_comment,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

    return "Spam" if prediction == 1 else "Not Spam"

# Main function for user input
def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Spam Detector: Enter a comment to check if it's spam. Type 'exit' to quit.")
    
    while True:
        comment = input("Enter a comment: ")
        if comment.lower() == 'exit':
            print("Exiting Spam Detector.")
            break
        
        result = predict_spam(comment, model, tokenizer, device)
        print(f"Result: {result}\n")

if __name__ == "__main__":
    main()