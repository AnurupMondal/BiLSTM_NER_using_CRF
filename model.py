import torch
import torch.nn as nn
import torch.optim as optim
from TorchCRF import CRF
import joblib
from datasets import load_from_disk
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

dataset = load_from_disk("..\conll2003")
# Check dataset structure
print(dataset)

# Display a sample example from the training set
print("\nSample from training set:")
print(dataset['train'][0])

# Extract the label names and create a mapping
label_list = dataset['train'].features['ner_tags'].feature.names
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
print("Label to ID mapping:", label_to_id)


# Initialize the fast tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, padding="max_length", max_length=128, is_split_into_words=True)

    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to word IDs
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignore subword tokens
            elif word_id != previous_word_id:
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization and alignment
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)


# Set format to PyTorch for using with DataLoader
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Create DataLoaders
train_loader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=16)
test_loader = DataLoader(tokenized_datasets["test"], batch_size=16)

# Define the BiLSTM-CRF model
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 75, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(150, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences, tags=None):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)

        if tags is not None:
            loss = -self.crf(lstm_feats, tags, reduction='mean')
            return loss
        else:
            return self.crf.decode(lstm_feats)

# Define device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# Model parameters
vocab_size = tokenizer.vocab_size  # Use the tokenizer's vocabulary size
tagset_size = len(label_list)  # Number of unique labels in the dataset
# Initialize the model and move it to the appropriate device
model = BiLSTM_CRF(vocab_size, tagset_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
print(model)


def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            sentences = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(sentences)

            for i in range(len(predictions)):
                pred_tags = predictions[i]
                true_tags = labels[i][:len(pred_tags)].tolist()

                # Filter out padding (-100) labels in `true_tags`
                filtered_true_tags = [tag for tag in true_tags if tag != -100]
                filtered_pred_tags = pred_tags[:len(filtered_true_tags)]

                all_preds.extend(filtered_pred_tags)
                all_labels.extend(filtered_true_tags)

    # Calculate metrics using aligned labels
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return precision, recall, f1, accuracy


def train_model(model, train_loader, val_loader, optimizer, epochs=10):
    history = {'train_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_accuracy': []}
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for batch in train_loader:
            sentences = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Filter out -100 values in labels (set them to a valid tag, e.g., 0)
            labels[labels == -100] = 0  # Choose any valid label index as placeholder

            optimizer.zero_grad()
            loss = model(sentences, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        val_precision, val_recall, val_f1, val_accuracy = evaluate_model(model, val_loader)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
              f"Val F1: {val_f1:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return history

if __name__ == "__main__":
    #train and save the model
    history = train_model(model, train_loader, val_loader, optimizer, epochs=10)

    # Save the model using torch save
    torch.save(model.state_dict(), 'model.pth')


    # joblib.dump(model.state_dict(), 'model.joblib')
    # print("Model saved as model.joblib")
