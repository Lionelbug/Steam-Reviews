import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, CamembertModel
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE


# Hyperparamètres
SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 2
PRETRAINED_MODEL = "camembert-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Charger le fichier de mappage
with open("label_mappings.json", "r", encoding="utf-8") as f:
    mappings = json.load(f)
label2Index = mappings["label2Index"]
index2label = {int(k): v for k, v in mappings["index2label"].items()}

df_train = pd.read_csv('train.csv')
df_val = pd.read_csv('val.csv')
df_test = pd.read_csv('test.csv')

# Tokenizer
tokenizer = CamembertTokenizer.from_pretrained(PRETRAINED_MODEL)

# Dataset
class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, seq_len):
        self.texts = df['text'].tolist()
        self.labels = df['labels'].tolist()
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.seq_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


vectorizer = TfidfVectorizer(max_features=5000)  # ou autre méthode, mais pas BERT directement ici
X_tfidf = vectorizer.fit_transform(df_train['text']).toarray()
y = df_train['labels']

# Application de SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Reconstruction du DataFrame équilibré
texts_resampled = vectorizer.inverse_transform(X_resampled)
texts_resampled = [" ".join(text) for text in texts_resampled]  # reconvertir en phrases

df_train_balanced = pd.DataFrame({'text': texts_resampled, 'labels': y_resampled})



# Créer Dataset et DataLoader
train_dataset = BERTDataset(df_train_balanced, tokenizer, SEQ_LEN)
val_dataset   = BERTDataset(df_val, tokenizer, SEQ_LEN)
test_dataset  = BERTDataset(df_test, tokenizer, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# BERT + Dropout + Linear
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = CamembertModel.from_pretrained(PRETRAINED_MODEL)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Utiliser le vecteur du token [CLS]
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        return logits
model = BERTClassifier(num_classes=len(label2Index)).to(DEVICE)

# Entrênement
def train(model, train_loader, val_loader, epochs, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(DEVICE)

    train_accs, val_accs = [], []  # Pour stocker les courbes

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total += labels.size(0)

            loop.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Batch Acc": f"{correct / labels.size(0):.4f}"
            })

        # Entraînement terminé pour cette époque
        train_acc = total_correct / total
        train_accs.append(train_acc)

        # Évaluation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1} Summary — Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return train_accs, val_accs

train_accs, val_accs = train(model, train_loader, val_loader, EPOCHS)

# plot
def plot_graph(train, val, title):
    plt.plot(train, label='Train')
    plt.plot(val, label='Validation')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

plot_graph(train_accs, val_accs, "Accuracy")

# Évaluation
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # Mapper les indices aux noms des étiquettes
    y_true = [index2label[i] for i in y_true]
    y_pred = [index2label[i] for i in y_pred]

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    return y_true, y_pred




y_true, y_pred = evaluate(model, test_loader)
torch.save(model.state_dict(), "camembert_classifier.pth")
