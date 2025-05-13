import torch
from transformers import CamembertTokenizer, CamembertModel
from torch import nn

# Définir la même architecture que lors de l'entraînement
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = CamembertModel.from_pretrained("camembert-base")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        return logits

label2Index = {"négatif": 0, "positif": 1}

# Charger le modèle (ajustez num_classes selon votre cas)
model = BERTClassifier(num_classes=len(label2Index))  # Remplacez label2Index par votre nombre de classes
model.load_state_dict(torch.load("camembert_classifier.pth"))
model.eval()

# Exemple de liste de commentaires
text = [
    "Ce jeu est incroyable ! Les graphismes sont époustouflants et l'histoire est captivante.",
    "Une grosse déception. Les graphismes sont médiocres et la jouabilité est plate.",
    "J'adore ce jeu ! La prise en main est facile, et les niveaux sont super bien conçus.",
    "Je n'ai pas du tout aimé ce jeu. Trop répétitif et sans intérêt.",
    "Un chef-d'œuvre absolu, je ne peux pas m'arrêter de jouer. Je le recommande à tout le monde.",
    "Ce jeu, quelle claque !",
    "Les bugs sont partout dans ce jeu. C'est frustrant et ça gâche toute l'expérience."
]

# Tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# Prédiction pour chaque commentaire de la liste `text`
for comment in text:
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        prediction = torch.argmax(outputs, dim=1).item()

    sentiment = "Positif" if prediction == 1 else "Négatif"
    print(f"Commentaire: {comment}\nClasse prédite: {sentiment}\n")
