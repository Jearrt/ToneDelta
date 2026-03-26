# python3 evaluate.py

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from datasets import load_dataset
import matplotlib.pyplot as plt

# 1. Load Financial PhraseBank
ds = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", "5768")
df = ds["train"].to_pandas()
# confirmed columns: "sentence" (string) and "label" (0=negative, 1=neutral, 2=positive)

# 2. Map numeric labels to text
label_map = {0: "negative", 1: "neutral", 2: "positive"}
ground_truth = [label_map[l] for l in df["label"]]

# 3. Load FinBERT like in analyse.py 
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
classifier = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

# 4. Run predictions
predictions = [classifier(text, truncation=True, max_length=512)[0]["label"].lower() for text in df["sentence"]]

# 5. Print results
print(f"\nAccuracy: {accuracy_score(ground_truth, predictions):.4f}")
print("\nDetailed Report:")
print()
print(classification_report(ground_truth, predictions))

# For the confusion matrix (to see where each misclassification landed)
labels = ["negative", "neutral", "positive"]
cm = confusion_matrix(ground_truth, predictions, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.savefig("confusion_matrix.png")
print(cm)
