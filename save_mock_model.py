import json
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_name = "google/muril-base-cased"
output_dir = "output/baseline_model"
os.makedirs(output_dir, exist_ok=True)

# We need the 15 labels to match the schema
labels = ['O', 'B-TITLE', 'I-TITLE', 'B-AUTHORITY', 'I-AUTHORITY', 'B-AUDIENCE', 'I-AUDIENCE', 'B-ELIGIBILITY', 'I-ELIGIBILITY', 'B-DOCUMENT', 'I-DOCUMENT', 'B-DATE', 'I-DATE', 'B-CONTACT', 'I-CONTACT']
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}

print("Downloading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

print(f"Saving to {output_dir}...")
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

os.makedirs("models/trustextract", exist_ok=True)
with open("models/trustextract/calibration.json", "w") as f:
    json.dump({"temperature": 1.5, "thresholds": {}}, f)
print("Done.")
