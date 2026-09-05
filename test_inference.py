import json
import torch
from transformers import AutoTokenizer, AutoConfig
import sys
sys.path.append("trustextract-n")
from src.training.model import TrustExtractMultiTaskModel

# Use the text from Document 2
text = "MINISTRY OF HEALTH AND FAMILY WELFARE (Department of Health and Family Welfare) NOTIFICATION New Delhi, the 19th August, 2026 G.S.R. 745(E).— The following draft of certain rules further to amend the Drugs Rules, 1945, which the Central Government proposes to make..."

model_dir = "trustextract-n/output/baseline_model/checkpoint-37"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)

doc_types = ["Circular", "Notification", "Ordinance", "Draft Rule"]

model = TrustExtractMultiTaskModel.from_pretrained(
    model_dir, 
    config=config,
    num_token_labels=len(config.id2label),
    num_doc_labels=len(doc_types)
)
model.eval()

inputs = tokenizer(
    text, 
    return_tensors="pt", 
    truncation=True, 
    max_length=512, 
    return_offsets_mapping=True
)
offset_mapping = inputs.pop("offset_mapping")[0]

with torch.no_grad():
    outputs = model(**inputs)

doc_probs = torch.softmax(outputs.doc_logits, dim=-1)[0]
doc_pred = int(torch.argmax(doc_probs))
print(f"Document Type Prediction: {doc_types[doc_pred]} (Confidence: {float(doc_probs[doc_pred]):.2f})")

token_probs = torch.softmax(outputs.logits, dim=-1)[0]
preds = torch.argmax(token_probs, dim=-1)

print("\nExtracted Entities:")
for i, pred_idx in enumerate(preds):
    label = config.id2label[pred_idx.item()]
    if label != "O" and label != -100:
        start, end = offset_mapping[i].tolist()
        if start != end:
            print(f"- {label}: '{text[start:end]}' (Score: {float(token_probs[i, pred_idx]):.2f})")
