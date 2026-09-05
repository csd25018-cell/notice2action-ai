import json

input_file = "trustextract-n/data/processed/train.jsonl"
output_file = "trustextract-n/data/processed/train_subset.jsonl"

with open(input_file, 'r') as f:
    lines = f.readlines()

# The injected notifications were the last 2 processed chunks, wait, maybe they were multiple chunks?
# Let's filter based on the presence of the text "MINISTRY OF HOME AFFAIRS" or "MINISTRY OF HEALTH" in the tokens.
# Wait, train.jsonl only contains input_ids, not text.
# The tokenizer can decode input_ids to text!

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

subset = []
for line in lines:
    data = json.loads(line)
    text = tokenizer.decode(data["input_ids"])
    if "HOME AFFAIRS" in text or "HEALTH AND FAMILY" in text:
        subset.append(line)

print(f"Found {len(subset)} matching chunks.")

with open(input_file, 'w') as f:
    for line in subset:
        f.write(line)
