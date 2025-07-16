from transformers import AutoTokenizer, AutoModel
import torch

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

# important: make sure to set padding="max_length" as that's how the model was trained

# categories = ["storage furniture", "bucket", "faucet", "folding chair", "laptop", "stapler", "toilet"]
categories = [
    "open the storage furniture",
    "bucket",
    "faucet",
    "open the folding chair",
    "open the laptop",
    "open the stapler",
    "open the toilet",
    "close the storage furniture",
    "close the folding chair",
    "close the laptop",
    "close the stapler",
    "close the toilet",
]


inputs = tokenizer(categories, padding="max_length", return_tensors="pt")
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
print(text_features.shape)  
print(text_features)  # should be float16
torch.save(text_features, "siglip_text_features.pt")