from transformers import AutoTokenizer, AutoModel
import torch

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

# important: make sure to set padding="max_length" as that's how the model was trained

categories = ["storage furniture", "bucket", "faucet", "folding chair", "laptop", "stapler", "toilet"]


inputs = tokenizer(categories, padding="max_length", return_tensors="pt")
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
print(text_features.shape)  

torch.save(text_features, "siglip_text_features.pt")