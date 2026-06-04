import os
import cv2
import pickle
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

REFERENCE_DIR = r"C:\Storage\maritime_final\refer"

OUTPUT_FILE = r"C:\Storage\maritime_final\data\embeddings\reference_embeddings.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load(
    'facebookresearch/dinov2',
    'dinov2_vitb14'
)

model.eval()
model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def embed(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(rgb)

    tensor = transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(tensor)

    emb = emb.cpu().numpy().flatten()

    emb = emb / np.linalg.norm(emb)

    return emb

embeddings = {}

for file in os.listdir(REFERENCE_DIR):

    path = os.path.join(REFERENCE_DIR, file)

    image = cv2.imread(path)

    if image is None:
        continue

    embeddings[file] = embed(image)

    print(file)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(embeddings, f)

print("DONE")