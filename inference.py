{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🐛 Insect Species Prediction Demo\n",
    "Upload or select an image, run the cell below, and see the model’s prediction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch, timm, os, random\n",
    "from torchvision import transforms\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# ---- paths ----\n",
    "WEIGHTS = 'best_pvt_model.pth'          #@param {type:\"string\"}\n",
    "DATA_DIR = 'dataset'                   #@param {type:\"string\"}\n",
    "\n",
    "# ---- build model ----\n",
    "class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "model = timm.create_model('pvt_v2_b0', pretrained=False, num_classes=len(class_names))\n",
    "model.load_state_dict(torch.load(WEIGHTS, map_location=device))\n",
    "model.to(device).eval()\n",
    "\n",
    "tfm = transforms.Compose([\n",
    "    transforms.Resize((224,224)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])\n",
    "])\n",
    "\n",
    "def predict(img_path):\n",
    "    img = Image.open(img_path).convert('RGB')\n",
    "    x = tfm(img).unsqueeze(0).to(device)\n",
    "    with torch.no_grad():\n",
    "        pred = model(x).argmax(1).item()\n",
    "    plt.imshow(img); plt.axis('off');\n",
    "    plt.title(f'Predicted: {class_names[pred]}'); plt.show()\n",
    "\n",
    "# ---- run on a random validation image ----\n",
    "subdir = random.choice(class_names)\n",
    "img_file = random.choice(os.listdir(os.path.join(DATA_DIR, subdir)))\n",
    "predict(os.path.join(DATA_DIR, subdir, img_file))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
  "language_info": { "name": "python" }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
