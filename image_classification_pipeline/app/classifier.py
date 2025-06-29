import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import io

class ImageClassifier:
    def __init__(self):
        self.model = resnet50(pretrained=True)
        self.model.eval()
        with open("imagenet_classes.txt") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        results = []
        for i in range(top5_prob.size(0)):
            results.append({
                "label": self.classes[top5_catid[i]],
                "confidence": float(top5_prob[i])
            })
        return results

