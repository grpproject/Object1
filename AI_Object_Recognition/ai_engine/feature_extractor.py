
import torch
import numpy as np

from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class FeatureExtractor:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = "facebook/dinov2-base"

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

        self.model = AutoModel.from_pretrained(
            self.model_name
        ).to(self.device)

        self.model.eval()

    def extract_features(self, image_path):

        # image = Image.open(image_path).convert("RGB")
        image = Image.open(image_path).convert("RGBA")

        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model(**inputs)

            features = outputs.last_hidden_state[:, 0, :]

            features = torch.nn.functional.normalize(
                features,
                p=2,
                dim=1
            )

        return features[0].cpu().numpy().astype(np.float32)
