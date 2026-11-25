
import cv2
import uuid
import numpy as np

from pathlib import Path
from ultralytics import YOLOE


class ObjectDetector:

    def __init__(self, model_path="yoloe-26s-seg-pf.pt"):
        self.model = YOLOE(model_path)

    def detect_and_segment(self, image_path, output_dir):
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Unable to read input image.")

        height, width = image.shape[:2]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=image,
            imgsz=1024,
            conf=0.15,
            iou=0.50,
            max_det=300,
            device="cpu",
            verbose=False
        )

        extracted_objects = []

        for result in results:

            if result.boxes is None or result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()

            for index, box in enumerate(result.boxes):

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                label = result.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].cpu().numpy()
                )

                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                mask = masks[index]

                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

                mask = (mask > 0.5).astype(np.uint8) * 255

                object_crop = image[y1:y2, x1:x2]
                object_mask = mask[y1:y2, x1:x2]

                if object_crop.size == 0:
                    continue

                rgba = cv2.cvtColor(
                    object_crop,
                    cv2.COLOR_BGR2BGRA
                )

                rgba[:, :, 3] = object_mask

                object_id = str(uuid.uuid4())

                filename = f"{object_id}.png"
                save_path = output_dir / filename

                cv2.imwrite(str(save_path), rgba)

                extracted_objects.append({
                    "object_id": object_id,
                    "name": str(label),
                    "confidence": round(confidence * 100, 2),
                    "box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "image_path": str(save_path),
                    "filename": filename
                })

        return extracted_objects
