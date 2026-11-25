
from ai_engine.detector import ObjectDetector


class ObjectSegmenter:

    def __init__(self):
        self.detector = ObjectDetector()

    def extract_objects(self, image_path, output_dir):
        return self.detector.detect_and_segment(
            image_path,
            output_dir
        )
