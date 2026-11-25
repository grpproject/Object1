
import numpy as np

from ai_engine.database import get_all_objects


class ObjectRecognizer:

    def __init__(self, similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold

    def find_same_object(self, query_embedding, category):

        database_objects = get_all_objects()

        if not database_objects:
            return {
                "matched": False,
                "message": "Object database is empty."
            }

        best_match = None
        best_similarity = -1.0

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        for stored_object in database_objects:

            if stored_object["name"].lower() != category.lower():
                continue

            stored_embedding = stored_object["embedding"]

            similarity = float(
                np.dot(query_embedding, stored_embedding)
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_match = stored_object

        if best_match is None:
            return {
                "matched": False,
                "message": "No stored object of this category."
            }

        similarity_percentage = round(
            max(0.0, best_similarity) * 100,
            2
        )

        matched = best_similarity >= self.similarity_threshold

        return {
            "matched": matched,
            "object_id": best_match["id"] if matched else None,
            "object_name": best_match["name"],
            "similarity": similarity_percentage,
            "message": (
                "Possible same-object match"
                if matched
                else "No reliable same-object match"
            )
        }

x = 87