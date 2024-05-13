import warnings

from numpy.linalg import norm
from numpy import dot
from insightface.app import FaceAnalysis

warnings.simplefilter('ignore', FutureWarning)

class FaceSimularity:
    def __init__(self) -> None:
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'], rcond=None)
        self.app.prepare(ctx_id=0, det_size=(256, 256))

    def get_embedding(self, image):
        result = self.app.get(image)
        if len(result) == 0:
            return None
        return result[0].embedding
    
    def cosine_simularity(self, image_1, image_2):
        embed_1 = self.get_embedding(image_1)
        embed_2 = self.get_embedding(image_2)
        if embed_1 is None or embed_2 is None:
            return 0
        return dot(embed_1, embed_2) / (norm(embed_1) * norm(embed_2))
