import numpy as np
from bentoml import service, api
from bentoml.images import Image
from bentoml.models import HuggingFaceModel

MODEL_ID = "BAAI/bge-m3"


@service(
    image=Image(python_version="3.12").requirements_file("./requirements.txt"),
    resources={"cpu": "2"},
)
class SentenceTransformers:
    model_path = HuggingFaceModel(MODEL_ID)

    def __init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_path, device=self.device)

    @api(batchable=True)
    def encode(
        self,
        sentences: list[str],
    ) -> np.ndarray:
        return self.model.encode(sentences)
