"""Single-sample prediction using the all-data-trained Semantic IRT model."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
import torch.nn.functional as F

from models.Semantic_IRT import Semantic_IRT
import utils


_PROJECT_ROOT = Path(__file__).resolve().parent
_ALL_DATA_DIR = _PROJECT_ROOT / "results" / "chunked" / "all_data"


@dataclass
class _Predictor:
    model: Semantic_IRT
    student_indices: dict[str, int]
    mean_ability: float
    embedding_dimension: int
    device: torch.device


@lru_cache(maxsize=1)
def _get_predictor() -> _Predictor:
    """Load model and inference metadata once per process."""
    summary_path = _ALL_DATA_DIR / "training_summary.json"
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)

    if summary.get("text_encoding") != utils.TEXT_ENCODING:
        raise ValueError("Retrain with the current chunked embeddings before prediction.")

    model_type = summary["model_type"]
    num_students = int(summary["subjects"])
    embedding_dimension = int(summary["embedding_dimension"])

    abilities_path = (
        _ALL_DATA_DIR / f"student_abilities_semantic_{model_type}.csv"
    )
    abilities = pd.read_csv(abilities_path).sort_values("student_id")
    student_ids = abilities["student_id"].tolist()
    if student_ids != list(range(num_students)):
        raise ValueError(
            f"{abilities_path} must contain contiguous student_id values "
            f"from 0 to {num_students - 1}"
        )

    student_names = abilities["normalized_name"].tolist()
    if len(set(student_names)) != num_students:
        raise ValueError(f"{abilities_path} contains duplicate subject names")
    student_indices = {
        name: index for index, name in enumerate(student_names)
    }
    mean_ability = float(abilities["ability"].mean())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Semantic_IRT(
        num_students=num_students,
        item_embed_dim=embedding_dimension,
        model_type=model_type,
    ).to(device)
    checkpoint_path = _ALL_DATA_DIR / "semantic_irt_model.pt"
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return _Predictor(
        model=model,
        student_indices=student_indices,
        mean_ability=mean_ability,
        embedding_dimension=embedding_dimension,
        device=device,
    )


def predict(input: list[dict], labeled: list) -> float:
    """Return the model's predicted response probability for one sample."""
    subject, item = input

    predictor = _get_predictor()

    try:
        normalized_name = subject["normalized_name"]
        embedding = torch.as_tensor(
            utils.embed_text(utils.get_sbert(), [item["item_content"]])[0],
            dtype=torch.float32,
        )
    except KeyError as error:
        raise ValueError(f"Sample is missing required field {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        raise ValueError("item_content could not be encoded as a numeric vector") from error

    if not isinstance(normalized_name, str):
        raise ValueError("Sample normalized_name must be a string")
    if embedding.ndim != 1 or embedding.numel() != predictor.embedding_dimension:
        raise ValueError(
            "Encoded item_content must be a 1D vector of length "
            f"{predictor.embedding_dimension}"
        )
    if not torch.isfinite(embedding).all():
        raise ValueError("Encoded item_content contains non-finite values")

    embedding = embedding.unsqueeze(0).to(predictor.device)

    with torch.inference_mode():
        if normalized_name in predictor.student_indices:
            student_index = predictor.student_indices[normalized_name]
            student_id = torch.nn.functional.one_hot(
                torch.tensor([student_index]),
                num_classes=len(predictor.student_indices),
            ).to(dtype=torch.float32, device=predictor.device)
            probability = predictor.model(student_id, embedding)
        else:
            # The trained student network cannot represent a name it has not
            # seen. Use its saved population-average ability for such subjects.
            model = predictor.model
            params = model.item_param_head(
                model.item_embed_proj(embedding)
            )
            beta = params[:, 0:1]
            if model.model_type == "1pl":
                a = torch.ones_like(beta)
            else:
                a = F.softplus(params[:, 1:2]) + 0.1
            if model.model_type in ("1pl", "2pl"):
                c = torch.zeros_like(beta)
            else:
                c = torch.sigmoid(params[:, 2:3]) * 0.3
            if model.model_type == "4pl":
                d = torch.sigmoid(params[:, 3:4])
            else:
                d = torch.ones_like(beta)
            theta = torch.tensor(
                [[predictor.mean_ability]],
                dtype=torch.float32,
                device=predictor.device,
            )
            probability = c + (d - c) * torch.sigmoid(a * (theta - beta))
    return float(probability.item())
