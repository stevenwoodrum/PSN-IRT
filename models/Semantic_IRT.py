# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import brier_score_loss
import numpy as np
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

if __package__:
    from .text_chunking import TEXT_ENCODING
else:
    from text_chunking import TEXT_ENCODING

RANDOM_SEED = 5814

# -------------------- Model Definition --------------------


class Semantic_IRT(nn.Module):
    """
    A variant of PSN-IRT that utilizes item embeddings; 
    the student side employs one-hot encoding, 
    while the item side relies exclusively on embeddings.
    """

    def __init__(self, num_students, item_embed_dim=768, hidden_dim=64, model_type='4pl'):
        super().__init__()
        self.model_type = model_type.lower()
        assert self.model_type in ['1pl', '2pl',
                                   '3pl', '4pl'], "The model type must be 1pl/2pl/3pl/4pl"

        # --------------------------
        # Student Ability Network (ID only)
        # --------------------------
        self.student_net = nn.Sequential(
            nn.Linear(num_students, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # --------------------------
        # Parameter Network (Embeddings only)
        # --------------------------
        self.item_embed_proj = nn.Sequential(
            nn.Linear(item_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Dynamic Parameter Head
        param_dims = {'1pl': 1, '2pl': 2, '3pl': 3, '4pl': 4}
        self.item_param_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, param_dims[self.model_type])
        )

    def forward(self, student_id, item_embed):
        theta = self.student_net(student_id)
        iem_feat = self.item_embed_proj(item_embed)
        params = self.item_param_head(iem_feat)

        beta = params[:, 0].unsqueeze(1)
        a = 1.0 if self.model_type == '1pl' else F.softplus(
            params[:, 1].unsqueeze(1)) + 0.1
        c = 0.0 if self.model_type in ['1pl', '2pl'] else torch.sigmoid(
            params[:, 2].unsqueeze(1)) * 0.3
        d = 1.0 if self.model_type != '4pl' else torch.sigmoid(
            params[:, 3]).unsqueeze(1)

        prob = c + (d - c) * torch.sigmoid(a * (theta - beta))
        return prob.clamp(1e-6, 1-1e-6)

# -------------------- Dataset Class --------------------


class ResponseSamplesDataset(Dataset):
    def __init__(self, samples, student_indices, item_indices, item_embeddings):
        self.num_students = len(student_indices)
        self.item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)
        self.data = [
            (
                student_indices[sample["normalized_name"]],
                item_indices[sample["item_id"]],
                int(sample["response"]),
            )
            for sample in samples
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sid, qid, label = self.data[idx]
        return {
            "student_id": F.one_hot(torch.tensor(sid), self.num_students).float(),
            "item_embed": self.item_embeddings[qid],
            "label": torch.tensor(label, dtype=torch.float32)
        }

# -------------------- Training Tools --------------------


def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    probs, labels = [], []
    total_loss = 0.0
    num_examples = 0
    if criterion is None:
        criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            student_ids = batch["student_id"].to(device)
            item_embeddings = batch["item_embed"].to(device)
            batch_labels = batch["label"].to(device)
            outputs = model(
                student_ids,
                item_embeddings,
            )
            total_loss += criterion(
                outputs.squeeze(-1), batch_labels
            ).item() * batch_labels.numel()
            num_examples += batch_labels.numel()
            probs.extend(outputs.squeeze(-1).cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())

    return {
        'loss': total_loss / num_examples,
        'brier': brier_score_loss(labels, probs),
    }


def plot_training_history(history, output_dir, test_brier=None):
    history_df = pd.DataFrame(history)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "semantic_irt_training_history.csv"
    plot_path = output_dir / "semantic_irt_training_history.png"
    history_df.to_csv(history_path, index=False)

    fig, loss_axis = plt.subplots(figsize=(10, 6))

    train_line, = loss_axis.plot(
        history_df["epoch"], history_df["training_loss"],
        label="Training Brier loss", color="tab:blue",
    )
    lines = [train_line]
    if "validation_loss" in history_df:
        val_loss_line, = loss_axis.plot(
            history_df["epoch"], history_df["validation_loss"],
            label="Validation Brier loss", color="tab:orange",
        )
        lines.append(val_loss_line)
    if test_brier is not None:
        test_brier_line = loss_axis.axhline(
            test_brier,
            label=f"Test Brier ({test_brier:.4f})",
            color="tab:green",
            linestyle="--",
        )
        lines.append(test_brier_line)

    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Brier loss")
    loss_axis.set_title("Semantic IRT Brier Training History")
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Training history saved to {history_path} and {plot_path}")


def save_params(
    model, student_names, item_ids, item_embeddings, model_type, output_dir
):
    model.eval()
    device = next(model.parameters()).device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    abilities = []
    with torch.no_grad():
        for sid in range(len(student_names)):
            student_id = F.one_hot(torch.tensor(
                sid), len(student_names)).float().to(device)
            abilities.append(model.student_net(student_id).item())

    pd.DataFrame({
        "student_id": range(len(student_names)),
        "normalized_name": student_names,
        "ability": abilities,
    }).to_csv(
        output_dir / f"student_abilities_semantic_{model_type}.csv", index=False)

    param_cols = ['item_id', 'difficulty',
                  'discriminability', 'guessing', 'feasibility']
    item_params = []

    with torch.no_grad():
        for qid, item_id in enumerate(item_ids):
            item_embed = torch.tensor(
                item_embeddings[qid], dtype=torch.float32).unsqueeze(0).to(device)
            params = model.item_param_head(model.item_embed_proj(item_embed))

            record = {'item_id': item_id}
            if model_type in ['1pl', '2pl', '3pl', '4pl']:
                record['difficulty'] = params[0, 0].item()
            if model_type in ['2pl', '3pl', '4pl']:
                record['discriminability'] = F.softplus(params[0, 1]).item()
            if model_type in ['3pl', '4pl']:
                record['guessing'] = torch.sigmoid(params[0, 2]).item()
            if model_type == '4pl':
                d_raw = torch.sigmoid(params[0, 3]).item()
                record['feasibility'] = record['guessing'] + \
                    (1 - record['guessing']) * d_raw

            item_params.append(record)

    df = pd.DataFrame(item_params)
    for col in param_cols:
        if col not in df.columns:
            df[col] = np.nan
    df[['item_id'] + param_cols[1:]].to_csv(
        output_dir / f"item_parameters_semantic_{model_type}.csv", index=False)

# -------------------- Main Program --------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='4pl',
                        choices=['1pl', '2pl', '3pl', '4pl'])
    parser.add_argument("--samples_json", type=str, default="samples_chunked.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model type: {args.model_type.upper()}")

    try:
        with open(args.samples_json, encoding="utf-8") as file:
            samples = json.load(file)
        if not samples:
            raise ValueError(f"No response samples found in {args.samples_json}")
        item_embeddings_by_id = {}
        for sample in samples:
            if sample.get("text_encoding") != TEXT_ENCODING:
                raise ValueError("Regenerate samples with the current chunked encoder before training.")
            required_fields = (
                "benchmark_id", "normalized_name", "item_id",
                "content_embedding", "response",
            )
            missing_fields = [
                field for field in required_fields if field not in sample
            ]
            if missing_fields:
                raise ValueError(
                    f"Sample is missing required fields: {missing_fields}"
                )
            if not isinstance(sample["normalized_name"], str):
                raise ValueError("Every sample must have a string normalized_name")

            embedding = np.asarray(sample["content_embedding"], dtype=np.float32)
            if embedding.ndim != 1 or embedding.size == 0:
                raise ValueError(
                    f"Item {sample['item_id']!r} has an invalid content_embedding"
                )
            if not np.isfinite(embedding).all():
                raise ValueError(
                    f"Item {sample['item_id']!r} has non-finite embedding values"
                )
            item_id = sample["item_id"]
            if item_id in item_embeddings_by_id:
                if not np.array_equal(item_embeddings_by_id[item_id], embedding):
                    raise ValueError(
                        f"Item {item_id!r} has inconsistent content_embedding vectors"
                    )
            else:
                item_embeddings_by_id[item_id] = embedding

            if sample["response"] not in (0, 1, 0.0, 1.0):
                raise ValueError(
                    f"Sample has non-binary response {sample['response']!r}"
                )
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    embedding_dimensions = {
        len(sample["content_embedding"]) for sample in samples
    }
    if len(embedding_dimensions) != 1:
        raise ValueError("All content_embedding vectors must have equal length")

    benchmark_ids = sorted({sample["benchmark_id"] for sample in samples})
    results_dir = Path("results") / "chunked" / "leave_one_benchmark_out"
    fold_results = []

    for fold_number, heldout_benchmark in enumerate(benchmark_ids):
        trainval_samples = [
            sample for sample in samples
            if sample["benchmark_id"] != heldout_benchmark
        ]
        all_test_samples = [
            sample for sample in samples
            if sample["benchmark_id"] == heldout_benchmark
        ]
        student_names = list(dict.fromkeys(
            sample["normalized_name"] for sample in trainval_samples
        ))
        known_students = set(student_names)
        test_samples = [
            sample for sample in all_test_samples
            if sample["normalized_name"] in known_students
        ]
        excluded_test_count = len(all_test_samples) - len(test_samples)
        if not trainval_samples:
            raise ValueError(
                f"No training responses available for {heldout_benchmark}"
            )
        if not test_samples:
            print(
                f"Skipping {heldout_benchmark}: no held-out responses have "
                "subjects present in the training benchmarks."
            )
            fold_results.append({
                "benchmark_id": heldout_benchmark,
                "test_responses": 0,
                "excluded_unseen_subject_responses": excluded_test_count,
                "test_brier": np.nan,
            })
            continue

        fold_samples = trainval_samples + test_samples
        item_ids = list(dict.fromkeys(
            sample["item_id"] for sample in fold_samples
        ))
        item_indices = {item_id: i for i, item_id in enumerate(item_ids)}
        student_indices = {
            name: i for i, name in enumerate(student_names)
        }
        item_embeddings = np.stack([
            item_embeddings_by_id[item_id] for item_id in item_ids
        ])

        dataset = ResponseSamplesDataset(
            fold_samples, student_indices, item_indices, item_embeddings
        )
        trainval_size = len(trainval_samples)
        train_size = int(0.8 * trainval_size)
        val_size = trainval_size - train_size
        if train_size == 0 or val_size == 0:
            raise ValueError(
                f"Not enough training-benchmark responses to split train/val "
                f"for {heldout_benchmark} ({trainval_size} responses)"
            )

        generator = torch.Generator().manual_seed(
            RANDOM_SEED + fold_number
        )
        trainval_set, test_set = random_split(
            dataset,
            [trainval_size, len(test_samples)],
            generator=generator,
        )
        train_set, val_set = random_split(
            trainval_set,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(
                RANDOM_SEED + fold_number
            ),
        )
        fold_dir = results_dir / heldout_benchmark
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size,
            pin_memory=device.type == "cuda",
        )

        model = Semantic_IRT(
            num_students=len(student_names),
            item_embed_dim=item_embeddings.shape[1],
            hidden_dim=64,
            model_type=args.model_type,
        ).to(device)
        optimizer = optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3, factor=0.5
        )
        criterion = nn.MSELoss()
        best_brier = float("inf")
        history = []

        print(
            f"\nHoldout: {heldout_benchmark} | "
            f"train/validation responses: {trainval_size} | "
            f"test responses: {len(test_samples)} | "
            f"excluded unseen-subject responses: {excluded_test_count}"
        )
        for epoch in range(args.epochs):
            model.train()
            train_loss_total = 0.0
            train_examples = 0

            for batch in tqdm(
                train_loader,
                desc=f"{heldout_benchmark} epoch {epoch + 1}",
            ):
                optimizer.zero_grad()
                outputs = model(
                    batch["student_id"].to(device),
                    batch["item_embed"].to(device),
                )
                loss = criterion(
                    outputs.squeeze(-1), batch["label"].to(device)
                )
                loss.backward()
                optimizer.step()
                batch_size = batch["label"].numel()
                train_loss_total += loss.item() * batch_size
                train_examples += batch_size

            training_loss = train_loss_total / train_examples
            val_metrics = evaluate(model, val_loader, device, criterion)
            history.append({
                "epoch": epoch + 1,
                "training_loss": training_loss,
                "validation_loss": val_metrics["loss"],
                "validation_brier": val_metrics["brier"],
            })
            scheduler.step(val_metrics["brier"])
            print(
                f"Train Loss: {training_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Brier: {val_metrics['brier']:.4f}"
            )

            if val_metrics["brier"] < best_brier:
                best_brier = val_metrics["brier"]
                torch.save(model.state_dict(), fold_dir / "best_model.pt")

        model.load_state_dict(torch.load(
            fold_dir / "best_model.pt",
            map_location=device,
            weights_only=True,
        ))
        test_metrics = evaluate(model, test_loader, device)
        plot_training_history(
            history, fold_dir, test_brier=test_metrics["brier"]
        )
        print(
            f"{heldout_benchmark} test Brier: "
            f"{test_metrics['brier']:.4f}"
        )
        save_params(
            model, student_names, item_ids, item_embeddings,
            args.model_type, fold_dir,
        )
        fold_results.append({
            "benchmark_id": heldout_benchmark,
            "test_responses": len(test_samples),
            "excluded_unseen_subject_responses": excluded_test_count,
            "test_brier": test_metrics["brier"],
        })

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.csv"
    pd.DataFrame(fold_results).to_csv(summary_path, index=False)
    print(f"\nLeave-one-benchmark-out results saved to {summary_path}")

    all_data_dir = Path("results") / "chunked" / "all_data"
    all_data_dir.mkdir(parents=True, exist_ok=True)
    all_student_names = list(dict.fromkeys(
        sample["normalized_name"] for sample in samples
    ))
    all_item_ids = list(dict.fromkeys(
        sample["item_id"] for sample in samples
    ))
    all_student_indices = {
        name: index for index, name in enumerate(all_student_names)
    }
    all_item_indices = {
        item_id: index for index, item_id in enumerate(all_item_ids)
    }
    all_item_embeddings = np.stack([
        item_embeddings_by_id[item_id] for item_id in all_item_ids
    ])
    all_data = ResponseSamplesDataset(
        samples,
        all_student_indices,
        all_item_indices,
        all_item_embeddings,
    )
    all_data_loader = DataLoader(
        all_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    all_data_model = Semantic_IRT(
        num_students=len(all_student_names),
        item_embed_dim=all_item_embeddings.shape[1],
        hidden_dim=64,
        model_type=args.model_type,
    ).to(device)
    all_data_optimizer = optim.AdamW(
        all_data_model.parameters(), lr=3e-4, weight_decay=1e-5
    )
    all_data_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        all_data_optimizer, "min", patience=3, factor=0.5
    )
    criterion = nn.MSELoss()
    all_data_history = []

    print(
        f"\nTraining final model on all {len(samples)} responses "
        f"({len(all_student_names)} subjects, {len(all_item_ids)} items)."
    )
    for epoch in range(args.epochs):
        all_data_model.train()
        train_loss_total = 0.0
        train_examples = 0

        for batch in tqdm(
            all_data_loader,
            desc=f"All-data epoch {epoch + 1}",
        ):
            all_data_optimizer.zero_grad()
            outputs = all_data_model(
                batch["student_id"].to(device),
                batch["item_embed"].to(device),
            )
            loss = criterion(
                outputs.squeeze(-1), batch["label"].to(device)
            )
            loss.backward()
            all_data_optimizer.step()
            batch_size = batch["label"].numel()
            train_loss_total += loss.item() * batch_size
            train_examples += batch_size

        training_loss = train_loss_total / train_examples
        training_metrics = evaluate(
            all_data_model, all_data_loader, device, criterion
        )
        all_data_history.append({
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "training_brier": training_metrics["brier"],
        })
        all_data_scheduler.step(training_metrics["brier"])
        print(
            f"All-data train loss: {training_loss:.4f} | "
            f"All-data Brier: {training_metrics['brier']:.4f}"
        )

    all_data_model_path = all_data_dir / "semantic_irt_model.pt"
    torch.save(all_data_model.state_dict(), all_data_model_path)
    plot_training_history(all_data_history, all_data_dir)
    save_params(
        all_data_model,
        all_student_names,
        all_item_ids,
        all_item_embeddings,
        args.model_type,
        all_data_dir,
    )
    all_data_summary = {
        "text_encoding": TEXT_ENCODING,
        "model_type": args.model_type,
        "epochs": args.epochs,
        "responses": len(samples),
        "subjects": len(all_student_names),
        "items": len(all_item_ids),
        "embedding_dimension": int(all_item_embeddings.shape[1]),
        "training_loss": training_metrics["loss"],
        "training_brier": training_metrics["brier"],
        "model_path": str(all_data_model_path),
    }
    all_data_summary_path = all_data_dir / "training_summary.json"
    with all_data_summary_path.open("w", encoding="utf-8") as file:
        json.dump(all_data_summary, file, indent=2)
    print(f"All-data model and statistics saved under {all_data_dir}")


if __name__ == "__main__":
    main()
