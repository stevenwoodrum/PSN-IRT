# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

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
        criterion = nn.BCELoss()

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

    preds = [1 if p > 0.5 else 0 for p in probs]
    return {
        'loss': total_loss / num_examples,
        'acc': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'auc': roc_auc_score(labels, probs)
    }


def plot_training_history(history, output_dir):
    history_df = pd.DataFrame(history)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "semantic_irt_training_history.csv"
    plot_path = output_dir / "semantic_irt_training_history.png"
    history_df.to_csv(history_path, index=False)

    fig, loss_axis = plt.subplots(figsize=(10, 6))
    auc_axis = loss_axis.twinx()

    train_line, = loss_axis.plot(
        history_df["epoch"], history_df["training_loss"],
        label="Training loss", color="tab:blue",
    )
    val_loss_line, = loss_axis.plot(
        history_df["epoch"], history_df["validation_loss"],
        label="Validation loss", color="tab:orange",
    )
    auc_line, = auc_axis.plot(
        history_df["epoch"], history_df["validation_auc"],
        label="Validation AUC", color="tab:green",
    )

    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    auc_axis.set_ylabel("Validation AUC")
    loss_axis.set_title("Semantic IRT Training History")
    loss_axis.legend(
        [train_line, val_loss_line, auc_line],
        [line.get_label() for line in (train_line, val_loss_line, auc_line)],
        loc="best",
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Training history saved to {history_path} and {plot_path}")


def save_params(model, student_names, item_ids, item_embeddings, model_type):
    model.eval()
    device = next(model.parameters()).device

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
        f"results/student_abilities_semantic_{model_type}_part2.csv", index=False)

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
    df[param_cols].to_csv(
        f"results/item_parameters_semantic_{model_type}_part2.csv", index=False)

# -------------------- Main Program --------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='4pl',
                        choices=['1pl', '2pl', '3pl', '4pl'])
    parser.add_argument("--samples_json", type=str, default="samples.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model type: {args.model_type.upper()}")

    try:
        with open(args.samples_json, encoding="utf-8") as file:
            samples = json.load(file)
        if not samples:
            raise ValueError(f"No response samples found in {args.samples_json}")

        student_names = list(dict.fromkeys(
            sample["normalized_name"] for sample in samples
        ))
        item_ids = list(dict.fromkeys(sample["item_id"] for sample in samples))
        student_indices = {
            name: index for index, name in enumerate(student_names)
        }
        item_indices = {item_id: index for index, item_id in enumerate(item_ids)}

        item_embeddings_by_id = {}
        for sample in samples:
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

        item_embeddings = np.stack(
            [item_embeddings_by_id[item_id] for item_id in item_ids]
        )
        dataset = ResponseSamplesDataset(
            samples, student_indices, item_indices, item_embeddings
        )
        train_test_size = int(0.8 * len(dataset))
        train_data, test_set = random_split(
            dataset,
            [train_test_size, len(dataset) - train_test_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )
        train_size = int(0.8 * len(train_data))
        train_set, val_set = random_split(
            train_data,
            [train_size, len(train_data) - train_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )
        print(
            f"Loaded {len(samples)} responses | {len(student_names)} subjects | "
            f"{len(item_ids)} items | embedding dimension {item_embeddings.shape[1]}"
        )
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    train_loader = DataLoader(
        train_set, batch_size=512, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=512, pin_memory=True)

    model = Semantic_IRT(
        num_students=len(student_names),
        item_embed_dim=item_embeddings.shape[1],
        hidden_dim=64,
        model_type=args.model_type
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.BCELoss()
    best_f1 = -1
    early_stop = 0
    history = []

    for epoch in range(100):
        model.train()
        train_loss_total = 0.0
        train_examples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(
                batch["student_id"].to(device),
                batch["item_embed"].to(device)
            )
            loss = criterion(outputs.squeeze(-1), batch["label"].to(device))

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
            "validation_auc": val_metrics["auc"],
        })
        scheduler.step(val_metrics['f1'])
        print(
            f"Train Loss: {training_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics['f1'] > best_f1 + 1e-4:
            best_f1 = val_metrics['f1']
            early_stop = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            early_stop += 1
            #if early_stop >= 5:
            #    print(f"Early stopping triggered at {epoch+1} 轮")
            #    break

    plot_training_history(history, "results")
    model.load_state_dict(torch.load("best_model.pt", weights_only=False))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest set results:")
    print(
        f"ACC: {test_metrics['acc']:.4f} | F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f}")

    save_params(
        model, student_names, item_ids, item_embeddings, args.model_type
    )
    print("Parameters have been saved to student_abilities.csv and item_parameters.csv")


if __name__ == "__main__":
    main()
