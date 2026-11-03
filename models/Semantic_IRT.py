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


class ItemOnlyDataset(Dataset):
    def __init__(self, df, item_embeddings):
        self.num_students = len(df)
        self.num_items = len(df.columns)
        self.item_embeddings = torch.tensor(
            item_embeddings, dtype=torch.float32)

        self.data = []
        for sid in range(self.num_students):
            for qid in range(self.num_items):
                if pd.notnull(df.iloc[sid, qid]):
                    self.data.append((sid, qid, int(df.iloc[sid, qid])))

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


def evaluate(model, dataloader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch["student_id"].to(device),
                batch["item_embed"].to(device)
            )
            probs.extend(outputs.cpu().numpy())
            labels.extend(batch["label"].cpu().numpy())

    preds = [1 if p > 0.5 else 0 for p in probs]
    return {
        'acc': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'auc': roc_auc_score(labels, probs)
    }


def save_params(model, train_df, item_embeddings, model_type):
    model.eval()
    device = next(model.parameters()).device

    abilities = []
    with torch.no_grad():
        for sid in range(len(train_df)):
            student_id = F.one_hot(torch.tensor(
                sid), len(train_df)).float().to(device)
            abilities.append(model.student_net(student_id).item())

    pd.DataFrame({"student_id": range(len(train_df)), "ability": abilities}).to_csv(
        f"results/student_abilities_semantic_{model_type}_part2.csv", index=False)

    param_cols = ['item_id', 'difficulty',
                  'discriminability', 'guessing', 'feasibility']
    item_params = []

    with torch.no_grad():
        for qid in range(len(train_df.columns)):
            item_embed = torch.tensor(
                item_embeddings[qid], dtype=torch.float32).unsqueeze(0).to(device)
            params = model.item_param_head(model.item_embed_proj(item_embed))

            record = {'item_id': qid}
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
    parser.add_argument("--item_emb", type=str,
                        default='models/embeddings.csv')
    parser.add_argument("--train_csv", type=str, default="./data/part2.csv")
    parser.add_argument("--test_csv", type=str, default="./data/test.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model type: {args.model_type.upper()}")

    try:
        train_df = pd.read_csv(args.train_csv, header=None)
        test_df = pd.read_csv(args.test_csv, header=None)
        item_emb = torch.load(args.item_emb, weights_only=False)
        if isinstance(item_emb, list):
            item_emb = torch.tensor(item_emb, dtype=torch.float32)
        print(f"Data loaded successfully | Number of students: {len(train_df)} | Number of questions: {len(train_df.columns)}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    train_dataset = ItemOnlyDataset(train_df, item_emb)
    train_size = int(0.8 * len(train_dataset))
    train_set, val_set = random_split(
        train_dataset, [train_size, len(train_dataset)-train_size])
    test_dataset = ItemOnlyDataset(test_df, item_emb)

    train_loader = DataLoader(
        train_set, batch_size=512, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, pin_memory=True)

    model = Semantic_IRT(
        num_students=len(train_df),
        item_embed_dim=item_emb.shape[1],
        hidden_dim=64,
        model_type=args.model_type
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.BCELoss()
    best_f1 = 0
    early_stop = 0

    for epoch in range(100):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(
                batch["student_id"].to(device),
                batch["item_embed"].to(device)
            )
            loss = criterion(outputs.squeeze(), batch["label"].to(device))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics['f1'])
        print(
            f"Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

        if val_metrics['f1'] > best_f1 + 1e-4:
            best_f1 = val_metrics['f1']
            early_stop = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            early_stop += 1
            if early_stop >= 5:
                print(f"Early stopping triggered at {epoch+1} 轮")
                break

    model.load_state_dict(torch.load("best_model.pt", weights_only=False))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest set results:")
    print(
        f"ACC: {test_metrics['acc']:.4f} | F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f}")

    save_params(model, train_df, item_emb, args.model_type)
    print("Parameters have been saved to student_abilities.csv and item_parameters.csv")


if __name__ == "__main__":
    main()
