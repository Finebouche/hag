
from sklearn.metrics import accuracy_score, mean_squared_error
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

# Select device: Apple GPU via MPS if available, else fallback
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def pad_collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, dim=0)
    return padded_seqs, labels, lengths


class BucketBatchSampler(Sampler):
    """
    Groups indices into buckets of size `bucket_size`, each bucket is
    sorted by sequence length, then yields batches of size `batch_size`
    from within each bucket.
    """
    def __init__(self, lengths, batch_size, bucket_size=None, shuffle=True):
        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size or batch_size * 20
        self.indices = np.arange(len(lengths))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.bucket_size):
            bucket_inds = self.indices[i: i + self.bucket_size]
            bucket_inds = sorted(bucket_inds, key=lambda idx: self.lengths[idx])
            for j in range(0, len(bucket_inds), self.batch_size):
                batch = bucket_inds[j: j + self.batch_size]
                if len(batch) == self.batch_size:
                    yield batch

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

class SequenceDataset(Dataset):
    def __init__(self, X_list, y_list):
        assert len(X_list) == len(y_list)
        self.X_list = X_list
        self.y_list = y_list
        # precompute lengths
        self.lengths = [len(x) for x in X_list]

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_list[idx], dtype=torch.float32)
        y = torch.tensor(self.y_list[idx], dtype=torch.float32)
        return x, y

class PrecomputedForecastDataset(Dataset):
    """
    Wraps pre-windowed inputs + targets.
      X_windows: array-like of shape (N, window, D)
      Y_targets: array-like of shape (N, horizon, D) or (N, D) if horizon=1
    """
    def __init__(self, X_windows, Y_targets):
        # convert to float-tensors
        self.X = torch.as_tensor(X_windows, dtype=torch.float32)
        self.y = torch.as_tensor(Y_targets, dtype=torch.float32)
        # if horizon=1 you might want to squeeze the time axis:
        if self.y.ndim == 3 and self.y.shape[1] == 1:
            # makes y.shape = (N, D)
            self.y = self.y.squeeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.0, bidirectional: bool = False):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.directions, output_size)

    def forward(self, x, lengths=None):
        # x: (B, T_max, D_in)
        out, _ = self.lstm(x)  # out: (B, T_max, H*dirs)
        B, T, _ = out.shape
        # grab the true last time step for each sequence
        if lengths is not None:
            idx = torch.arange(B, device=out.device)
            last = out[idx, lengths - 1]  # (B, H*dirs)
        else:
            last = out[:, -1, :]  # (B, H*dirs)

        last = self.dropout(last)  # apply dropout here
        return self.fc(last)


def make_sliding_windows(X, y, window):
    """
    X: np.ndarray, shape (T, D)
    y: np.ndarray, shape (T,) or (T, D_out)
    window: int, length of the input sequence
    horizon: int, how many steps ahead to predict (default=1)

    Returns:
      X_windows: np.ndarray, shape (N, window, D)
      y_targets: np.ndarray, shape (N, D_out) if horizon==1, else (N, horizon, D_out)
    """
    if X.ndim == 1:
        X = X[:, None]
    if y.ndim == 1:
        y = y[:, None]

    T = len(X)
    N = T - window + 1
    if N <= 0:
        raise ValueError(f"Not enough time steps: T={T}, window={window}")

    # build X_windows
    X_windows = np.stack([X[i: i + window]
                          for i in range(N)], axis=0)  # (N, window, D)

    y_targets = np.stack([y[i + window: i + window]  for i in range(N)], axis=0)  # (N, horizon, D_out)

    # if you’d rather have y_targets shape (N,) when D_out==1:
    if y_targets.shape[-1] == 1:
        y_targets = y_targets.squeeze(-1)

    return X_windows, y_targets


def train(model, loader, criterion, optimizer, task_type="classification"):
    model.train()
    total_loss = 0.0
    for batch in loader:
        # support both (x,y,lengths) and (x,y)
        if len(batch) == 3:
            X_batch, y_batch, lengths = batch
            lengths = lengths.to(DEVICE)
        else:
            X_batch, y_batch = batch
            lengths = None

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch, lengths)  # shape (B, C) or (B,1)
        if task_type == "classification":
            target = y_batch.argmax(dim=1)
        else:
            # regression: last hidden state predicts next value
            # y_batch shape (B, D) or (B, D_out)
            target = y_batch

        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, task_type="classification"):
    """
    Run evaluation over loader.
    Returns accuracy (classification) or MSE (regression).
    """
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                X_batch, y_batch, lengths = batch
                lengths = lengths.to(DEVICE)
            else:
                X_batch, y_batch = batch
                lengths = None

            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch, lengths).cpu().numpy()
            if task_type == "classification":
                y_true = y_batch.argmax(dim=1).cpu().numpy()
            else:
                y_true = y_batch.cpu().numpy()

            ys.append(y_true)
            ps.append(preds)
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    if task_type == "classification":
        y_pred_labels = y_pred.argmax(axis=1)
        return accuracy_score(y_true, y_pred_labels)
    else:
        return mean_squared_error(y_true, y_pred)

def main(args):
    data = np.load(args.data_path, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    input_size = X_train.shape[2]
    output_size = y_train.shape[1] if y_train.ndim > 1 else 1

    model = LSTMModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=output_size,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss() if args.task == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_metric = -np.inf if args.task == "classification" else np.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer)
        metric = evaluate(model, val_loader, task_type=args.task)
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} - val_{args.task}: {metric:.4f}")
        # Save best
        if (args.task == "classification" and metric > best_metric) or \
           (args.task == "regression" and metric < best_metric):
            best_metric = metric
            torch.save(model.state_dict(), args.save_path)

    print(f"Best validation {args.task}: {best_metric:.4f}")

if __name__ == '__main__':
    import torch

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")