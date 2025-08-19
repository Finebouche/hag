
from sklearn.metrics import accuracy_score, mean_squared_error
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from performances.losses import nrmse_multivariate

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
        # ALWAYS keep y 2D: (N, D_out).  For D_out=1, y will be (N,1).
        if self.y.ndim == 3 and self.y.shape[1] == 1:
            # instead of removing the time axis, just reshape to (N,1)
            self.y = self.y.reshape(self.y.shape[0], 1)
        elif self.y.ndim == 1:
            # if someone passed in a flat (N,), make it (N,1)
            self.y = self.y[:, None]

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

class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.0, bidirectional: bool = False):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.directions, output_size)

    def forward(self, x, lengths=None):
        # x: (B, T_max, D_in)
        out, _ = self.gru(x)  # out: (B, T_max, H*dirs)
        B, T, _ = out.shape
        # grab the true last time step for each sequence
        if lengths is not None:
            idx = torch.arange(B, device=out.device)
            last = out[idx, lengths - 1]  # (B, H*dirs)
        else:
            last = out[:, -1, :]  # (B, H*dirs)

        last = self.dropout(last)
        return self.fc(last)

class RNNModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 *,
                 custom_ih: torch.Tensor = None,
                 custom_hh: torch.Tensor = None,
                 custom_bias: torch.Tensor = None):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          nonlinearity="tanh",
                          batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if num_layers>1 else 0.0)
        self.directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.directions, output_size)

        # If a custom weight matrices we copy them into the RNN.
        #   custom_ih  → (num_layers * directions, hidden_size, input_size)
        #   custom_hh  → (num_layers * directions, hidden_size, hidden_size)
        #   custom_bias→ (num_layers * directions,  hidden_size)

        if custom_ih is not None:
            assert custom_ih.shape == self.rnn.weight_ih_l0.shape, \
                f"weight_ih_l0 shape mismatch: got {custom_ih.shape}, expected {self.rnn.weight_ih_l0.shape}"
            self.rnn.weight_ih_l0.data.copy_(custom_ih)

        if custom_hh is not None:
            assert custom_hh.shape == self.rnn.weight_hh_l0.shape, \
                f"weight_hh_l0 shape mismatch: got {custom_hh.shape}, expected {self.rnn.weight_hh_l0.shape}"
            self.rnn.weight_hh_l0.data.copy_(custom_hh)

        if custom_bias is not None:
            # RNN has two bias vectors per layer: bias_ih_l0 and bias_hh_l0
            assert custom_bias.shape == self.rnn.bias_ih_l0.shape, \
                f"bias shape mismatch: got {custom_bias.shape}, expected {self.rnn.bias_ih_l0.shape}"
            self.rnn.bias_ih_l0.data.copy_(custom_bias)
            self.rnn.bias_hh_l0.data.copy_(custom_bias)

    def forward(self, x, lengths=None):
        out, _ = self.rnn(x)
        B, T_max, _ = out.shape

        if lengths is not None:
            idx = torch.arange(B, device=out.device)
            last_hidden = out[idx, lengths - 1]
        else:
            last_hidden = out[:, -1, :]

        return self.fc(last_hidden)


def make_sliding_windows(X, y, window):
    """
    X: np.ndarray, shape (T, D)
    y: np.ndarray, shape (T, D_out)  (already 5-step-ahead targets)
    window: int, length of each input sequence

    Returns:
      X_windows: np.ndarray, shape (N, window, D)
      y_targets: np.ndarray, shape (N, D_out)
    """
    # make sure both are 2D
    if X.ndim == 1:
        X = X[:, None]
    if y.ndim == 1:
        y = y[:, None]

    T, D  = X.shape
    _, D_out = y.shape
    N = T - window + 1
    if N <= 0:
        raise ValueError(f"Not enough time steps: T={T}, window={window}")

    # 1) your sliding windows from X
    X_windows = np.stack(
        [X[i : i + window] for i in range(N)],
        axis=0,   # → (N, window, D)
    )

    # 2) just take the first N targets from y
    y_targets = y[:N]        # → (N, D_out)

    # if you really want a 1D target when D_out==1:
    if D_out == 1:
        y_targets = y_targets.squeeze(-1)  # → (N,)

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
        return nrmse_multivariate(y_true, y_pred)

if __name__ == '__main__':
    import torch

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")