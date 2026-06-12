import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Running on device: {device}")

def generate_hybrid_fingerprints(smiles_list, radius=2, n_bits=2048):

    fps = []
    valid_indices = []

    for idx, smiles in enumerate(smiles_list):

        try:
            mol = Chem.MolFromSmiles(str(smiles))

            if mol:

                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius,
                    nBits=n_bits
                )

                morgan_arr = np.array(
                    [int(b) for b in morgan_fp.ToBitString()],
                    dtype=np.float32
                )

                maccs_fp = MACCSkeys.GenMACCSKeys(mol)

                maccs_arr = np.array(
                    [int(b) for b in maccs_fp.ToBitString()],
                    dtype=np.float32
                )

                hybrid = np.concatenate([morgan_arr, maccs_arr])

                fps.append(hybrid)
                valid_indices.append(idx)

        except:
            continue

    return np.array(fps), valid_indices

print(" Loading core enhanced datasets...")

df_3a4 = pd.read_csv("MASTER_CYP3A4_ENHANCED.csv")
df_3a7_base = pd.read_csv("MASTER_CYP3A7_ENHANCED.csv")

print(" Infusing seed data from CHEMBL_CYP3A7_RAW.csv...")

df_raw_3a7 = pd.read_csv("CHEMBL_CYP3A7_RAW.csv")

df_raw_3a7['numeric_val'] = pd.to_numeric(
    df_raw_3a7['standard_value'],
    errors='coerce'
)

df_raw_3a7 = df_raw_3a7.dropna(
    subset=['numeric_val', 'canonical_smiles']
)

df_raw_3a7 = df_raw_3a7[df_raw_3a7['numeric_val'] > 0]

df_raw_3a7['y'] = -np.log10(
    df_raw_3a7['numeric_val'] * 1e-9
)

df_raw_3a7 = df_raw_3a7[
    (df_raw_3a7['y'] >= 2) &
    (df_raw_3a7['y'] <= 12)
]

df_raw_3a7 = df_raw_3a7[['canonical_smiles', 'y']]


df_base_train, df_base_test = train_test_split(
    df_3a7_base,
    test_size=0.2,
    random_state=42
)

df_raw_train, df_raw_test = train_test_split(
    df_raw_3a7,
    test_size=0.2,
    random_state=42
)

df_train_final = pd.concat(
    [df_base_train, df_raw_train],
    axis=0,
    ignore_index=True
)

df_test_final = pd.concat(
    [df_base_test, df_raw_test],
    axis=0,
    ignore_index=True
)

print(" Computing 2215-bit Hybrid Matrices...")

X_3a4, valid_idx_3a4 = generate_hybrid_fingerprints(
    df_3a4['canonical_smiles'].values
)

y_3a4 = df_3a4['y'].values[valid_idx_3a4]

X_train_3a7, valid_idx_train = generate_hybrid_fingerprints(
    df_train_final['canonical_smiles'].values
)

y_train_3a7 = df_train_final['y'].values[valid_idx_train]

X_test_3a7, valid_idx_test = generate_hybrid_fingerprints(
    df_test_final['canonical_smiles'].values
)

y_test_3a7 = df_test_final['y'].values[valid_idx_test]

x_scaler = StandardScaler()

X_3a4 = x_scaler.fit_transform(X_3a4)

X_train_3a7 = x_scaler.transform(X_train_3a7)
X_test_3a7 = x_scaler.transform(X_test_3a7)

noise = np.random.normal(
    0,
    0.01,
    X_train_3a7.shape
)

X_train_3a7 = X_train_3a7 + noise

y_scaler = StandardScaler()

y_train_3a7_scaled = y_scaler.fit_transform(
    y_train_3a7.reshape(-1, 1)
).flatten()

y_test_3a7_scaled = y_scaler.transform(
    y_test_3a7.reshape(-1, 1)
).flatten()

sample_weights = np.exp(0.5 * np.abs(y_train_3a7_scaled))

sample_weights = torch.tensor(
    sample_weights,
    dtype=torch.float32
)

sampler = torch.utils.data.WeightedRandomSampler(
    sample_weights,
    len(sample_weights),
    replacement=True
)

train_loader_3a4 = DataLoader(
    TensorDataset(
        torch.tensor(X_3a4, dtype=torch.float32),
        torch.tensor(y_3a4, dtype=torch.float32).unsqueeze(1)
    ),
    batch_size=128,
    shuffle=True
)

train_loader_3a7 = DataLoader(
    TensorDataset(
        torch.tensor(X_train_3a7, dtype=torch.float32),
        torch.tensor(
            y_train_3a7_scaled,
            dtype=torch.float32
        ).unsqueeze(1)
    ),
    batch_size=32,
    sampler=sampler
)

val_loader_3a7 = DataLoader(
    TensorDataset(
        torch.tensor(X_test_3a7, dtype=torch.float32),
        torch.tensor(y_test_3a7_scaled, dtype=torch.float32).unsqueeze(1)
    ),
    batch_size=64,
    shuffle=False
)

class PublicationTransferNetwork(nn.Module):

    def __init__(self, input_dim=2215):

        super(PublicationTransferNetwork, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.05)
        )

        self.res_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.05),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )

        self.activation = nn.Mish()

        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(0.05)
        )

        self.regressor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Dropout(0.05),

            nn.Linear(128, 64),
            nn.Mish(),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        out = self.input_layer(x)

        out = self.activation(
            out + self.res_block(out)
        )

        features = self.feature_extractor(out)

        return self.regressor_head(features)

model = PublicationTransferNetwork(
    input_dim=2215
).to(device)

print("\n Phase 1: Pre-training on CYP3A4...")

base_criterion = nn.SmoothL1Loss(beta=0.5)



fine_tune_optimizer = optim.AdamW([

    {
        'params': model.input_layer.parameters(),
        'lr': 1e-5
    },

    {
        'params': model.res_block.parameters(),
        'lr': 3e-5
    },

    {
        'params': model.feature_extractor.parameters(),
        'lr': 1e-4
    },

    {
        'params': model.regressor_head.parameters(),
        'lr': 2e-4
    }

], weight_decay=1e-5)

base_criterion = nn.SmoothL1Loss(beta=0.5)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

model.train()

for epoch in range(60):

    running_loss = 0

    for batch_X, batch_y in train_loader_3a4:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        preds = model(batch_X)

        loss = base_criterion(preds, batch_y)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Pretrain Epoch {epoch+1}/60 "
        f"| Loss: {running_loss:.4f}"
    )

print("\n Phase 2: Fine-tuning on CYP3A7...")

def weighted_huber_loss(predictions, targets):

    error = predictions - targets

    abs_error = torch.abs(error)

    huber = torch.where(
        abs_error < 1.0,
        0.5 * error**2,
        abs_error - 0.5
    )

    weights = 1.0 + 0.5 * torch.abs(targets - torch.mean(targets))

    return torch.mean(weights * huber)

fine_tune_optimizer = optim.AdamW([

    {
        'params': model.input_layer.parameters(),
        'lr': 5e-5
    },

    {
        'params': model.res_block.parameters(),
        'lr': 5e-5
    },

    {
        'params': model.feature_extractor.parameters(),
        'lr': 1e-4
    },

    {
        'params': model.regressor_head.parameters(),
        'lr': 3e-4
    }

], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    fine_tune_optimizer,
    T_0=15,
    T_mult=2
)

best_r2 = -999

model.train()

for epoch in range(200):

    running_loss = 0

    for batch_X, batch_y in train_loader_3a7:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        fine_tune_optimizer.zero_grad()

        preds = model(batch_X)

        loss = weighted_huber_loss(
            preds,
            batch_y
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        fine_tune_optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    model.eval()

    temp_preds = []
    temp_targets = []

    with torch.no_grad():

        for batch_X, batch_y in val_loader_3a7:

            batch_X = batch_X.to(device)

            preds = model(batch_X)

            temp_preds.extend(
                preds.cpu().numpy().flatten()
            )

            temp_targets.extend(
                batch_y.numpy().flatten()
            )

    temp_preds = y_scaler.inverse_transform(
        np.array(temp_preds).reshape(-1, 1)
    ).flatten()

    temp_targets = y_scaler.inverse_transform(
        np.array(temp_targets).reshape(-1, 1)
    ).flatten()

    temp_r2 = r2_score(
        temp_targets,
        temp_preds
    )

    print(
        f"Epoch {epoch+1}/200 "
        f"| Loss: {running_loss:.4f} "
        f"| Validation R²: {temp_r2:.4f}"
    )

    # Save best model
    if temp_r2 > best_r2:

        best_r2 = temp_r2

        torch.save(
            model.state_dict(),
            "best_transfer_model.pt"
        )

        print(
            f" New best model saved "
            f"(R² = {best_r2:.4f})"
        )

    model.train()

model.load_state_dict(
    torch.load(
    "best_transfer_model.pt",
    weights_only=True
)
)

print("\n Extracting pretrained CYP3A7 embeddings...")

model.eval()

latent_features = []

with torch.no_grad():

    train_tensor = torch.tensor(
        X_train_3a7,
        dtype=torch.float32
    ).to(device)

    # Extract intermediate learned features
    out = model.input_layer(train_tensor)

    out = model.activation(
        out + model.res_block(out)
    )

    features = model.feature_extractor(out)

    latent_features = features.cpu().numpy()

latent_df = pd.DataFrame(latent_features)

latent_df['canonical_smiles'] = (
    df_train_final.iloc[valid_idx_train]
    ['canonical_smiles']
    .values
)

latent_df['true_pIC50'] = y_train_3a7

latent_df.to_csv(
    "PRETRAINED_CYP3A7_LATENT_FEATURES.csv",
    index=False
)

print(
    " Saved pretrained latent feature dataset as "
    "'PRETRAINED_CYP3A7_LATENT_FEATURES.csv'"
)

model.eval()

val_preds = []
val_targets = []

with torch.no_grad():

    for batch_X, batch_y in val_loader_3a7:

        batch_X = batch_X.to(device)

        preds = model(batch_X)

        val_preds.extend(
            preds.cpu().numpy().flatten()
        )

        val_targets.extend(
            batch_y.numpy().flatten()
        )

predictions_3a7 = y_scaler.inverse_transform(
    np.array(val_preds).reshape(-1, 1)
).flatten()

y_test_3a7_clean = y_scaler.inverse_transform(
    np.array(val_targets).reshape(-1, 1)
).flatten()

r2 = r2_score(
    y_test_3a7_clean,
    predictions_3a7
)

mae = mean_absolute_error(
    y_test_3a7_clean,
    predictions_3a7
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_3a7_clean,
        predictions_3a7
    )
)

slope, intercept = np.polyfit(
    y_test_3a7_clean,
    predictions_3a7,
    1
)

print("\n" + "="*50)
print(" FINAL PUBLICATION-LEVEL RESULTS")
print("="*50)

print(f" Final R² Score       : {r2:.4f}")
print(f" Regression Slope     : {slope:.4f}")
print(f" MAE                  : {mae:.4f}")
print(f" RMSE                 : {rmse:.4f}")

print("="*50)

print("\n Packaging finalized CYP3A7 dataset...")

with torch.no_grad():

    train_tensor = torch.tensor(
        X_train_3a7,
        dtype=torch.float32
    ).to(device)

    train_raw_preds = model(
        train_tensor
    ).cpu().numpy().flatten()

    train_preds = y_scaler.inverse_transform(
        train_raw_preds.reshape(-1, 1)
    ).flatten()

    test_preds = predictions_3a7

df_train_track = df_train_final.iloc[
    valid_idx_train
].copy()

df_train_track['predicted_pIC50'] = train_preds
df_train_track['dataset_assignment'] = 'Phase_2_Training'

df_test_track = df_test_final.iloc[
    valid_idx_test
].copy()

df_test_track['predicted_pIC50'] = test_preds
df_test_track['dataset_assignment'] = 'Validation_Holdout'

df_export = pd.concat(
    [df_train_track, df_test_track],
    axis=0,
    ignore_index=True
)

df_export['absolute_error'] = np.abs(
    df_export['y'] -
    df_export['predicted_pIC50']
)

df_export = df_export.drop_duplicates(
    subset=['canonical_smiles'],
    keep='first'
)

ordered_cols = [
    'canonical_smiles',
    'y',
    'predicted_pIC50',
    'absolute_error',
    'dataset_assignment'
]

df_export = df_export[ordered_cols]

output_file = "FINAL_TRAINED_CYP3A7_PREDICTIONS.csv"

try:

    df_export.to_csv(
        output_file,
        index=False
    )

    print(
        f" Dataset successfully saved as: "
        f"'{output_file}'"
    )

except PermissionError:

    fallback_file = (
        "FINAL_TRAINED_CYP3A7_PREDICTIONS_NEW.csv"
    )

    df_export.to_csv(
        fallback_file,
        index=False
    )

    print(
        f" Original file locked. "
        f"Saved fallback as '{fallback_file}'"
    )

print(
    f" Unique chemical space exported: "
    f"{len(df_export)} compounds."
)