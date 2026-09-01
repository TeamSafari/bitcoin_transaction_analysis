from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("./data")

OUTPUT_DIR = Path("./outputs")

ARTIFACT_DIR = Path("./artifacts")

WALLET_FEATURES_FILE = (
    DATA_DIR / "wallet_features.csv"
)

GRAPH_FEATURES_FILE = (
    DATA_DIR / "graph_features.csv"
)

GRAPHSAGE_FILE = (
    DATA_DIR / "graphsage_embeddings.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "autoencoder_scores.csv"
)

MODEL_FILE = (
    ARTIFACT_DIR / "autoencoder.pt"
)

SCALER_FILE = (
    ARTIFACT_DIR / "autoencoder_scaler.pkl"
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

RANDOM_STATE = 42

EPOCHS = 50

BATCH_SIZE = 256

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

MAX_LATENT_DIM = 32


np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(
    path: Path,
    name: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"{name} is empty."
        )

    if "wallet_id" not in df.columns:
        raise ValueError(
            f"{name} must contain "
            "'wallet_id'."
        )

    df["wallet_id"] = (
        df["wallet_id"]
        .astype(str)
    )

    if df["wallet_id"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate "
            "wallet_id values."
        )

    return df


# ============================================================
# KEEP NUMERIC FEATURES
# ============================================================

def numeric_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = pd.DataFrame({
        "wallet_id": df["wallet_id"]
    })

    for column in df.columns:

        if column == "wallet_id":
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric.notna().sum() > 0:
            result[column] = numeric

    return result


# ============================================================
# FUSE THREE DATASETS
# ============================================================

def build_fused_features():

    print("=" * 60)
    print("LOADING INPUT DATA")
    print("=" * 60)

    wallet_df = load_csv(
        WALLET_FEATURES_FILE,
        "wallet_features.csv",
    )

    graph_df = load_csv(
        GRAPH_FEATURES_FILE,
        "graph_features.csv",
    )

    sage_df = load_csv(
        GRAPHSAGE_FILE,
        "graphsage_embeddings.csv",
    )

    print(
        f"wallet_features: "
        f"{len(wallet_df)} rows"
    )

    print(
        f"graph_features: "
        f"{len(graph_df)} rows"
    )

    print(
        f"graphsage_embeddings: "
        f"{len(sage_df)} rows"
    )

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    wallet_df = numeric_features(
        wallet_df
    )

    graph_df = numeric_features(
        graph_df
    )

    sage_df = numeric_features(
        sage_df
    )

    # --------------------------------------------------------
    # Verify common wallets
    # --------------------------------------------------------

    common_wallets = (
        set(wallet_df["wallet_id"])
        &
        set(graph_df["wallet_id"])
        &
        set(sage_df["wallet_id"])
    )

    if not common_wallets:
        raise ValueError(
            "No common wallet_id values found "
            "across the three input files."
        )

    print(
        f"Common wallets: "
        f"{len(common_wallets)}"
    )

    # --------------------------------------------------------
    # Fuse
    # --------------------------------------------------------

    fused = wallet_df.merge(
        graph_df,
        on="wallet_id",
        how="inner",
        suffixes=(
            "",
            "_graph",
        ),
    )

    fused = fused.merge(
        sage_df,
        on="wallet_id",
        how="inner",
        suffixes=(
            "",
            "_graphsage",
        ),
    )

    if fused.empty:
        raise ValueError(
            "Feature fusion produced zero rows."
        )

    # --------------------------------------------------------
    # Remove duplicate column names
    # --------------------------------------------------------

    fused = fused.loc[
        :,
        ~fused.columns.duplicated()
    ]

    # --------------------------------------------------------
    # Identify features
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in fused.columns
        if column != "wallet_id"
    ]

    # --------------------------------------------------------
    # Clean infinities
    # --------------------------------------------------------

    fused[feature_columns] = (
        fused[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    for column in feature_columns:

        median = (
            fused[column]
            .median()
        )

        if pd.isna(median):
            median = 0.0

        fused[column] = (
            fused[column]
            .fillna(median)
        )

    # --------------------------------------------------------
    # Remove constant features
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in feature_columns
        if fused[column].nunique() > 1
    ]

    if not feature_columns:
        raise ValueError(
            "No non-constant numeric features "
            "remain after preprocessing."
        )

    fused = fused[
        ["wallet_id"]
        + feature_columns
    ]

    print(
        f"Fused rows: "
        f"{len(fused)}"
    )

    print(
        f"Fused features: "
        f"{len(feature_columns)}"
    )

    return fused, feature_columns


# ============================================================
# PYTORCH AUTOENCODER
# ============================================================

class Autoencoder(
    nn.Module
):

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
    ):

        super().__init__()

        hidden_dim = max(
            8,
            min(
                128,
                input_dim * 2,
            ),
        )

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                latent_dim,
            ),
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                input_dim,
            ),
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = (
            self.decoder(latent)
        )

        return reconstructed


# ============================================================
# TRAIN AUTOENCODER
# ============================================================

def train_autoencoder(
    X_scaled: np.ndarray,
    input_dim: int,
):

    # --------------------------------------------------------
    # Determine latent dimension
    # --------------------------------------------------------

    latent_dim = max(
        2,
        min(
            MAX_LATENT_DIM,
            input_dim // 2,
        ),
    )

    if latent_dim >= input_dim:
        latent_dim = max(
            1,
            input_dim - 1,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Input dimension: {input_dim}"
    )

    print(
        f"Latent dimension: {latent_dim}"
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        total_loss = 0.0
        total_samples = 0

        for batch_tuple in loader:

            batch = (
                batch_tuple[0]
                .to(device)
            )

            optimizer.zero_grad()

            reconstructed = (
                model(batch)
            )

            loss = criterion(
                reconstructed,
                batch,
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                batch.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

        epoch_loss = (
            total_loss
            / max(
                total_samples,
                1,
            )
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:03d}/{EPOCHS} "
                f"- loss: "
                f"{epoch_loss:.6f}"
            )

    # --------------------------------------------------------
    # Calculate reconstruction error
    # --------------------------------------------------------

    model.eval()

    errors = []

    inference_loader = DataLoader(
        TensorDataset(tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    with torch.no_grad():
        for batch_tuple in inference_loader:
            batch = batch_tuple[0].to(device)

            reconstructed = model(batch)

            reconstruction_error = (
                (reconstructed - batch) ** 2
            ).mean(dim=1)

            errors.extend(
                reconstruction_error.cpu().numpy().tolist()
            )

    return (
        model,
        np.asarray(
            errors,
            dtype=float,
        ),
        latent_dim,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    fused: pd.DataFrame,
    feature_columns: list[str],
    model: Autoencoder,
    scaler: StandardScaler,
    errors: np.ndarray,
    input_dim: int,
    latent_dim: int,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Autoencoder scores
    # --------------------------------------------------------

    output = pd.DataFrame({

        "wallet_id":
            fused["wallet_id"],

        "autoencoder_reconstruction_error":
            errors,
    })

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 2. Save PyTorch model
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim":
                input_dim,

            "latent_dim":
                latent_dim,

            "feature_columns":
                feature_columns,

            "architecture":
                "MLP_Autoencoder",
        },
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # 3. IMPORTANT:
    # Save the EXACT fitted scaler
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_FILE,
    )

    print("\nSaved artifacts:")

    print(
        f"  Scores: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"  Model:  "
        f"{MODEL_FILE}"
    )

    print(
        f"  Scaler: "
        f"{SCALER_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 60)
    print("# AUTOENCODER PIPELINE")
    print("#" * 60)

    # --------------------------------------------------------
    # STEP 1
    # Fuse the three required CSVs
    # --------------------------------------------------------

    fused, feature_columns = (
        build_fused_features()
    )

    # --------------------------------------------------------
    # STEP 2
    # Prepare matrix
    # --------------------------------------------------------

    X = (
        fused[feature_columns]
        .values
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # STEP 3
    # FIT SCALER
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = (
        scaler
        .fit_transform(X)
        .astype(np.float32)
    )

    print(
        "\nStandardScaler fitted."
    )

    # --------------------------------------------------------
    # STEP 4
    # Train Autoencoder
    # --------------------------------------------------------

    (
        model,
        errors,
        latent_dim,
    ) = train_autoencoder(
        X_scaled=X_scaled,
        input_dim=X_scaled.shape[1],
    )

    # --------------------------------------------------------
    # STEP 5
    # Save everything
    # --------------------------------------------------------

    save_results(
        fused=fused,
        feature_columns=feature_columns,
        model=model,
        scaler=scaler,
        errors=errors,
        input_dim=X_scaled.shape[1],
        latent_dim=latent_dim,
    )

    print("\n" + "=" * 60)
    print("AUTOENCODER PIPELINE COMPLETE")
    print("=" * 60)

    print(
        f"Wallets processed: "
        f"{len(fused)}"
    )

    print(
        f"Features used: "
        f"{len(feature_columns)}"
    )

    print(
        f"Mean reconstruction error: "
        f"{errors.mean():.6f}"
    )

    print(
        f"Max reconstruction error: "
        f"{errors.max():.6f}"
    )


if __name__ == "__main__":
    main()