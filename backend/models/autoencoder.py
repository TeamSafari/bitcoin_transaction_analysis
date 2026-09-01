from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "features"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
)

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "artifacts"
)


WALLET_FEATURES_FILE = (
    FEATURE_DIR
    / "wallet_features.csv"
)

GRAPH_FEATURES_FILE = (
    GRAPH_DIR
    / "graph_features.csv"
)

GRAPHSAGE_FILE = (
    GRAPH_DIR
    / "graphsage_embeddings.csv"
)

OUTPUT_FILE = (
    MODEL_OUTPUT_DIR
    / "autoencoder_scores.csv"
)

MODEL_FILE = (
    ARTIFACT_DIR
    / "autoencoder.pt"
)

SCALER_FILE = (
    ARTIFACT_DIR
    / "autoencoder_scaler.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

HIDDEN_DIM_1 = 64

HIDDEN_DIM_2 = 32

LATENT_DIM = 16

EPOCHS = 100

BATCH_SIZE = 256

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-5


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(
    RANDOM_STATE
)

np.random.seed(
    RANDOM_STATE
)


# ============================================================
# AUTOENCODER
# ============================================================

class Autoencoder(
    nn.Module
):

    def __init__(
        self,
        input_dim: int,
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_DIM_1,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM_1,
                HIDDEN_DIM_2,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM_2,
                LATENT_DIM,
            ),
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                LATENT_DIM,
                HIDDEN_DIM_2,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM_2,
                HIDDEN_DIM_1,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM_1,
                input_dim,
            ),
        )

    def encode(
        self,
        x,
    ):

        return self.encoder(x)

    def forward(
        self,
        x,
    ):

        latent = self.encoder(x)

        reconstruction = (
            self.decoder(latent)
        )

        return (
            reconstruction,
            latent,
        )


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(
    path: Path,
    name: str,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    if df.empty:

        raise ValueError(
            f"{name} is empty."
        )

    if "wallet_id" not in df.columns:

        raise ValueError(
            f"{name} must contain "
            "wallet_id."
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
# NUMERIC FEATURES
# ============================================================

def extract_numeric_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = pd.DataFrame(
        {
            "wallet_id":
                df["wallet_id"]
        }
    )

    for column in df.columns:

        if column == "wallet_id":
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.notna().sum() > 0:

            result[column] = values

    return result


# ============================================================
# FUSE FEATURES
# ============================================================

def build_feature_matrix():

    wallet = load_csv(
        WALLET_FEATURES_FILE,
        "wallet_features.csv",
    )

    graph = load_csv(
        GRAPH_FEATURES_FILE,
        "graph_features.csv",
    )

    graphsage = load_csv(
        GRAPHSAGE_FILE,
        "graphsage_embeddings.csv",
    )

    wallet = extract_numeric_features(
        wallet
    )

    graph = extract_numeric_features(
        graph
    )

    graphsage = extract_numeric_features(
        graphsage
    )

    # --------------------------------------------------------
    # Avoid accidental column collisions.
    # --------------------------------------------------------

    graph_columns = [
        column
        for column in graph.columns
        if column != "wallet_id"
    ]

    graph = graph.rename(
        columns={
            column:
                f"graph_{column}"
            for column in graph_columns
            if column in wallet.columns
        }
    )

    graphsage_columns = [
        column
        for column in graphsage.columns
        if column != "wallet_id"
    ]

    graphsage = graphsage.rename(
        columns={
            column:
                f"graphsage_{column}"
            for column in graphsage_columns
            if column in wallet.columns
        }
    )

    # --------------------------------------------------------
    # Inner join ensures every row has all required
    # representations.
    # --------------------------------------------------------

    fused = wallet.merge(
        graph,
        on="wallet_id",
        how="inner",
        validate="one_to_one",
    )

    fused = fused.merge(
        graphsage,
        on="wallet_id",
        how="inner",
        validate="one_to_one",
    )

    if fused.empty:

        raise ValueError(
            "Feature fusion produced zero rows."
        )

    # --------------------------------------------------------
    # Numeric cleaning.
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in fused.columns
        if column != "wallet_id"
    ]

    fused[feature_columns] = (
        fused[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

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
    # Remove constant columns.
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in feature_columns
        if fused[column].nunique() > 1
    ]

    if not feature_columns:

        raise ValueError(
            "No variable numeric features "
            "remain after preprocessing."
        )

    return (
        fused[
            ["wallet_id"]
            + feature_columns
        ],
        feature_columns,
    )


# ============================================================
# TRAIN AUTOENCODER
# ============================================================

def train_autoencoder(
    X_scaled: np.ndarray,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    dataset = torch.utils.data.TensorDataset(
        X_tensor
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = Autoencoder(
        input_dim=X_scaled.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        total_loss = 0.0

        sample_count = 0

        for (
            batch,
        ) in loader:

            batch = (
                batch[0]
                .to(device)
            )

            optimizer.zero_grad()

            reconstruction, _ = (
                model(batch)
            )

            loss = criterion(
                reconstruction,
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

            sample_count += (
                batch_size
            )

        epoch_loss = (
            total_loss
            / sample_count
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:03d}/{EPOCHS} "
                f"loss={epoch_loss:.6f}"
            )

    return (
        model,
        device,
    )


# ============================================================
# CALCULATE RECONSTRUCTION ERROR
# ============================================================

def calculate_scores(
    model,
    device,
    X_scaled,
):

    model.eval()

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():

        reconstruction, latent = (
            model(X_tensor)
        )

        errors = torch.mean(
            (
                X_tensor
                - reconstruction
            ) ** 2,
            dim=1,
        )

    return (
        errors.cpu().numpy(),
        latent.cpu().numpy(),
    )


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_outputs(
    model,
    scaler,
    feature_columns,
    fused,
    scores,
):

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    output = pd.DataFrame(
        {
            "wallet_id":
                fused["wallet_id"],

            "autoencoder_reconstruction_error":
                scores,
        }
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Model checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim":
                len(feature_columns),

            "hidden_dim_1":
                HIDDEN_DIM_1,

            "hidden_dim_2":
                HIDDEN_DIM_2,

            "latent_dim":
                LATENT_DIM,

            "feature_columns":
                feature_columns,
        },
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    joblib.dump(
        {
            "scaler":
                scaler,

            "feature_columns":
                feature_columns,
        },
        SCALER_FILE,
    )

    print(
        f"\nScores saved:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Model saved:\n"
        f"{MODEL_FILE}"
    )

    print(
        f"Scaler saved:\n"
        f"{SCALER_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PYTORCH AUTOENCODER")
    print("=" * 60)

    fused, feature_columns = (
        build_feature_matrix()
    )

    print(
        f"Wallets: "
        f"{len(fused)}"
    )

    print(
        f"Input features: "
        f"{len(feature_columns)}"
    )

    X = (
        fused[feature_columns]
        .astype(np.float32)
        .values
    )

    # --------------------------------------------------------
    # Fit scaler on the current training dataset.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(X)
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model, device = (
        train_autoencoder(
            X_scaled
        )
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    scores, _ = (
        calculate_scores(
            model,
            device,
            X_scaled,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        model,
        scaler,
        feature_columns,
        fused,
        scores,
    )

    print("\n")
    print("=" * 60)
    print("AUTOENCODER COMPLETE")
    print("=" * 60)

    print(
        f"Mean reconstruction error: "
        f"{scores.mean():.6f}"
    )

    print(
        f"Maximum reconstruction error: "
        f"{scores.max():.6f}"
    )


if __name__ == "__main__":
    main()