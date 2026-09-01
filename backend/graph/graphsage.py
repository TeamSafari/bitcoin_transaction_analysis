from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "wallet_features_scaled.csv"
)

EDGE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
    / "graph_edges.csv"
)

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
    / "graphsage_embeddings.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "artifacts"
    / "graphsage.pt"
)


# ============================================================
# CONFIGURATION
# ============================================================

HIDDEN_DIM = 64

EMBEDDING_DIM = 32

EPOCHS = 100

LEARNING_RATE = 0.01

RANDOM_STATE = 42


# ============================================================
# MODEL
# ============================================================

class GraphSAGEEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim,
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            embedding_dim,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index,
        )

        return x


# ============================================================
# LOAD DATA
# ============================================================

def load_graph_data():

    features = pd.read_csv(
        FEATURE_FILE
    )

    edges = pd.read_csv(
        EDGE_FILE
    )

    if "wallet_id" not in features.columns:

        raise ValueError(
            "wallet_features_scaled.csv "
            "must contain wallet_id."
        )

    feature_columns = [
        column
        for column in features.columns
        if column != "wallet_id"
    ]

    features["wallet_id"] = (
        features["wallet_id"]
        .astype(str)
    )

    edges[
        "source_wallet_id"
    ] = (
        edges[
            "source_wallet_id"
        ]
        .astype(str)
    )

    edges[
        "target_wallet_id"
    ] = (
        edges[
            "target_wallet_id"
        ]
        .astype(str)
    )

    wallet_ids = (
        features["wallet_id"]
        .tolist()
    )

    wallet_to_index = {
        wallet_id: index
        for index, wallet_id
        in enumerate(wallet_ids)
    }

    X = (
        features[
            feature_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .astype(np.float32)
        .values
    )

    sources = []
    targets = []

    for _, row in edges.iterrows():

        source = row[
            "source_wallet_id"
        ]

        target = row[
            "target_wallet_id"
        ]

        if (
            source in wallet_to_index
            and
            target in wallet_to_index
        ):

            sources.append(
                wallet_to_index[source]
            )

            targets.append(
                wallet_to_index[target]
            )

    # GraphSAGE message passing uses both directions.
    edge_pairs = []

    for source, target in zip(
        sources,
        targets,
    ):

        edge_pairs.append(
            (source, target)
        )

        edge_pairs.append(
            (target, source)
        )

    if edge_pairs:

        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long,
        ).t().contiguous()

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    data = Data(
        x=torch.tensor(
            X,
            dtype=torch.float32,
        ),
        edge_index=edge_index,
    )

    return (
        data,
        wallet_ids,
        feature_columns,
        sources,
        targets,
    )


# ============================================================
# TRAIN
# ============================================================

def train_graphsage(
    data,
    sources,
    targets,
):

    torch.manual_seed(
        RANDOM_STATE
    )

    model = GraphSAGEEncoder(
        input_dim=data.x.shape[1],
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    if not sources:

        raise ValueError(
            "No graph edges available "
            "for GraphSAGE training."
        )

    positive_edges = torch.tensor(
        list(
            zip(
                sources,
                targets,
            )
        ),
        dtype=torch.long,
    )

    num_nodes = data.x.shape[0]

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    # --------------------------------------------------------
    # Negative samples
    # --------------------------------------------------------

    positive_set = {
        (int(source), int(target))
        for source, target
        in zip(sources, targets)
    }

    negative_pairs = []

    target_count = len(
        positive_edges
    )

    while len(negative_pairs) < target_count:

        source = int(
            rng.integers(
                0,
                num_nodes,
            )
        )

        target = int(
            rng.integers(
                0,
                num_nodes,
            )
        )

        if source == target:
            continue

        if (
            source,
            target,
        ) in positive_set:
            continue

        negative_pairs.append(
            (source, target)
        )

    negative_edges = torch.tensor(
        negative_pairs,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        optimizer.zero_grad()

        z = model(
            data.x,
            data.edge_index,
        )

        positive_scores = (
            (
                z[
                    positive_edges[:, 0]
                ]
                *
                z[
                    positive_edges[:, 1]
                ]
            )
            .sum(dim=1)
        )

        negative_scores = (
            (
                z[
                    negative_edges[:, 0]
                ]
                *
                z[
                    negative_edges[:, 1]
                ]
            )
            .sum(dim=1)
        )

        positive_loss = (
            -torch.log(
                torch.sigmoid(
                    positive_scores
                )
                + 1e-8
            ).mean()
        )

        negative_loss = (
            -torch.log(
                1.0
                -
                torch.sigmoid(
                    negative_scores
                )
                + 1e-8
            ).mean()
        )

        loss = (
            positive_loss
            + negative_loss
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 1
            or epoch % 20 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:03d}/{EPOCHS} "
                f"loss={loss.item():.6f}"
            )

    return model


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    model,
    data,
    wallet_ids,
    feature_columns,
):

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EMBEDDING_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    with torch.no_grad():

        embeddings = model(
            data.x,
            data.edge_index,
        ).cpu().numpy()

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    embedding_columns = [
        f"graphsage_dim_{i}"
        for i in range(
            embeddings.shape[1]
        )
    ]

    output = pd.DataFrame(
        embeddings,
        columns=embedding_columns,
    )

    output.insert(
        0,
        "wallet_id",
        wallet_ids,
    )

    output.to_csv(
        EMBEDDING_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save model checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim":
                data.x.shape[1],

            "hidden_dim":
                HIDDEN_DIM,

            "embedding_dim":
                EMBEDDING_DIM,

            "node_feature_columns":
                feature_columns,
        },
        MODEL_FILE,
    )

    print(
        f"\nEmbeddings saved:\n"
        f"{EMBEDDING_FILE}"
    )

    print(
        f"Model saved:\n"
        f"{MODEL_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GRAPHSAGE")
    print("=" * 60)

    (
        data,
        wallet_ids,
        feature_columns,
        sources,
        targets,
    ) = load_graph_data()

    print(
        f"Nodes: {len(wallet_ids)}"
    )

    print(
        f"Input features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Training edges: "
        f"{len(sources)}"
    )

    model = train_graphsage(
        data,
        sources,
        targets,
    )

    save_outputs(
        model,
        data,
        wallet_ids,
        feature_columns,
    )


if __name__ == "__main__":
    main()