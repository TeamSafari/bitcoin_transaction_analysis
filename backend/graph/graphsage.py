from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        embedding_dim=32
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            embedding_dim
        )

        self.relu = nn.ReLU()


    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = self.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


class GraphSAGEModel:

    def __init__(
        self,
        model_path: Path
    ):

        checkpoint = torch.load(
            model_path,
            map_location="cpu"
        )

        self.feature_columns = (
            checkpoint[
                "node_feature_columns"
            ]
        )

        self.model = GraphSAGEEncoder(
            input_dim=
                checkpoint["input_dim"],

            hidden_dim=
                checkpoint["hidden_dim"],

            embedding_dim=
                checkpoint["embedding_dim"]
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()


    def create_data(
        self,
        features: pd.DataFrame,
        edges: pd.DataFrame
    ):

        features = features.copy()
        edges = edges.copy()

        features["wallet_id"] = (
            features["wallet_id"]
            .astype(str)
        )

        edges["source_wallet_id"] = (
            edges["source_wallet_id"]
            .astype(str)
        )

        edges["target_wallet_id"] = (
            edges["target_wallet_id"]
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

        # Node feature matrix
        X = (
            features[
                self.feature_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .fillna(0)
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

        forward = np.array(
            [
                sources,
                targets
            ],
            dtype=np.int64
        )

        reverse = np.array(
            [
                targets,
                sources
            ],
            dtype=np.int64
        )

        edge_index = np.concatenate(
            [
                forward,
                reverse
            ],
            axis=1
        )

        return (
            Data(
                x=torch.tensor(
                    X,
                    dtype=torch.float32
                ),
                edge_index=torch.tensor(
                    edge_index,
                    dtype=torch.long
                )
            ),
            wallet_ids
        )


    def generate_embeddings(
        self,
        features: pd.DataFrame,
        edges: pd.DataFrame
    ):

        data, wallet_ids = (
            self.create_data(
                features,
                edges
            )
        )

        with torch.no_grad():

            embeddings = self.model(
                data.x,
                data.edge_index
            )

        return (
            wallet_ids,
            embeddings.numpy()
        )