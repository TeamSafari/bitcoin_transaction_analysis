import pandas as pd
import networkx as nx


def calculate_graph_features(
    graph: nx.DiGraph
) -> pd.DataFrame:

    nodes = list(
        graph.nodes()
    )

    degree = dict(
        graph.degree()
    )

    in_degree = dict(
        graph.in_degree()
    )

    out_degree = dict(
        graph.out_degree()
    )

    weighted_degree = dict(
        graph.degree(
            weight="transaction_count"
        )
    )

    weighted_in_degree = dict(
        graph.in_degree(
            weight="transaction_count"
        )
    )

    weighted_out_degree = dict(
        graph.out_degree(
            weight="transaction_count"
        )
    )

    if graph.number_of_edges() > 0:

        pagerank = nx.pagerank(
            graph,
            weight="transaction_count"
        )

        betweenness = (
            nx.betweenness_centrality(
                graph
            )
        )

    else:

        pagerank = {
            node: 0.0
            for node in nodes
        }

        betweenness = {
            node: 0.0
            for node in nodes
        }

    undirected = (
        graph.to_undirected()
    )

    clustering = nx.clustering(
        undirected
    )

    # -----------------------------
    # Communities
    # -----------------------------

    community_id = {}
    community_size = {}

    if undirected.number_of_edges() > 0:

        communities = (
            nx.community
            .greedy_modularity_communities(
                undirected
            )
        )

        for cid, community in enumerate(
            communities
        ):

            for node in community:

                community_id[node] = cid

                community_size[node] = (
                    len(community)
                )

    else:

        for node in nodes:

            community_id[node] = -1
            community_size[node] = 1

    # -----------------------------
    # Two-hop neighbors
    # -----------------------------

    two_hop = {}

    for node in nodes:

        first_hop = set(
            graph.successors(node)
        )

        second_hop = set()

        for neighbor in first_hop:

            second_hop.update(
                graph.successors(
                    neighbor
                )
            )

        second_hop.discard(node)

        second_hop -= first_hop

        two_hop[node] = len(
            second_hop
        )

    # -----------------------------
    # DataFrame
    # -----------------------------

    return pd.DataFrame({

        "wallet_id": nodes,

        "degree": [
            degree.get(node, 0)
            for node in nodes
        ],

        "in_degree": [
            in_degree.get(node, 0)
            for node in nodes
        ],

        "out_degree": [
            out_degree.get(node, 0)
            for node in nodes
        ],

        "weighted_degree": [
            weighted_degree.get(node, 0)
            for node in nodes
        ],

        "weighted_in_degree": [
            weighted_in_degree.get(node, 0)
            for node in nodes
        ],

        "weighted_out_degree": [
            weighted_out_degree.get(node, 0)
            for node in nodes
        ],

        "pagerank": [
            pagerank.get(node, 0.0)
            for node in nodes
        ],

        "betweenness_centrality": [
            betweenness.get(node, 0.0)
            for node in nodes
        ],

        "clustering_coefficient": [
            clustering.get(node, 0.0)
            for node in nodes
        ],

        "community_id": [
            community_id.get(node, -1)
            for node in nodes
        ],

        "community_size": [
            community_size.get(node, 1)
            for node in nodes
        ],

        "two_hop_neighbor_count": [
            two_hop.get(node, 0)
            for node in nodes
        ],
    })