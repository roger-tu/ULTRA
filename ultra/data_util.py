# pathing
import os
import os.path as osp
import pickle
import pathlib as Path

# data sci
from click import Tuple
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ultra import datasets


def combine_parquet_results(path: str) -> pl.DataFrame:
    """
    returns a dataframe combining the ultra inference predictions made in a particular file path
    """

    df_ls = []
    # iterate through parquet files in the path
    for i in os.listdir(path):
        if i.endswith(".parquet"):
            # error handling if parquet file is malformed (just in case)
            try:
                df_ls.append(pl.read_parquet(osp.join(path, i)))
            except Exception as e:
                print(f"Error reading {i}:\n{e}")

    return pl.concat(df_ls)


def load_id_dict(
    path: str = "/home/sagemaker-user/git/Ultra/PrimeKG1/",
) -> tuple[dict, dict]:
    """
    loads pickled dictionaries for embedding to entity and relation dictionaries,
    and entity to name dictionary, respectively
    """
    # load id2ent and id2rel dict, if it doesn't exist, load the dataset
    try:
        with open(os.path.join(path, "id2ent_dict.pkl"), "rb") as f:
            id2ent_dict = pickle.load(f)

        with open(os.path.join(path, "id2rel_dict.pkl"), "rb") as f:
            id2rel_dict = pickle.load(f)

    except:
            # default fail, raise error
        raise ValueError(
                "id2ent_dict or id2rel_dict cannot be loaded. Please check for missing pickle files."
            )

    try:
        with open(os.path.join(path, "ent2name_dict.pkl"), "rb") as f:
            ent2name_dict = pickle.load(f)
    except:
        raise ValueError(f"ent2name_dict not in {path}")

    return id2ent_dict, id2rel_dict, ent2name_dict


def load_nodes(
    data_path: str = "~/knowledge-graph-workflows-and-models-team-primeKG/data/nodes.txt",
) -> pl.DataFrame:
    """
    returns a dataframe of nodes with columns ['name','type','source','source_id','source_label']
    :data_path: is the path (including file name) to the directory of the dataset
    """
    nodes = pl.read_csv(
        data_path,
        separator="\t",
        schema={
            "name": pl.String,
            "type": pl.String,
            "source": pl.String,
            "source_id": pl.String,
            "source_label": pl.String,
        },
    )

    return nodes


def load_graph(
    data_path: str = "~/knowledge-graph-workflows-and-models-team-primeKG/data/primekg1/raw/",
    node_path: str = "~/knowledge-graph-workflows-and-models-team-primeKG/data/nodes.txt",
) -> pl.DataFrame:
    """
    returns a dataframe of edges with columns ['h','r','t','h_type','t_type']
    :data_path: is the path (including file name) to the directory of the dataset
    """
    graph = pl.concat(
        [
            pl.read_csv(
                os.path.join(data_path, "train.txt"),
                separator="\t",
                new_columns=["h", "r", "t"],
            ),
            pl.read_csv(
                os.path.join(data_path, "test.txt"),
                separator="\t",
                new_columns=["h", "r", "t"],
            ),
            pl.read_csv(
                os.path.join(data_path, "valid.txt"),
                separator="\t",
                new_columns=["h", "r", "t"],
            ),
        ]
    )

    nodes = load_nodes(node_path)
    # add head and tail types
    graph = graph.join(
        nodes[["source_label", "type"]].rename({"source_label": "h", "type": "h_type"}),
        on="h",
        how="left",
    ).join(
        nodes[["source_label", "type"]].rename({"source_label": "t", "type": "t_type"}),
        on="t",
        how="left",
    )

    return graph


def get_graph_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    returns a dataframe with the graph schema of the input dataframe
    :df: is a polars dataframe of triples with columns ['h','r','t','h_type','t_type']
    :schema: is a polars dataframe of unique ['h_type','r','t_type'] combinations
    """
    schema = df[["h_type", "r", "t_type"]].unique()
    return schema


def translate_hrt(df, data_path: str) -> pl.DataFrame:
    """
    returns a dataframe that translates results from embedding to cui identifier
    :df: is a polars dataframe of triples with columns ['h','r','t']
    :data_path: is the path to the directory of the dataset
    """

    # load translation dictionary
    id2ent, id2rel, ent2name = load_id_dict(data_path)

    # translate the results
    df = df.with_columns(  # cui
        pl.col("h").cast(pl.String).replace(id2ent).alias("h_label"),
        pl.col("t").cast(pl.String).replace(id2ent).alias("t_label"),
        pl.col("r").cast(pl.String).replace(id2rel).alias("r_label"),
    ).with_columns(  # natural language name
        pl.col("h_label").replace(ent2name).alias("h_name"),
        pl.col("t_label").replace(ent2name).alias("t_name"),
    )

    return df


def load_and_translate_results(
    data_path: str, results_folder: str, top_k: int = None
) -> pl.DataFrame:
    """
    returns a dataframe that translates results from embedding to cui identifier
    :data_path: is the path to the directory of the dataset
    :results_folder: is the folder name of the data you'd like to process
    :top_k: keep the top k results, default is None
    """
    # get directory to the results and combine them
    df = combine_parquet_results(os.path.join(data_path, results_folder)).unique()

    # translate results
    df = translate_hrt(df, data_path)

    return df


def filter_process_results(df, results_path, filter_ent: list = None) -> pl.DataFrame:
    """
    returns a dataframe with prediction scores as well as prediction novelty (whether or not the edge already exists in primekg)
    the dataframe is filtered for entities in :filter_ent: if it's not None

    """
    # get translations
    id2ent, id2rel, ent2name = load_id_dict(results_path)
    # get known links
    known_associations_df = (
        df[["h_label", "r_label", "t_label"]]
        .group_by(["h_label", "r_label"])
        .agg("t_label")
    )
    # sort the score list and associate them to unfiltered rank
    score_df = (
        df.with_columns(pl.col("t_pred_score").list.sort(descending=True))[
            ["h_label", "h_name", "r_label", "t_pred_unfilt", "t_pred_score"]
        ]
        .explode(["t_pred_unfilt", "t_pred_score"])
        .with_columns(pl.col("t_pred_unfilt").cast(pl.String).replace(id2ent))
    )
    # filter df for genes of interest
    if filter_ent is not None:
        score_df = score_df.filter(pl.col("t_pred_unfilt").is_in(filter_ent))
    # join known links with scored dataframe, and label if edge is novel
    score_df = (
        score_df.join(known_associations_df, on=["h_label", "r_label"], how="left")
        .with_columns(
            pl.col("t_pred_unfilt").is_in("t_label").alias("edge_in_primekg"),
            pl.col("t_pred_unfilt").replace(ent2name).alias("t_pred_name"),
        )
        .drop("t_label")
        .rename({"t_pred_unfilt": "t_pred_label"})
    )[
        [
            "h_label",
            "t_pred_label",
            "h_name",
            "r_label",
            "t_pred_name",
            "t_pred_score",
            "edge_in_primekg",
        ]
    ]

    return score_df


def structure_results(
    df: pl.DataFrame,
    node_path="~/knowledge-graph-workflows-and-models-team-primeKG/data/nodes.txt",
    graph_path="~/knowledge-graph-workflows-and-models-team-primeKG/data/primekg1/raw/",
    top_k=None,
) -> pl.DataFrame:
    """
    returns a dataframe with predictions filtered by graph schema
    :df: is a polars dataframe of triples with columns ['h_label','h_name','r_label','t_pred_label','t_pred_name','t_pred_score','edge_in_primekg'] extracted from filter_process_results()
    :node_path: is the path (including file name) to the directory of the dataset nodes
    :graph_path: is the path to the directory of the dataset graph
    :top_k: keep the top k results, default is None which keeps all results
    """
    # load nodes and graph schema
    nodes = load_nodes(node_path)
    schema = get_graph_schema(load_graph(graph_path, node_path))

    # add head and tail types to results, merge schema, filter by schema, group results to clean up
    df = (
        df.join(
            nodes[["source_label", "type"]].rename(
                {"source_label": "h_label", "type": "h_type"}
            ),
            on="h_label",
            how="left",
        )  # get h_type
        .join(
            nodes[["source_label", "type"]].rename(
                {"source_label": "t_pred_label", "type": "t_pred_type"}
            ),
            on="t_pred_label",
            how="left",
        )  # get pred_t_type
        .join(
            schema.group_by(["h_type", "r"])
            .agg("t_type")
            .rename({"r": "r_label", "t_type": "schema_t_type"}),
            on=["h_type", "r_label"],
            how="left",
        )  # merge schema
        .with_columns(
            pl.col("t_pred_type").is_in(pl.col("schema_t_type")).alias("schema_match")
        )  # check schema
        .drop("schema_t_type")
        .filter(
            pl.col("schema_match") == True
        )  # keep predictions that match the schema
        .drop("schema_match")
        .sort("t_pred_score", descending=True)
        .group_by(["h_label", "h_name", "h_type", "r_label"], maintain_order=True)
        .agg(
            [
                "t_pred_label",
                "t_pred_name",
                "t_pred_score",
                "t_pred_type",
                "edge_in_primekg",
            ]
        )
    )
    if top_k is not None:
        df = df.with_columns(
            pl.col("t_pred_label").list.head(top_k + 1),
            pl.col("t_pred_name").list.head(top_k + 1),
            pl.col("t_pred_score").list.head(top_k + 1),
            pl.col("t_pred_type").list.head(top_k + 1),
            pl.col("edge_in_primekg").list.head(top_k + 1),
        )

    return df


def extract_ht_score(df: pl.DataFrame) -> dict:
    """
    returns a dict with extracted score for known tail predictions as a tuple

    {'query':(h,r), 'top':t1, 'bottom':t2, 'top_score':t1_score, 'bot_score':t2_score}
    """
    # best answers have lower rank, more positive score
    df = df.with_columns(pl.col("t_pred_score").list.get("t").alias("t_score")).sort(
        "t_unfilt_rank"
    )

    # since answers are sorted we get the best and worse from head and tail of df
    # bwka = best-worst-known-answers
    bwka = pl.concat([df.head(1), df.tail(1)])

    results = {
        "query": (bwka["h_name"].first(), bwka["r_label"].first()),  # query:(h,r)
        "top": (
            bwka["t_name"].first(),
            bwka["t_score"].first(),
        ),  # top:(t_best, t_best_score)
        "bottom": (
            bwka["t_name"].last(),
            bwka["t_score"].last(),
        ),  # bottom:(t_worst, t_worst_score)
    }

    return results


def get_last_run_path(path: str) -> Path:
    """
    Checks given directory path for the last created folder and returns it
    """
    # all items in directory path sorted from newest -> oldest
    paths = sorted(Path.Path(path).iterdir(), key=os.path.getmtime, reverse=True)
    # check if path is a directory, get first directory
    for p in paths:
        if p.is_dir():
            return p
