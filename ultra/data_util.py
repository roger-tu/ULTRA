# pathing
import os
import os.path as osp
import pickle
import pathlib as Path

# data sci
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
    return pl.concat(
        [
            pl.read_parquet(os.path.join(path, i))
            for i in os.listdir(path)
            if i.endswith(".parquet")
        ]
    )


def load_id_dict(
    path: str = "/home/sagemaker-user/git/Ultra/PrimeKG1/",
) -> (dict, dict):
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
        try:
            # if pickle files don't exist, load the dataset
            dataset = getattr(datasets, Path.Path(data_dir).name)
            dataset = dataset(
                root="/home/sagemaker-user/knowledge-graph-workflows-and-models-team-primeKG/data/"
            )
            id2ent_dict = {v: k for k, v in dataset.entity_vocab.items()}
            id2rel_dict = {v: k for k, v in dataset.relation_vocab.items()}
            # export
            with open(os.path.join(path, "id2ent_dict.pkl"), "wb") as f:
                pickle.dump(id2ent_dict, f)

            with open(os.path.join(path, "id2rel_dict.pkl"), "wb") as f:
                pickle.dump(id2rel_dict, f)
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


def filter_process_results(df, results_path, filter_ent=None) -> pl.DataFrame:
    """
    returns a dataframe with prediction scores as well as prediction novelty (whether or not the edge already exists in primekg)
    the dataframe is filtered by gene_queries if it's not None
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
