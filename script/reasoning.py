import os
import sys
import math
import pprint
import polars as pl
import torch
import pickle

from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra

separator = ">" * 30
line = "-" * 30


def read_entity_vocab(cfg):
    """
    Reads entity_vocab from configuration file and returns the dictionary
    """
    with open(cfg.entity_vocab, "rb") as f:
        ent2name_dict = pickle.load(f)
    return ent2name_dict


def structure_results(paths, weights, cfg, dataset, entity_vocab=None, label=None):
    """
    Structure the reasoning paths and weights into a Polars DataFrame.
    """
    id2ent_dict = {v: k for k, v in dataset.entity_vocab.items()}
    id2rel_dict = {v: k for k, v in dataset.relation_vocab.items()}
    # add reverse edges
    id2rel_dict.update({k + len(id2rel_dict): v for k, v in id2rel_dict.items()})
    # convert ids to names
    df = (
        pl.DataFrame({"paths": paths, "weights": weights})
        .with_columns(pl.int_range(1, len(weights) + 1).alias("rank"))
        .explode("paths")
        .with_columns(
            pl.col("paths")
            .list.first()
            .cast(pl.String)
            .replace(id2ent_dict)
            .alias("head"),
            pl.col("paths")
            .list.get(1)
            .cast(pl.String)
            .replace(id2ent_dict)
            .alias("tail"),
            pl.col("paths")
            .list.last()
            .cast(pl.String)
            .replace(id2rel_dict)
            .alias("relation"),
        )
    )
    # add query triple information, expecting query to be of shape (1, 3)
    if label is not None:
        head_label, tail_label, relation_label = (
            id2ent_dict[label[:, 0].item()],
            id2ent_dict[label[:, 1].item()],
            id2rel_dict[label[:, 2].item()],
        )
        df = df.with_columns(
            pl.lit(head_label).alias("query_head"),
            pl.lit(tail_label).alias("query_tail"),
            pl.lit(relation_label).alias("query_relation"),
        )
    # translate ids to names if entity_vocab is provided
    if entity_vocab is not None:
        df = df.with_columns(
            pl.col("head").replace(entity_vocab).alias("head"),
            pl.col("tail").replace(entity_vocab).alias("tail"),
            pl.col("query_head").replace(entity_vocab).alias("query_head"),
            pl.col("query_tail").replace(entity_vocab).alias("query_tail"),
        )

    return df


def test(cfg, model, test_data, query, logger, dataset, entity_vocab):
    world_size = util.get_world_size()
    rank = util.get_rank()

    # build distributed test loader
    test_triplets = torch.cat(
        [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]
    ).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(
        test_triplets, cfg.train.batch_size, sampler=sampler
    )

    model.eval()
    # Initialize the model with a mock input
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            t_batch, h_batch = tasks.all_negative(test_data, batch)
            break
        query_rels = t_batch[:, 0, 2]
        relation_representation = model.relation_model(
            test_data.relation_graph, query=query_rels
        )
        # run entity model to get scores for all entities for each query
        score = model.entity_model(test_data, relation_representation, t_batch)

    # Check query shape
    if not query.size(0) >= 1 or not query.size(1) == 3:
        raise ValueError(
            "Triple must be of shape (1, 3) or (N, 3) where each triple is (head, tail, relation)"
        )

    # Generate reasoning paths for each query triple
    df_list = list()
    for q in query:
        q = q.unsqueeze(0)  # Make it (1, 3)
        paths, weights = model.entity_model.visualize(data=test_data, batch=q)
        # structure the results into a DataFrame
        df = structure_results(
            paths,
            weights,
            cfg,
            dataset,
            entity_vocab=entity_vocab,
            label=q,
        )
        df_list.append(df)
    df = pl.concat(df_list)

    # Simplify full paths
    df = (
        df.with_columns(
            pl.col("relation").cast(pl.List(pl.String)),
            ht=(
                pl.col("head")
                .cast(pl.List(pl.String))
                .list.concat(pl.col("tail").cast(pl.List(pl.String)))
            ),
        )
        .group_by(
            ["rank", "weights", "query_head", "query_relation", "query_tail"],
            maintain_order=True,
        )
        .agg([pl.col("paths"), pl.col("ht").flatten(), pl.col("relation").flatten()])
        .with_columns(pl.col("ht").list.unique(maintain_order=True))
        .with_columns(
            pl.col("ht").list.first().alias("first"), pl.col("ht").list.slice(1)
        )
        .with_columns(
            pl.struct(["ht", "relation"])
            .map_elements(
                lambda x: [
                    item for pair in zip(x["relation"], x["ht"]) for item in pair
                ],
                return_dtype=pl.List(pl.String),
            )
            .alias("interleaved")
        )
        .with_columns(pl.concat_list(["first", "interleaved"]).alias("full_path"))
    )[
        [
            "rank",
            "paths",
            "weights",
            "query_head",
            "query_relation",
            "query_tail",
            "full_path",
        ]
    ]

    # export results
    output_path = os.path.join(
        cfg.output_dir,
        cfg.model["class"],
        cfg.dataset["class"],
        f"reasoning_results_{rank}.parquet",
    )
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning(f"Exporting results to: {output_path}")
        logger.warning(separator)

    df.write_parquet(output_path)


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = util.build_dataset(cfg)
    # device = util.get_device(cfg)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    infer_data = util.reasoning_data(cfg, dataset)
    entity_vocab = read_entity_vocab(cfg)

    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning(f"Running reasoning ...")

    infer_res = test(
        cfg,
        model,
        test_data,
        infer_data,
        logger=logger,
        dataset=dataset,
        entity_vocab=entity_vocab,
    )
