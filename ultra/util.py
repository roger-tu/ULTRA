import os
import sys
import ast
import copy
import time
import logging
import argparse
import json
import yaml
import jinja2
from jinja2 import meta
import easydict

import polars as pl
import pickle
import torch
from torch import distributed as dist
from torch_geometric.data import Data
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR

from ultra import models, datasets, tasks


logger = logging.getLogger(__file__)


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        # Try parsing as JSON if ast.literal_eval fails
        try:
            return json.loads(string)
        except:  # (ValueError, json.JSONDecodeError):
            return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument(
        "-s", "--seed", help="random seed for PyTorch", type=int, default=1024
    )

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def broadcast(to_cast: list):
    """
    barrier to capture workers until workers are complete and conduct broadcast op
    """
    if get_world_size() > 1:
        dist.barrier()
        dist.broadcast_object_list(object_list=to_cast, src=0)


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    if "out" in cfg.model.keys():
        output = f"{cfg.model['out']}-{time.strftime('%m-%d-%H-%M-%S')}"
    else:
        output = time.strftime("%Y-%m-%d-%H-%M-%S")
    working_dir = os.path.join(
        os.path.expanduser(cfg.logging_output_dir),
        cfg.model["class"],
        cfg.dataset["class"],
        f"{output}",
    )

    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def build_dataset(cfg):
    data_config = copy.deepcopy(cfg.dataset)
    cls = data_config.pop("class")

    ds_cls = getattr(datasets, cls)
    dataset = ds_cls(**data_config)

    if get_rank() == 0:
        logger.warning(
            "%s dataset"
            % (cls if "version" not in cfg.dataset else f"{cls}({cfg.dataset.version})")
        )
        if cls != "JointDataset":
            logger.warning(
                "#train: %d, #valid: %d, #test: %d"
                % (
                    dataset[0].target_edge_index.shape[1],
                    dataset[1].target_edge_index.shape[1],
                    dataset[2].target_edge_index.shape[1],
                )
            )
        else:
            logger.warning(
                "#train: %d, #valid: %d, #test: %d"
                % (
                    sum(d.target_edge_index.shape[1] for d in dataset._data[0]),
                    sum(d.target_edge_index.shape[1] for d in dataset._data[1]),
                    sum(d.target_edge_index.shape[1] for d in dataset._data[2]),
                )
            )

    return dataset


def get_entity_relation_dict(working_dir, dataset):
    """
    Extract entity-relation dictionary from dataset object and export it if it doesn't exist at cfg.logging_output_dir
    """
    # output directory
    output_dir = os.path.dirname(working_dir)
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # check if dictionary files already exist
    for i, f in enumerate(
        ["id2ent_dict.pkl", "ent2id_dict.pkl", "id2rel_dict.pkl", "rel2id_dict.pkl"]
    ):
        f_output_dir = os.path.join(output_dir, f)
        # if dictionary doesn't exist, create it
        if not os.path.exists(f_output_dir):
            # create dictionary
            if i == 0:
                with open(f_output_dir, "wb") as fout:
                    id2ent_dict = {v: k for k, v in dataset.entity_vocab.items()}
                    pickle.dump(id2ent_dict, fout)
            elif i == 1:
                with open(f_output_dir, "wb") as fout:
                    ent2id_dict = dataset.entity_vocab
                    pickle.dump(ent2id_dict, fout)
            elif i == 2:
                with open(f_output_dir, "wb") as fout:
                    id2rel_dict = {v: k for k, v in dataset.relation_vocab.items()}
                    pickle.dump(id2rel_dict, fout)
            else:
                with open(f_output_dir, "wb") as fout:
                    rel2id_dict = dataset.relation_vocab
                    pickle.dump(rel2id_dict, fout)
        else:
            # load dictionary

            if i == 0:
                with open(f_output_dir, "rb") as fin:
                    id2ent_dict = pickle.load(fin)
            elif i == 1:
                with open(f_output_dir, "rb") as fin:
                    ent2id_dict = pickle.load(fin)
            elif i == 2:
                with open(f_output_dir, "rb") as fin:
                    id2rel_dict = pickle.load(fin)
            else:
                with open(f_output_dir, "rb") as fin:
                    rel2id_dict = pickle.load(fin)

    return id2ent_dict, ent2id_dict, id2rel_dict, rel2id_dict


def translate_entity(dataset, entity: str):
    """
    Translate entity string to integer index using dataset entity vocabulary
    """
    ent2id_dict = dataset.entity_vocab
    try:
        return ent2id_dict[entity]
    except:
        raise KeyError(
            f'Entity, {entity}, not found in graph vocabulary. Please check string follows appropriate format (i.e "SOURCE:12345" or "MONDO:1024").'
        )


def translate_relation(dataset, relation: str):
    """
    Translate relation string to integer index using dataset relation vocabulary
    """
    rel_dict = dataset.relation_vocab
    try:
        return rel_dict[relation]
    except:
        raise KeyError(
            f'Relation, {relation}, not found in graph vocabulary. Please check string follows appropriate format (i.e "associated with").'
        )


def build_graph(dataset, translate=False):
    """
    Build a polars dataframe representation of the graph from the dataset object.
    If translate is True, will convert entity and relation strings to integer indices using dataset vocabularies.
    """

    g = pl.concat(
        [
            pl.read_csv(
                os.path.join(dataset.raw_dir, "train.txt"),
                separator="\t",
                new_columns=["h", "r", "t"],
            ),  # train
            pl.read_csv(
                os.path.join(dataset.raw_dir, "valid.txt"),
                separator="\t",
                new_columns=["h", "r", "t"],
            ),  # valid
            pl.read_csv(
                os.path.join(dataset.raw_dir, "test.txt"),
                separator="\t",
                new_columns=["h", "r", "t"],
            ),  # test
        ]
    )

    if translate:
        ent_dict = dataset.entity_vocab
        rel_dict = dataset.relation_vocab
        g = g.with_columns(
            pl.col("h").replace(ent_dict).cast(pl.Int64),
            pl.col("t").replace(ent_dict).cast(pl.Int64),
            pl.col("r").replace(rel_dict).cast(pl.Int64),
        )

    return g


def inference_data_single(
    dataset,
    head_entity: str = "MONDO:1024",
    relation: str = "associated with",
):
    """
    Extract all facts matching `head_entity` and `relation`, transforms the set of triples/facts into a torch_geometric.data.Data object.
    Example: given the double ("pneumonic plague", "associated with") extract all triples that match ("MONDO:1024", "associated with", [tail entity]), converts them to (69738, 10, [tail inv_entity_vocab]) and finally transforms into a Data object.
    """

    # TODO: check if nodes / relations we are querying exists, if not, throw exception
    # Translate head entity and relation to integer index
    trans_h_ent = translate_entity(dataset, head_entity)
    trans_t_ent_dummy = translate_entity(
        dataset, "NCBI:5340"
    )  # Plasminogen gene is our dummy
    trans_rel = translate_relation(dataset, relation)
    # build polars dataframe representation of the graph
    g = build_graph(dataset, translate=True)
    # extract all relevant answers and translate them
    filtered_g = g.filter(pl.col("h") == trans_h_ent, pl.col("r") == trans_rel)
    if filtered_g.shape[0] == 0:
        #  if true head/rel combination doesn't exist, just generate a dummy one to make predictions on
        target_edge_index = torch.tensor(
            [[trans_h_ent], [trans_t_ent_dummy]]
        )  # [[x], [y]] size (2,1)
        target_edge_type = torch.tensor([trans_rel])  # [10] size 1

    elif filtered_g.shape[0] == 1:
        # if exactly 1 entry
        target_edge_index = (
            filtered_g.select(["h", "t"]).to_torch().t()
        )  # 101rows x 2 col -> 2 rows x 101 col
        target_edge_type = torch.tensor([trans_rel])  # [10] size 1

    else:
        # get the set of real edges and their ranks given a h,r combo
        target_edge_index = (
            filtered_g.select(["h", "t"]).to_torch().t()
        )  # 101rows x 2 col -> 2 rows x 101 col
        target_edge_type = (
            filtered_g.select("r").to_torch().t().squeeze()
        )  # 101 rows x 1 col -> 1 rows x 101 col -> 0 rows x 101 cols

    # creates torch_geometric graph dataset
    inference_data = Data(
        edge_index=dataset[0].edge_index,
        edge_type=dataset[0].edge_type,
        target_edge_index=target_edge_index,
        target_edge_type=target_edge_type,
        num_relations=dataset[0].num_relations,
        num_nodes=dataset[0].num_nodes,
        relation_graph=dataset[0].relation_graph,
    )

    # adds relation graph
    # inference_data = tasks.build_relation_graph(inference_data)

    return inference_data


def inference_data_batch(dataset, inference_file: str):
    """
    Load a batch of inference triples from a tsv file and transform them into a torch_geometric.data.Data object.
    The tsv file should have three columns: h_ent, rel, t_ent.
    """
    # load inference triples
    infer_triples = pl.read_csv(
        inference_file,
        separator="\t",
        new_columns=["h", "r", "t"],
    )
    # translate to integer indices
    ent_dict = dataset.entity_vocab
    rel_dict = dataset.relation_vocab
    infer_triples = infer_triples.with_columns(
        pl.col("h").replace(ent_dict).cast(pl.Int64),
        pl.col("t").replace(ent_dict).cast(pl.Int64),
        pl.col("r").replace(rel_dict).cast(pl.Int64),
    )
    # transform into torch_geometric.data.Data object
    data = Data(
        edge_index=dataset[0].edge_index,
        edge_type=dataset[0].edge_type,
        target_edge_index=infer_triples[["h", "t"]].to_torch().t(),
        target_edge_type=infer_triples["r"].to_torch(),
        # target_edge_index = target_edge_index,
        # target_edge_type = target_edge_type,
        num_relations=dataset[0].num_relations,
        num_nodes=dataset[0].num_nodes,
        relation_graph=dataset[0].relation_graph,
    )
    # adds relation graph
    # data = tasks.build_relation_graph(data)

    return data


def reasoning_data(cfg, dataset):
    """Generate reasoning data object from cfg.infer triple_list or h_ent, rel, t_ent"""
    ent_dict = dataset.entity_vocab
    rel_dict = dataset.relation_vocab

    # check for correct formatting
    if "triple_list" in cfg.infer.keys():
        triple_list = cfg.infer.triple_list
        if not isinstance(triple_list, list):
            raise ValueError(
                "cfg.infer['triple_list'] must be a list of triples in the format of [{'h_ent': 'MONDO:12345', 'rel': 'associated with', 't_ent': 'CHEMBL.COMPOUND:CHEMBL112'}, ..., ]"
            )
        for triple in triple_list:
            if not isinstance(triple, dict):
                raise ValueError(
                    "Each entry in cfg.infer['triple_list'] must be a dict in the format of {'h_ent': 'MONDO:12345', 'rel': 'associated with', 't_ent': 'CHEMBL.COMPOUND:CHEMBL112'}"
                )
            if not all(key in triple.keys() for key in ["h_ent", "rel", "t_ent"]):
                raise ValueError(
                    f"{triple} is an invalid Entry. Each entry in cfg.infer['triple_list'] must have keys 'h_ent', 'rel', 't_ent'"
                )
        # construct triple list from cfg.infer['triple_list']
        reasoning_data = torch.tensor(
            [
                [
                    ent_dict[triple["h_ent"]],
                    ent_dict[triple["t_ent"]],
                    rel_dict[triple["rel"]],
                ]
                for triple in triple_list
            ]
        )  # size (N, 3)

    return reasoning_data
