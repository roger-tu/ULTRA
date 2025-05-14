import os
import sys
import ast
import copy
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import polars as pl

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
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

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


def broadcast(to_cast:list):
    '''
    barrier to capture workers until workers are complete and conduct broadcast op
    '''
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

    if 'out' in cfg.model.keys():
        output = f"{cfg.model['out']}-{time.strftime('%m-%d-%H-%M-%S')}"
    else:
        output = time.strftime('%Y-%m-%d-%H-%M-%S')
    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], f"{output}")

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
        logger.warning("%s dataset" % (cls if "version" not in cfg.dataset else f'{cls}({cfg.dataset.version})'))
        if cls != "JointDataset":
            logger.warning("#train: %d, #valid: %d, #test: %d" %
                        (dataset[0].target_edge_index.shape[1], dataset[1].target_edge_index.shape[1],
                            dataset[2].target_edge_index.shape[1]))
        else:
            logger.warning("#train: %d, #valid: %d, #test: %d" %
                           (sum(d.target_edge_index.shape[1] for d in dataset._data[0]),
                            sum(d.target_edge_index.shape[1] for d in dataset._data[1]),
                            sum(d.target_edge_index.shape[1] for d in dataset._data[2]),
                            ))

    return dataset

def inference_data(
    dataset,
    head_entity: str = 'MONDO:1024',
    relation: str = 'associated with',
):
    '''
    Extract all facts matching `head_entity` and `relation`, transforms the set of triples/facts into a torch_geometric.data.Data object. 
    Example: given the double ("pneumonic plague", "associated with") extract all triples that match ("MONDO:1024", "associated with", [tail entity]), converts them to (69738, 10, [tail inv_entity_vocab]) and finally transforms into a Data object.
    '''
    ent_dict = dataset.entity_vocab
    rel_dict = dataset.relation_vocab

    # check if nodes / relations we are querying exists, if not, throw exception
    try:
        trans_h_ent = ent_dict[head_entity]
        trans_t_ent_dummy = ent_dict['NCBI:5340'] # Plasminogen gene is our dummy
    except:
        raise KeyError(f'Entity, {trans_h_ent}, not found in graph vocabulary. Please check string follows appropriate format (i.e "SOURCE:12345" or "MONDO:1024").')
    try:
        trans_rel = rel_dict[relation]
    except:
        raise KeyError(f'Relation, {relation}, not found in graph vocabulary. Please check string follows appropriate format (i.e "associated with"')

    # extract all relevant answers and translate them
    g = (
        pl.concat(
            [
                pl.read_csv(os.path.join(dataset.raw_dir,'train.txt'),separator='\t',new_columns=['h','r','t']), # train
                pl.read_csv(os.path.join(dataset.raw_dir,'valid.txt'),separator='\t',new_columns=['h','r','t']), # valid
                pl.read_csv(os.path.join(dataset.raw_dir,'test.txt'),separator='\t',new_columns=['h','r','t']), # test
            ]
        )
        .filter(pl.col('h')==head_entity, pl.col('r')==relation)
        .with_columns(
            pl.col('h').replace(ent_dict).cast(pl.Int64),
            pl.col('t').replace(ent_dict).cast(pl.Int64),
            pl.col('r').replace(rel_dict).cast(pl.Int64)
        )
    )

    if g.shape[0]==0:
        #  if true head/rel combination doesn't exist, just generate a dummy one to make predictions on
        target_edge_index=torch.tensor([[trans_h_ent],[trans_t_ent_dummy]]) # [[x], [y]] size (2,1)
        target_edge_type=torch.tensor([trans_rel]) # [10] size 1
        
    elif g.shape[0]==1:
        # if exactly 1 entry
        target_edge_index = g.select(['h','t']).to_torch().t() #101rows x 2 col -> 2 rows x 101 col
        target_edge_type=torch.tensor([trans_rel]) # [10] size 1
        
    else:
        # get the set of real edges and their ranks given a h,r combo
        target_edge_index = g.select(['h','t']).to_torch().t() #101rows x 2 col -> 2 rows x 101 col
        target_edge_type = g.select('r').to_torch().t().squeeze() #101 rows x 1 col -> 1 rows x 101 col -> 0 rows x 101 cols
        
    # creates torch_geometric graph dataset
    inference_data = Data(
        edge_index=dataset[0].edge_index,
        edge_type=dataset[0].edge_type,
        target_edge_index=target_edge_index,
        target_edge_type=target_edge_type,
        num_relations=dataset[0].num_relations,
        num_nodes=dataset[0].num_nodes,
    )
    
    # adds relation graph
    inference_data = tasks.build_relation_graph(inference_data)
    
    return inference_data