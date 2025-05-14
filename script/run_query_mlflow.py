import os
import sys
import csv
import math
import time
import pprint
from itertools import islice
from tqdm import tqdm
from datetime import timedelta
import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch import distributed as dist
from torch.nn import functional as F
from torch.utils import data as torch_data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import datasets_query, tasks, util, query_utils
from ultra.models import Ultra
from ultra.ultraquery import UltraQuery
from ultra.query_utils import batch_evaluate, evaluate, gather_results
from ultra.variadic import variadic_softmax
from timeit import default_timer as timer

import mlflow

separator = ">" * 30
line = "-" * 30

def predict_and_target(model, graph, batch): # parallel_model, train_grpah, batch
    query = batch["query"]
    type = batch["type"]
    easy_answer = batch["easy_answer"]
    hard_answer = batch["hard_answer"]
    # dist.breakpoint(0)
    # turn off symbolic traversal at inference time
    pred = model(graph, query, symbolic_traversal=model.training)
    if not model.training:
        # eval
        target = (type, easy_answer, hard_answer)
        restrict_nodes = getattr(graph, "restrict_nodes", None)
        ranking, answer_ranking = batch_evaluate(pred, target, restrict_nodes)
        # answer set cardinality prediction
        prob = F.sigmoid(pred)
        num_pred = (prob * (prob > 0.5)).sum(dim=-1)
        num_easy = easy_answer.sum(dim=-1)
        num_hard = hard_answer.sum(dim=-1)
        return (ranking, num_pred), (type, answer_ranking, num_easy, num_hard)
    else:
        target = easy_answer.float()

    return pred, target

def train_and_validate(cfg, model, train_graph, train_data, valid_graph, valid_data, query_id2type, device, logger, batch_per_epoch=None):
    if cfg.train.num_epoch == 0:
        return
        
    world_size = util.get_world_size()
    rank = util.get_rank()

    sampler = torch_data.DistributedSampler(train_data, world_size, rank)
    train_loader = torch_data.DataLoader(train_data, cfg.train.batch_size, sampler=sampler)

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    # mlflow tracker tracking setting
    if cfg.train['mlflow_tracker'] and (util.get_rank() == 0):
        mlflow.set_tracking_uri(cfg.train.mlflow_arn)
        mlflow.set_experiment(cfg.train.mlflow_experiment)
        mlflow.start_run(run_name=cfg.train.mlflow_runid)
        mlflow.log_params(
            params = {
                'dataset':cfg.dataset['class'],
                'gpu':cfg.train['gpus'], 
                'batch_size':cfg.train['batch_size'],
                'batch_per_epoch':cfg.train['batch_per_epoch'],
                'num_epochs':cfg.train['num_epoch'],
                'optimizer':cls,
                'lr':cfg.optimizer['lr']
            }
        )
    
    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1
    
    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                if device.type == "cuda":
                    train_graph = train_graph.to(device)
                    batch = query_utils.cuda(batch, device=device)
                    # dist.breakpoint(0)
                pred, target = predict_and_target(parallel_model, train_graph, batch)

                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

                is_positive = target > 0.5
                is_negative = target <= 0.5
                num_positive = is_positive.sum(dim=-1)
                num_negative = is_negative.sum(dim=-1)

                neg_weight = torch.zeros_like(pred)
                neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(num_positive)

                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        logit = pred[is_negative] / cfg.task.adversarial_temperature
                        neg_weight[is_negative] = variadic_softmax(logit, num_negative)
                        #neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(num_negative)
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                    if cfg.train.mlflow_tracker:
                        mlflow.log_metric('binary_cross_entropy_loss', loss, batch_id)
                    # save model every 10k steps
                    if batch_id % (cfg.train.log_interval * 100) == 0:
                        torch.save(state, f"model_step_{batch_id}.pth")
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if util.get_rank() == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_graph, valid_data, query_id2type=query_id2type, device=device, logger=logger)
        util.synchronize()
        if util.get_rank() == 0:

            # add tracker for metrics
            if cfg.train.mlflow_tracker:
                mlflow.log_metric('avg_train_loss', avg_loss, epoch)
                mlflow.log_metric('valid_mrr', result['mrr'].item(), epoch)
                mlflow.log_metric('valid_hits-1', result['hits@1'].item(), epoch)
                mlflow.log_metric('valid_hits-3', result['hits@3'].item(), epoch)
                mlflow.log_metric('valid_hits-10', result['hits@10'].item(), epoch)
        util.synchronize()
        if result['mrr'] > best_result:
            best_result = result['mrr']
            best_epoch = epoch

    if util.get_rank() == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
        state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
        model.load_state_dict(state["model"])
        
    util.synchronize()
    if cfg.train.mlflow_tracker:
        mlflow.end_run()


@torch.no_grad()
def test(cfg, model, test_graph, test_data, query_id2type, device, logger, return_metrics=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    sampler = torch_data.DistributedSampler(test_data, world_size, rank)
    test_loader = torch_data.DataLoader(test_data, cfg.train.batch_size, sampler=sampler)

    model.eval()
    preds, targets = [], []
    for batch in tqdm(test_loader):
        if device.type == "cuda":
            test_graph = test_graph.to(device)
            batch = query_utils.cuda(batch, device=device)
        
        predictions, target = predict_and_target(model, test_graph, batch)
        preds.append(predictions)
        targets.append(target)
    
    pred = query_utils.cat(preds)
    target = query_utils.cat(targets)

    pred, target = gather_results(pred, target, rank, world_size, device)
    
    metrics = {}
    if rank == 0:
        metrics = evaluate(pred, target, cfg.task.metric, query_id2type)
        query_utils.print_metrics(metrics, logger)
    else:
        metrics['mrr'] = (1 / pred[0].float()).mean().item()
    util.synchronize()
    return metrics


if __name__ == "__main__":

    if util.get_world_size() > 1:
        # # storing vars to main host, set rank0 as main
        # if util.get_rank() == 0:
        #     server_store = dist.TCPStore(host_name="127.0.0.1", port=8890, is_master=True)
        # else:
        # # client to contact main host
        #     server_store = dist.TCPStore(host_name="127.0.0.1", port=8890, is_master=False)
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=1, minutes=30), # set longer timeout
            # store=server_store,
            # init_method="env://",
            # device_id=torch.device('cuda:0'),
            world_size=util.get_world_size(),
            rank=util.get_rank()
        )
    
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    torch.manual_seed(args.seed + util.get_rank())
    device = util.get_device(cfg)
    task_name = cfg.task["name"]
    eval_mode = cfg.task["mode"]
    
    # path = os.path.dirname(os.path.expanduser(__file__))
    path = working_dir
    results_file = os.path.join(path, f"ultraquery_results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
    
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning("World size: %d" % util.get_world_size())
        logger.warning(f"Device: {device}")
        logger.warning(f"Device Type: {device.type}")
        logger.warning(pprint.pformat(cfg))
    
    # tried to execute dataset generation to just 1 gpu so we don't overload memory, but it hangs at dist.broadcast_object_list()
    dataset = query_utils.build_query_dataset(cfg)
    #     if util.get_world_size() > 1:
    #         dummy = [dataset]
    # else:
    #     dummy = [None]
    # logger.warning(f"rank{util.get_rank()} reached barrier")
    # util.synchronize()
    # logger.warning(f"rank{util.get_rank()} reached barrier")
    # dist.broadcast_object_list(dummy, src=0)
    # util.synchronize()
    # if util.get_rank() != 0:
    #     dataset = dummy[0]
    
    train_data, valid_data, test_data = dataset.split()
    train_graph, valid_graph, test_graph = dataset.train_graph, dataset.valid_graph, dataset.test_graph
    
    model = UltraQuery(
        model=Ultra(
            rel_model_cfg=cfg.model.model.relation_model,
            entity_model_cfg=cfg.model.model.entity_model,
        ),
        logic=cfg.model.logic,
        dropout_ratio=cfg.model.dropout_ratio,
        threshold=cfg.model.threshold,
        more_dropout=cfg.model.get('more_dropout', 0.0),
    )
    
    # initialize with pre-trained ultra for link prediction
    if "ultra_ckpt" in cfg and cfg.ultra_ckpt is not None:
        state = torch.load(cfg.ultra_ckpt, map_location="cpu", weights_onlys=False)
        model.model.model.load_state_dict(state["model"])

    # initialize with a pre-trained ultraquery model for query answering
    if "ultraquery_ckpt" in cfg and cfg.ultraquery_ckpt is not None:
        state = torch.load(cfg.ultraquery_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    
    if "fast_test" in cfg.train:
        if util.get_rank() == 0:
            logger.warning("Quick test mode on. Only evaluate on %d samples for valid" % cfg.train.fast_test)
        g = torch.Generator()
        g.manual_seed(1024) #1024, 42, 580, 775, 958
        valid_data = torch_data.random_split(valid_data, [cfg.train.fast_test, len(valid_data) - cfg.train.fast_test], generator=g)[0]

    #     # broadcast decisions baed on eval mode. to decrease size and vol of objects
    #     if eval_mode == "train":
    #         broadcast_train = [
    #             train_data, train_graph, valid_data, valid_graph, dataset.id2type, model
    #         ]
    #         # train_graph, valid_graph
            
    #     elif eval_mode == "valid":
    #         broadcast_valid =[
    #             valid_data, valid_graph, dataset.id2type, model
    #         ]
    #     elif eval_mode == "test":
    #         broadcast_test = [
    #             test_data, test_graph, dataset.id2type, model
    #         ]
    #     else:
    #         raise ValueError(f"--mode parameter is invalid: {eval_mode} must be train, valid or test")
    # else:
    #     broadcast_train = [None] * 6
    #     broadcast_valid = [None] * 4
    #     broadcast_test = [None] * 4

    # # broadcast retrieval
    # if util.get_world_size() > 1:

    #     if eval_mode == "train":
    #         if util.get_rank() == 0:
    #             logger.warning("broadcasting train variables.")
    #         dist.broadcast_object_list(object_list=broadcast_train, src=0)
    #         dist.barrier()
    #         # retrieve
    #         train_data, train_graph = broadcast_train[0], broadcast_train[1]
    #         valid_data, valid_graph = broadcast_train[2], broadcast_test[3]
    #         dataset_id2type, model = broadcast_train[4], broadcast_train[5]
            
    #     elif eval_mode == "valid":
    #         if util.get_rank() == 0:
    #             logger.warning("broadcasting valid variables.")
    #         dist.broadcast_object_list(object_list=broadcast_valid, src=0)
    #         dist.barrier()
    #         valid_data, valid_graph = broadcast_valid[0], broadcast_valid[1]
    #         dataset_id2type, model = broadcast_valid[2], broadcast_valid[3]
                        
    #     elif eval_mode == "test":
    #         if util.get_rank() == 0:
    #             logger.warning("broadcasting test variables.")
    #         dist.broadcast_object_list(object_list=broadcast_test, src=0)
    #         dist.barrier()
    #         test_data, test_graph = broadcast_test[0], broadcast_test[1]
    #         dataset_id2type, model = broadcast_test[2], broadcast_test[3]
            
    model = model.to(device)

    if eval_mode=="train":
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Run training")
        dist.barrier()
        train_and_validate(cfg, model, train_graph, train_data, valid_graph, valid_data, query_id2type=dataset.id2type, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)

    elif eval_mode == "valid":
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        start = timer()
        val_metrics = test(cfg, model, valid_graph, valid_data, query_id2type=dataset.id2type, device=device, logger=logger)
        end = timer()

        if util.get_rank()==0:
            logger.warning(f"Valid time: {end - start}")
            # write to the log file
            val_metrics['dataset'] = str(dataset)
            query_utils.print_metrics_to_file(val_metrics, results_file)

    elif eval_mode == "test":
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on test")
        start = timer()
        metrics = test(cfg, model, test_graph, test_data, query_id2type=dataset.id2type, device=device, logger=logger)
        end = timer()
    
        # write to the log file
        if util.get_rank() == 0:
            logger.warning(f"Test time: {end - start}")
            metrics['dataset'] = str(dataset)
            query_utils.print_metrics_to_file(metrics, results_file)

    if util.get_world_size() >1:
        dist.destroy_process_group()