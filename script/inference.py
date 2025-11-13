import os
import sys
import math
import pprint
import polars as pl
import torch

from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra

separator = ">" * 30
line = "-" * 30


@torch.no_grad()
def test(
    cfg,
    model,
    test_data,
    device,
    logger,
    filtered_data=None,
    return_metrics=False,
    label="valid",
    export_results=False,
    top_ranks: int = 100,
):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat(
        [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]
    ).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(
        test_triplets, cfg.train.batch_size, sampler=sampler
    )

    model.eval()
    rankings = []
    num_negatives = []
    tail_rankings, tail_unfilt_rankings, num_tail_negs = (
        [],
        [],
        [],
    )  # for explicit tail-only evaluation needed for 5 datasets

    for i, batch in enumerate(test_loader):
        # t- and h-batch are (n x 3) tensors that map every hr and rt in batch to all tails and head (nodes), respectively
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        # score all hr-tail_predictions
        t_pred = model(test_data, t_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            # extract tail and head masks so true targets are not assigned as negatives
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)

        pos_h_index, pos_t_index, pos_r_index = batch.t()
        # get the rank of the tail given (h,r). if mask is included it will be a filtered rank
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        t_unfilt_ranking = tasks.compute_ranking(t_pred, pos_t_index)

        num_t_negative = t_mask.sum(dim=-1)

        rankings += [t_ranking]
        num_negatives += [num_t_negative]
        tail_rankings += [t_ranking]
        tail_unfilt_rankings += [t_unfilt_ranking]
        num_tail_negs += [num_t_negative]

        # batch export of predictions results
        if export_results:
            if world_size > 1:
                t_predictions = tasks.get_predictions(t_pred, t_mask, top_ranks)
                t_unfilt_predictions = tasks.get_predictions(pred=t_pred, top=top_ranks)

                # create placeholders
                batch_ = [
                    torch.ones(batch.size(), dtype=torch.long, device=device)
                    for _ in range(world_size)
                ]
                t_predictions_ = [
                    torch.ones(t_predictions.size(), dtype=torch.long, device=device)
                    for _ in range(world_size)
                ]
                t_unfilt_predictions_ = [
                    torch.ones(
                        t_unfilt_predictions.size(), dtype=torch.long, device=device
                    )
                    for _ in range(world_size)
                ]
                t_rank_ = [
                    torch.ones(t_ranking.size(), dtype=torch.long, device=device)
                    for _ in range(world_size)
                ]
                t_unfilt_rank_ = [
                    torch.ones(t_unfilt_ranking.size(), dtype=torch.long, device=device)
                    for _ in range(world_size)
                ]

                # gather tensors from rank into placeholders (ends with '_')
                dist.all_gather(batch_, batch)
                dist.all_gather(t_predictions_, t_predictions.contiguous())
                dist.all_gather(
                    t_unfilt_predictions_, t_unfilt_predictions.contiguous()
                )
                dist.all_gather(t_rank_, t_ranking)
                dist.all_gather(t_unfilt_rank_, t_unfilt_ranking)

                # stack the gathered tensors
                batch = torch.cat(batch_)
                t_predictions = torch.cat(t_predictions_)
                t_unfilt_predictions = torch.cat(t_unfilt_predictions_)
                t_ranking = torch.cat(t_rank_)
                t_unfilt_ranking = torch.cat(t_unfilt_rank_)

                # add replicates of masks and prediction so we can build the dataframe
                t_pred = torch.cat([t_pred for _ in range(world_size)])
                t_mask = torch.cat([t_mask for _ in range(world_size)])

            # world size =1
            else:
                t_predictions = tasks.get_predictions(t_pred, t_mask, top_ranks)
                t_unfilt_predictions = tasks.get_predictions(pred=t_pred, top=top_ranks)

        if rank == 0:
            logger.warning(separator)
            logger.warning(
                f"Exporting Predictions for batch {i}"
            )  # get train edge predictions, batch-wise to decrease mem overhead
            pl.DataFrame(
                {
                    "h": batch[:, 0].tolist(),
                    "r": batch[:, 2].tolist(),
                    "t": batch[:, 1].tolist(),
                    "t_filt_rank": t_ranking.tolist(),
                    "t_unfilt_rank": t_unfilt_ranking.tolist(),
                    "t_pred_filt": t_predictions.tolist(),
                    "t_pred_unfilt": t_unfilt_predictions.tolist(),
                    "t_pred_score": t_pred.tolist(),
                    "t_mask": t_mask.tolist(),
                }
            ).unique().write_parquet(os.path.join(working_dir, f"{label}_{i}.parquet"))

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    # ugly repetitive code for tail-only ranks processing
    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    # obtaining all ranks
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank] : cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank] : cum_size[rank]] = num_negative

    # the same for tails-only ranks
    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank] : cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank] : cum_size_t[rank]] = (
        num_tail_neg
    )
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)


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

    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    id2ent_dict, ent2id_dict, id2rel_dict, rel2id_dict = util.get_entity_relation_dict(
        working_dir, dataset
    )  # make sure vocabulary export is the same as that used in inference
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]

    infer_data = util.inference_data_single(
        dataset, cfg.infer["h_ent"], cfg.infer["rel"]
    )

    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    infer_data = infer_data.to(device)

    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model = model.to(device)

    # Filter the predictions
    # Transductive setting we can use the whole graph for filtered ranking
    filtered_data = Data(
        edge_index=dataset._data.target_edge_index,
        edge_type=dataset._data.target_edge_type,
        num_nodes=dataset[0].num_nodes,
    )
    infer_filtered_data = val_filtered_data = test_filtered_data = filtered_data
    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)
    infer_filtered_data

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning(f'Run inference on {cfg.infer["h_ent"]} - {cfg.infer["rel"]}')

    infer_res = test(
        cfg,
        model,
        infer_data,
        filtered_data=infer_filtered_data,
        device=device,
        logger=logger,
        return_metrics=False,
        export_results=True,
        top_ranks=None,
        label="inference",
    )
    # valid_res = test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger, return_metrics = True, export_results=True, label = 'valid')
