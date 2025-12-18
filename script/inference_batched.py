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
from ultra import tasks, util, data_util
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
                f"Exporting Predictions for batch {i} to {cfg['result_output_dir']}"
            )  # get train edge predictions, batch-wise to decrease mem overhead
            # use the dataset root dir to retrieve dictionaries. If it doesn't exist, the script will build it

            export_df = pl.DataFrame(
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
            ).unique()
            export_df = data_util.translate_hrt(
                df=export_df, data_path=os.path.dirname(working_dir)
            )
            export_df = data_util.filter_process_results(
                df=export_df, results_path=os.path.dirname(working_dir)
            )
            export_df = data_util.structure_results(
                export_df,
                node_path=os.path.join(data_dir, "raw", "nodes.txt"),
                graph_path=os.path.join(data_dir, "raw"),
            )
            (
                export_df.with_columns(
                    pl.int_ranges(1, pl.col("edge_in_primekg").list.len() + 1).alias(
                        "rank"
                    ),
                    pl.col("r_label").str.replace(" ", "_"),
                )
                .explode(
                    [
                        "t_pred_label",
                        "t_pred_name",
                        "t_pred_score",
                        "t_pred_type",
                        "edge_in_primekg",
                        "rank",
                    ]
                )
                .write_parquet(
                    cfg["result_output_dir"], partition_by=["h_label", "r_label"]
                )
            )
            # .write_parquet(os.path.join(working_dir, f"{label}_{i}.parquet"))

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

    # set result output dir
    if util.get_rank() == 0:
        if cfg.result_output_dir is None:
            cfg.result_output_dir = working_dir
        else:
            # if not none, check if it exists, if not create it
            if not os.path.exists(cfg["result_output_dir"]):
                os.makedirs(cfg["result_output_dir"], exist_ok=True)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    infer_data = util.inference_data_batch(dataset, cfg.infer["file"])

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
    infer_filtered_data = infer_filtered_data.to(device)

    # if util.get_rank() == 0:
    #     logger.warning(separator)
    #     logger.warning(f"Run inference on Test")

    # get dataset dir loc, need this for post inference processing
    data_dir = os.path.join(cfg.dataset["root"], dataset.name)
    # infer_res = test(
    #     cfg,
    #     model,
    #     test_data,
    #     filtered_data=test_filtered_data,
    #     device=device,
    #     logger=logger,
    #     return_metrics=False,
    #     export_results=True,
    #     top_ranks=None,
    #     label="test",
    # )

    # if util.get_rank() == 0:
    #     logger.warning(separator)
    #     logger.warning(f"Run inference on Valid")

    # infer_res = test(
    #     cfg,
    #     model,
    #     valid_data,
    #     filtered_data=val_filtered_data,
    #     device=device,
    #     logger=logger,
    #     return_metrics=False,
    #     export_results=True,
    #     top_ranks=None,
    #     label="valid",
    # )

    # if util.get_rank() == 0:
    #     logger.warning(separator)
    #     logger.warning(f"Run inference on Train")

    # infer_res = test(
    #     cfg,
    #     model,
    #     train_data,
    #     filtered_data=test_filtered_data,  # doesn't matter for train data
    #     device=device,
    #     logger=logger,
    #     return_metrics=False,
    #     export_results=True,
    #     top_ranks=None,
    #     label="train",
    # )
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning(f'Run inference on {cfg.infer["file"]}')

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
