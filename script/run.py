import os
import sys
import math
import pprint
from itertools import islice
import polars as pl

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra


separator = ">" * 30
line = "-" * 30


def train_and_validate(
    cfg,
    model,
    train_data,
    valid_data,
    device,
    logger,
    filtered_data=None,
    batch_per_epoch=None,
    tracker=None,
):
    if cfg.train.num_epoch == 0:
        return
    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat(
        [train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]
    ).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(
        train_triplets, cfg.train.batch_size, sampler=sampler
    )

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
                batch = tasks.negative_sampling(
                    train_data,
                    batch,
                    cfg.task.num_negative,
                    strict=cfg.task.strict_negative,
                )
                pred = parallel_model(train_data, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(
                    pred, target, reduction="none"
                )
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(
                            pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1
                        )
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)
                # if tracker is not None:
                #     writer.add_scalar('Train Loss', avg_loss, epoch)
                #     writer.flush()

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")

        results = test(
            cfg,
            model,
            valid_data,
            filtered_data=filtered_data,
            return_metrics=True,
            device=device,
            logger=logger,
        )

        # result = results['mrr'].item()

        if rank == 0:
            result = results["mrr"].item()
            # add tracker for metrics
            if tracker is not None:
                writer.add_scalar(
                    tag="Train/Loss", scalar_value=avg_loss, global_step=epoch
                )  # loss
                writer.add_scalar(
                    tag="Valid/MRR", scalar_value=result, global_step=epoch
                )  # mrr
                writer.add_scalar(
                    tag="Valid/hits@1",
                    scalar_value=results["hits@1"].item(),
                    global_step=epoch,
                )  # hits at 1, 3, 10
                writer.add_scalar(
                    tag="Valid/hits@3",
                    scalar_value=results["hits@3"].item(),
                    global_step=epoch,
                )  # hits at 1, 3, 10
                writer.add_scalar(
                    tag="Valid/hits@10",
                    scalar_value=results["hits@10"].item(),
                    global_step=epoch,
                )  # hits at 1, 3, 10
                writer.flush()

            if result > best_result:
                best_result = result
                best_epoch = epoch
    if rank == 0:
        logger.warning(f"Load checkpoint from model_epoch_{best_epoch}.pth")
        state = torch.load(f"model_epoch_{best_epoch}.pth", map_location=device)
        model.load_state_dict(state["model"])
        # if result > best_result:
        #     best_result = result
        #     best_epoch = epoch

    # if rank == 0:
    #     logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    # state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    # model.load_state_dict(state["model"])
    util.synchronize()


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
    tail_rankings, num_tail_negs = (
        [],
        [],
    )  # for explicit tail-only evaluation needed for 5 datasets
    head_rankings = []
    batch_ls, h_predictions, t_predictions = (
        [],
        [],
        [],
    )  # added line to store predictions

    for batch_id, batch in enumerate(test_loader):
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)

        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        head_rankings += [h_ranking]
        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]

        # batch export of predictions results
        if export_results:

            pl.DataFrame(
                {
                    "h": batch[:, 0].tolist(),
                    "r": batch[:, 2].tolist(),
                    "t": batch[:, 1].tolist(),
                    "h_pred": tasks.get_predictions(h_pred, h_mask, 100).tolist(),
                    "t_pred": tasks.get_predictions(t_pred, t_mask, 100).tolist(),
                    "h_rank": h_ranking.tolist(),
                    "t_rank": t_ranking.tolist(),
                }
            ).write_parquet(
                os.path.join(working_dir, f"{label}_rank{rank}_{batch_id}.parquet")
            )
            # batch_ls += batch
            # t_predictions += tasks.get_predictions(t_pred, t_mask, 100)
            # h_predictions += tasks.get_predictions(h_pred, h_mask, 100)

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

    # ONLY IF YOU WANT TO EXPORT ALL RESULTS AT ONCE. OTHERWISE CONCATENATE THE DATAFRAMES THAT HAVE BEEN EXPORTED #
    # if export_results:
    #     # stack predictions
    #     batches = torch.stack(batch_ls)
    #     h_predictions = torch.stack(h_predictions)
    #     t_predictions = torch.stack(t_predictions)
    #     h_rank = torch.cat(head_rankings)
    #     t_rank = torch.cat(tail_rankings)

    #     del batch_ls, head_rankings, tail_rankings
    #     torch.cuda.empty_cache()

    #     # gather and stack the stacked predictions
    #     if world_size >1:
    #         # create placeholder tensors
    #         batches_ = [torch.ones(batches.size(), dtype=torch.long, device=device) for _ in range(world_size)]
    #         h_predictions_ = [torch.ones(h_predictions.size(), dtype=torch.long, device=device) for _ in range(world_size)]
    #         t_predictions_ = [torch.ones(t_predictions.size(), dtype=torch.long, device=device) for _ in range(world_size)]
    #         h_rank_ = [torch.ones(h_rank.size(), dtype=torch.long, device=device) for _ in range(world_size)]
    #         t_rank_ = [torch.ones(t_rank.size(), dtype=torch.long, device=device) for _ in range(world_size)]

    #         # gather tensors from rank into placeholders
    #         dist.all_gather(batches_, batches)
    #         dist.all_gather(h_predictions_, h_predictions)
    #         dist.all_gather(t_predictions_, t_predictions)
    #         dist.all_gather(h_rank_, h_rank)
    #         dist.all_gather(t_rank_, t_rank)

    #         # stack the gathered tensors
    #         batches = torch.cat(batches_)
    #         h_predictions = torch.cat(h_predictions_)
    #         t_predictions = torch.cat(t_predictions_)
    #         h_rank = torch.cat(h_rank_)
    #         t_rank = torch.cat(t_rank_)
    #         del batches_, h_predictions_, t_predictions_, h_rank_, t_rank_
    #         torch.cuda.empty_cache()

    # # export predictions
    # if export_results & (rank==0):
    #         pl.DataFrame({
    #             'h': batches[:, 0].tolist(),
    #             'r': batches[:, 2].tolist(),
    #             't': batches[:, 1].tolist(),
    #             'h_pred': h_predictions.tolist(),
    #             't_pred': t_predictions.tolist(),
    #             'h_rank': h_rank.tolist(),
    #             't_rank': t_rank.tolist(),
    #         }).write_parquet(os.path.join(working_dir, f'{label}.parquet'))

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

    metrics = {}
    if rank == 0:
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking
                _num_neg = all_num_negative
                _metric_name = metric

            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = (
                            math.factorial(num_sample - 1)
                            / math.factorial(i)
                            / math.factorial(num_sample - i - 1)
                        )
                        score += (
                            num_comb
                            * (fp_rate**i)
                            * ((1 - fp_rate) ** (num_sample - i - 1))
                        )
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score.detach().cpu()
    mrr = (1 / all_ranking.float()).mean().detach().cpu()

    return mrr if not return_metrics else metrics


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    if cfg.train.tensorboard is not None:
        writer = SummaryWriter(cfg.train.tensorboard)

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)

    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    # model = pyg.compile(model, dynamic=True)
    model = model.to(device)

    if task_name == "InductiveInference":
        # filtering for inductive datasets
        # Grail, MTDEA, HM datasets have validation sets based off the training graph
        # ILPC, Ingram have validation sets from the inference graph
        # filtering dataset should contain all true edges (base graph + (valid) + test)
        if "ILPC" in cfg.dataset["class"] or "Ingram" in cfg.dataset["class"]:
            # add inference, valid, test as the validation and test filtering graphs
            full_inference_edges = torch.cat(
                [
                    valid_data.edge_index,
                    valid_data.target_edge_index,
                    test_data.target_edge_index,
                ],
                dim=1,
            )
            full_inference_etypes = torch.cat(
                [
                    valid_data.edge_type,
                    valid_data.target_edge_type,
                    test_data.target_edge_type,
                ]
            )
            test_filtered_data = Data(
                edge_index=full_inference_edges,
                edge_type=full_inference_etypes,
                num_nodes=test_data.num_nodes,
            )
            val_filtered_data = test_filtered_data
        else:
            # test filtering graph: inference edges + test edges
            full_inference_edges = torch.cat(
                [test_data.edge_index, test_data.target_edge_index], dim=1
            )
            full_inference_etypes = torch.cat(
                [test_data.edge_type, test_data.target_edge_type]
            )
            test_filtered_data = Data(
                edge_index=full_inference_edges,
                edge_type=full_inference_etypes,
                num_nodes=test_data.num_nodes,
            )

            # validation filtering graph: train edges + validation edges
            val_filtered_data = Data(
                edge_index=torch.cat(
                    [train_data.edge_index, valid_data.target_edge_index], dim=1
                ),
                edge_type=torch.cat(
                    [train_data.edge_type, valid_data.target_edge_type]
                ),
            )
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(
            edge_index=dataset._data.target_edge_index,
            edge_type=dataset._data.target_edge_type,
            num_nodes=dataset[0].num_nodes,
        )
        val_filtered_data = test_filtered_data = filtered_data

    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)
    # won't run if epoch == 0, selects best model to out for final valid/test scores
    train_and_validate(
        cfg,
        model,
        train_data,
        valid_data,
        filtered_data=val_filtered_data,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
        tracker=cfg.train.tensorboard,
        logger=logger,
    )
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    valid_res = test(
        cfg,
        model,
        valid_data,
        filtered_data=val_filtered_data,
        device=device,
        logger=logger,
        return_metrics=True,
        export_results=True,
        label="valid",
    )
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test_res = test(
        cfg,
        model,
        test_data,
        filtered_data=test_filtered_data,
        device=device,
        logger=logger,
        return_metrics=True,
        export_results=True,
        label="test",
    )

    # if (cfg.train.tensorboard is not None) and util.get_rank()==0:

    # writer.add_histogram('best_model/valid/mrr',valid_res['mrr'].item(),0)
    # writer.add_histogram('best_model/valid/hits@1',valid_res['hits@1'].item(),0)
    # writer.add_histogram('best_model/valid/hits@3',valid_res['hits@3'].item(),0)
    # writer.add_histogram('best_model/valid/hits@10',valid_res['hits@10'].item(),0)

    # writer.add_histogram('best_model/test/mrr',test_res['mrr'].item(),0)
    # writer.add_histogram('best_model/test/hits@1',test_res['hits@1'].item(),0)
    # writer.add_histogram('best_model/test/hits@3',test_res['hits@3'].item(),0)
    # writer.add_histogram('best_model/test/hits@10',test_res['hits@10'].item(),0)
    # writer.flush()
    # writer.close()
    # if util.get_rank() ==0:
    #     logger.warning(separator)
    #     logger.warning("Evaluate on train") # get train edge predictions
    # test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger, return_metrics = True, export_results=True, label = 'train')
