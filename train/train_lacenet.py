import argparse
import os
import time
import warnings
from collections import defaultdict

import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import LACENet
from model.ema import ModelEMA, load_training_state, save_ckpt
from utils.lmdb_dataset import TrajectoryLmdbDataset, collate_fn
from utils.utils import (
    AverageMeter,
    BestMeter,
    compute_cart_mean_absolute_displacement_wrap_pred,
    create_dir,
)

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def append_text_log(log_path, msg):
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def print_and_log(msg, log_path=None):
    print(msg)
    append_text_log(log_path, msg)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LACENet.")
    parser.add_argument("--data_root", type=str, required=True, help="data directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default="xmno",
        choices=["xmno", "mp", "c2db", "vdW", "jarvis"],
        help="dataset to train on",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_norm", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--steps_per_epoch", type=int, default=800)
    parser.add_argument("--early_stop_epoch", type=int, default=10)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--cell_loss_weight", type=float, default=1.0)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--log_txt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--disable_tqdm", action="store_true")

    parser.add_argument("--hidden_channels", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_rbf", type=int, default=128)
    parser.add_argument("--cutoff", type=float, default=30.0)
    parser.add_argument("--num_elements", type=int, default=118)
    parser.add_argument("--model_variant", type=str, default="Z", choices=["Z", "O"])
    parser.add_argument(
        "--message_passing",
        type=str,
        default="reference_gated",
        choices=["reference_gated", "reference_masked", "gated", "masked", "mp1", "mp2"],
    )
    parser.add_argument("--max_ell", type=int, default=3)
    
    return parser.parse_args()


def build_model(args, return_cell=True):
    return LACENet(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        num_elements=args.num_elements,
        return_cell=return_cell,
        model_variant=args.model_variant,
        message_passing=args.message_passing,
        max_ell=args.max_ell,
    )


def val(model, dataloader, device, use_cell_loss, cell_loss_weight, disable_tqdm=False):
    model.eval()

    running_loss = AverageMeter()
    running_loss_pos = AverageMeter()
    running_loss_cell = AverageMeter() if use_cell_loss else None
    pred_quantity_dict = defaultdict(list) if use_cell_loss else None
    total_pos_abs = 0.0
    total_pos_count = 0

    progress = tqdm(dataloader, desc="Validating", leave=False, disable=disable_tqdm)
    for data in progress:
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
            if use_cell_loss:
                pos_pred_list, cell_pred_list = outputs
            else:
                pos_pred_list = outputs
                cell_pred_list = None

            pos_label = getattr(data, "pos_r", None)
            cell_label = getattr(data, "cell_r", None)
            if pos_label is None:
                pos_label = data.pos_relaxed
            if cell_label is None:
                cell_label = data.cell

            loss_pos = 0
            loss_cell = 0 if use_cell_loss else None
            for layer_idx, pos_pred in enumerate(pos_pred_list):
                loss_pos = loss_pos + compute_cart_mean_absolute_displacement_wrap_pred(
                    pos_pred,
                    pos_label,
                    cell_label,
                    data.batch,
                )
                if use_cell_loss:
                    cell_pred = cell_pred_list[layer_idx]
                    loss_cell = loss_cell + (cell_pred - cell_label).abs().mean()
            loss = loss_pos
            if use_cell_loss:
                loss = loss + cell_loss_weight * loss_cell
                pred_quantity_dict["cell_label"].append(cell_label)
                pred_quantity_dict["cell_pred"].append(cell_pred_list[-1])

            batch_pos_mae = compute_cart_mean_absolute_displacement_wrap_pred(
                pos_pred_list[-1],
                pos_label,
                cell_label,
                data.batch,
            )
            total_pos_abs += batch_pos_mae.item() * pos_label.numel()
            total_pos_count += pos_label.numel()

            running_loss.update(loss.item())
            running_loss_pos.update(loss_pos.item())
            if running_loss_cell is not None:
                running_loss_cell.update(loss_cell.item())
            progress.set_postfix(loss=f"{running_loss.get_average():.4f}", pos=f"{running_loss_pos.get_average():.4f}")

    if use_cell_loss and pred_quantity_dict["cell_label"]:
        cell_label = torch.cat(pred_quantity_dict["cell_label"], dim=0)
        cell_pred = torch.cat(pred_quantity_dict["cell_pred"], dim=0)
        valid_cell_mae = (cell_label - cell_pred).abs().mean().item()
    else:
        valid_cell_mae = None

    model.train()
    return {
        "loss": running_loss.get_average(),
        "loss_pos": running_loss_pos.get_average(),
        "loss_cell": running_loss_cell.get_average() if running_loss_cell is not None else None,
        "pos_mae": total_pos_abs / (total_pos_count + 1e-12),
        "cell_mae": valid_cell_mae,
    }


def main():
    args = parse_args()
    train_set = TrajectoryLmdbDataset({"src": os.path.join(args.data_root, "train")})
    valid_set = TrajectoryLmdbDataset({"src": os.path.join(args.data_root, "val")})
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f"LACENet_{args.dataset}_{args.model_variant}_{args.message_passing}_{timestamp}"
    wandb.init(
        project="LACENet",
        group=args.dataset,
        config=vars(args) | {"train_len": len(train_set), "valid_len": len(valid_set)},
        name=log_name,
        id=log_name,
    )

    checkpoint_dir = args.checkpoint_dir or wandb.run.dir
    create_dir([checkpoint_dir])
    log_txt = args.log_txt or os.path.join(checkpoint_dir, "validation_log.txt")
    append_text_log(log_txt, f"\n===== Training run {log_name} started at {timestamp} =====")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print_and_log("CUDA is not available; falling back to CPU.", log_txt)
    else:
        device = torch.device(args.device)

    use_cell_loss = args.cell_loss_weight > 0
    model = build_model(args, return_cell=use_cell_loss).to(device)
    ema = ModelEMA(model, decay=0.999, warmup_steps=args.steps_per_epoch * 5, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5, min_lr=1.0e-8)

    running_loss = AverageMeter()
    running_loss_pos = AverageMeter()
    running_loss_cell = AverageMeter() if use_cell_loss else None
    running_grad_norm = AverageMeter()
    running_best_mae = BestMeter("min")
    global_step = 0
    global_epoch = 0

    if args.resume_path is not None:
        ema, resume_ckpt = load_training_state(
            args.resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            ema_decay=0.999,
            warmup_steps=args.steps_per_epoch * 5,
        )
        global_step = int(resume_ckpt.get("step", 0))
        global_epoch = int(resume_ckpt.get("epoch", 0))
        best_mae = resume_ckpt.get("best_mae", None)
        if best_mae is not None:
            running_best_mae.update(best_mae)
            running_best_mae.count = int(resume_ckpt.get("best_counter", 0))
        print_and_log(
            f"Resumed from {args.resume_path} at epoch {global_epoch}, step {global_step}, best_mae {running_best_mae.get_best():.6f}",
            log_txt,
        )

    def save_training_checkpoint(filename):
        path = os.path.join(checkpoint_dir, filename)
        save_ckpt(
            path,
            ema,
            model,
            optimizer,
            scheduler=scheduler,
            epoch=global_epoch,
            step=global_step,
            best_mae=running_best_mae.get_best(),
            best_counter=running_best_mae.count,
        )
        return path

    total_steps = args.epochs * args.steps_per_epoch
    break_flag = False
    model.train()

    try:
        while global_step < total_steps and not break_flag:
            train_progress = tqdm(
                train_loader,
                desc=f"Training epoch {global_epoch + 1}/{args.epochs}",
                leave=False,
                disable=args.disable_tqdm,
            )
            for data in train_progress:
                if global_step >= total_steps or break_flag:
                    break

                global_step += 1
                data = data.to(device)
                outputs = model(data)
                if use_cell_loss:
                    pos_pred_list, cell_pred_list = outputs
                else:
                    pos_pred_list = outputs
                    cell_pred_list = None

                pos_label = getattr(data, "pos_r", None)
                cell_label = getattr(data, "cell_r", None)
                if pos_label is None:
                    pos_label = data.pos_relaxed
                if cell_label is None:
                    cell_label = data.cell

                loss_pos = 0
                loss_cell = 0 if use_cell_loss else None
                for layer_idx, pos_pred in enumerate(pos_pred_list):
                    loss_pos = loss_pos + compute_cart_mean_absolute_displacement_wrap_pred(
                        pos_pred,
                        pos_label,
                        cell_label,
                        data.batch,
                    )
                    if use_cell_loss:
                        cell_pred = cell_pred_list[layer_idx]
                        loss_cell = loss_cell + (cell_pred - cell_label).abs().mean()
                loss = loss_pos
                if use_cell_loss:
                    loss = loss + loss_cell

                if not torch.isfinite(loss):
                    print_and_log(f"Skipping step {global_step} due to non-finite loss", log_txt)
                    continue

                optimizer.zero_grad()
                loss.backward()
                
                # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                ema.update(model)

                running_loss.update(loss.item())
                running_loss_pos.update(loss_pos.item())
                if running_loss_cell is not None:
                    running_loss_cell.update(loss_cell.item())
                running_grad_norm.update(grad_norm.item())
                train_progress.set_postfix(
                    step=global_step,
                    loss=f"{running_loss.get_average():.4f}",
                    pos=f"{running_loss_pos.get_average():.4f}",
                    grad_norm=f"{running_grad_norm.get_average():.4f}",
                )

                if global_step % args.steps_per_epoch != 0:
                    continue

                global_epoch += 1
                train_loss = running_loss.get_average()
                train_loss_pos = running_loss_pos.get_average()
                train_loss_cell = running_loss_cell.get_average() if running_loss_cell is not None else None
                train_grad_norm = running_grad_norm.get_average()

                running_loss.reset()
                running_loss_pos.reset()
                if running_loss_cell is not None:
                    running_loss_cell.reset()
                running_grad_norm.reset()

                val_metrics = val(
                    ema,
                    valid_loader,
                    device,
                    use_cell_loss=use_cell_loss,
                    cell_loss_weight=args.cell_loss_weight,
                    disable_tqdm=args.disable_tqdm,
                )
                valid_pos_mae = val_metrics["pos_mae"]
                scheduler.step(valid_pos_mae)

                log_dict = {
                    "train/epoch": global_epoch,
                    "train/loss": train_loss,
                    "train/loss_pos": train_loss_pos,
                    "train/grad_norm": train_grad_norm,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "val/valid_loss": val_metrics["loss"],
                    "val/valid_loss_pos": val_metrics["loss_pos"],
                    "val/valid_pos_mae": val_metrics["pos_mae"],
                }
                if train_loss_cell is not None:
                    log_dict["train/loss_cell"] = train_loss_cell
                for key in ["loss_cell", "cell_mae"]:
                    value = val_metrics.get(key)
                    if value is not None:
                        log_dict[f"val/valid_{key}"] = value
                wandb.log(log_dict)

                msg = "Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Pos MAE: {:.4f}".format(
                    global_epoch,
                    args.epochs,
                    train_loss,
                    val_metrics["loss"],
                    valid_pos_mae,
                )
                if val_metrics["cell_mae"] is not None:
                    msg += ", Valid Cell MAE: {:.4f}".format(val_metrics["cell_mae"])
                print_and_log(msg, log_txt)

                if valid_pos_mae < running_best_mae.get_best():
                    running_best_mae.update(valid_pos_mae)
                    if args.save_model:
                        best_path = save_training_checkpoint("model.pt")
                        print_and_log(f"Saved best checkpoint to {best_path}", log_txt)
                elif running_best_mae.counter() > args.early_stop_epoch:
                    print_and_log(f"early stop in epoch {global_epoch}", log_txt)
                    print_and_log(f"best_mae: {running_best_mae.get_best()}", log_txt)
                    break_flag = True

                last_path = save_training_checkpoint("last.pt")
                print_and_log(f"Saved latest checkpoint to {last_path}", log_txt)
    except KeyboardInterrupt:
        interrupt_path = save_training_checkpoint("interrupt.pt")
        last_path = save_training_checkpoint("last.pt")
        print_and_log(f"Training interrupted. Saved checkpoint to {interrupt_path}", log_txt)
        print_and_log(f"Latest checkpoint also updated at {last_path}", log_txt)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
