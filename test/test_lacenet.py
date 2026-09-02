import argparse
import os
import time
import warnings
from collections import defaultdict

import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import LACENet
from model.ema import load_ckpt_for_resume
from utils.lmdb_dataset import TrajectoryLmdbDataset, collate_fn
from utils.utils import compute_cart_mean_absolute_displacement_wrap_pred, create_dir

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LACENet.")
    parser.add_argument("--dataset", type=str, default="custom", help="dataset name for output filename")
    parser.add_argument("--data_root", type=str, required=True, help="dataset root directory")
    parser.add_argument("--model_path", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--test_subdir", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument(
        "--save_structures",
        action="store_true",
        help="whether to save predicted relaxed CIF structures"
    )

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
        choices=["reference_gated", "reference_masked", "gated", "masked", "mp1", "mp2","first_order", "second_order", "first_second"],
    )
    parser.add_argument("--max_ell", type=int, default=3)
    return parser.parse_args()


def build_model(args):
    return LACENet(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        num_elements=args.num_elements,
        model_variant=args.model_variant,
        message_passing=args.message_passing,
        max_ell=args.max_ell,
    )


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        
    if args.save_structures:
        pred_dir = "./predicted_structures/" + args.dataset
        create_dir([pred_dir])

    test_set = TrajectoryLmdbDataset({"src": os.path.join(args.data_root, args.test_subdir)})
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = build_model(args).to(device)
    ema = load_ckpt_for_resume(args.model_path, model, device)
    model = model.to(device)
    model.eval()

    performance_dict = defaultdict(list)
    matcher = StructureMatcher()
    progress = tqdm(test_loader, desc="Testing", leave=True, disable=args.disable_tqdm)

    for data in progress:
        with torch.no_grad():
            data = data.to(device)
            start_time = time.time()
            pos_pred_list, cell_pred_list = ema(data)
            pos_pred, cell_pred = pos_pred_list[-1], cell_pred_list[-1]

            if cell_pred.dim() == 2:
                cell_pred = cell_pred.unsqueeze(0)

            elapsed_time = time.time() - start_time
            pos_r = data.pos_r
            pos_u = data.pos_u
            cell_u = data.cell_u
            cell_r = data.cell_r

            if cell_u.dim() == 2:
                cell_u = cell_u.unsqueeze(0)
            if cell_r.dim() == 2:
                cell_r = cell_r.unsqueeze(0)

            batch_indices = data.batch
            unique_graphs = batch_indices.unique(sorted=True)

            for graph_idx in unique_graphs.cpu().tolist():
                mask = batch_indices == graph_idx
                if mask.sum().item() == 0:
                    continue

                single_batch = torch.zeros_like(batch_indices[mask])
                single_cell_r = cell_r[graph_idx].unsqueeze(0)

                mae_pos_dummy = compute_cart_mean_absolute_displacement_wrap_pred(
                    pos_u[mask],
                    pos_r[mask],
                    single_cell_r,
                    single_batch,
                ).item()
                mae_pos_lacenet = compute_cart_mean_absolute_displacement_wrap_pred(
                    pos_pred[mask],
                    pos_r[mask],
                    single_cell_r,
                    single_batch,
                ).item()

                cell_u_graph = cell_u[graph_idx]
                cell_r_graph = cell_r[graph_idx]
                cell_pred_graph = cell_pred[graph_idx]

                metric_tensor_unrelaxed = cell_u_graph @ cell_u_graph.T
                metric_tensor_relaxed = cell_r_graph @ cell_r_graph.T
                metric_tensor_predicted = cell_pred_graph @ cell_pred_graph.T

                mae_metric_tensor_dummy = torch.norm(
                    metric_tensor_unrelaxed - metric_tensor_relaxed,
                    p="fro",
                ).item()
                mae_metric_tensor_lacenet = torch.norm(
                    metric_tensor_predicted - metric_tensor_relaxed,
                    p="fro",
                ).item()

                mae_volume_dummy = (
                    torch.linalg.det(cell_u_graph) - torch.linalg.det(cell_r_graph)
                ).abs().item()
                mae_volume_lacenet = (
                    torch.linalg.det(cell_pred_graph) - torch.linalg.det(cell_r_graph)
                ).abs().item()

                atomic_numbers = getattr(data, "atomic_numbers", None)
                if atomic_numbers is None:
                    atomic_numbers = getattr(data, "x", None)

                structure_match_unrelaxed_gt = False
                structure_match_pred_gt = False
                if atomic_numbers is not None:
                    try:
                        species_graph = atomic_numbers[mask].detach().cpu().long().view(-1).tolist()
                        structure_gt = Structure(
                            lattice=cell_r_graph.detach().cpu().numpy(),
                            species=species_graph,
                            coords=pos_r[mask].detach().cpu().numpy(),
                            coords_are_cartesian=True,
                        )
                        structure_pred = Structure(
                            lattice=cell_pred_graph.detach().cpu().numpy(),
                            species=species_graph,
                            coords=pos_pred[mask].detach().cpu().numpy(),
                            coords_are_cartesian=True,
                        )
                        structure_unrelaxed = Structure(
                            lattice=cell_u_graph.detach().cpu().numpy(),
                            species=species_graph,
                            coords=pos_u[mask].detach().cpu().numpy(),
                            coords_are_cartesian=True,
                        )
                        structure_match_unrelaxed_gt = matcher.fit(structure_unrelaxed, structure_gt)
                        structure_match_pred_gt = matcher.fit(structure_pred, structure_gt)
                        
                        if args.save_structures:
                            save_path = os.path.join(
                                pred_dir,
                                data.cif_id[graph_idx] + "_predicted.cif"
                            )

                            structure_pred.to(
                                filename=save_path,
                                fmt="cif"
                            )
                    except Exception:
                        structure_match_unrelaxed_gt = None
                        structure_match_pred_gt = None

                performance_dict["cif_id"].append(data.cif_id[graph_idx])
                performance_dict["mae_pos_Dummy"].append(mae_pos_dummy)
                performance_dict["mae_pos_LACENet"].append(mae_pos_lacenet)
                performance_dict["mae_metric_tensor_Dummy"].append(mae_metric_tensor_dummy)
                performance_dict["mae_metric_tensor_LACENet"].append(mae_metric_tensor_lacenet)
                performance_dict["mae_volume_Dummy"].append(mae_volume_dummy)
                performance_dict["mae_volume_LACENet"].append(mae_volume_lacenet)
                performance_dict["structure_match_unrelaxed_gt"].append(structure_match_unrelaxed_gt)
                performance_dict["structure_match_pred_gt"].append(structure_match_pred_gt)
                performance_dict["elapsed_time"].append(elapsed_time)
            progress.set_postfix(graphs=len(performance_dict["cif_id"]))

    create_dir([args.results_dir])
    
    performance_df = pd.DataFrame(performance_dict)
    result_path = os.path.join(args.results_dir, f"{args.dataset}_LACENet.csv")
    performance_df.to_csv(result_path, index=False)

    avg_metrics = {
        "avg_mae_pos_Dummy": performance_df["mae_pos_Dummy"].mean(),
        "avg_mae_pos_LACENet": performance_df["mae_pos_LACENet"].mean(),
        "avg_mae_metric_tensor_Dummy": performance_df["mae_metric_tensor_Dummy"].mean(),
        "avg_mae_metric_tensor_LACENet": performance_df["mae_metric_tensor_LACENet"].mean(),
        "avg_mae_volume_Dummy": performance_df["mae_volume_Dummy"].mean(),
        "avg_mae_volume_LACENet": performance_df["mae_volume_LACENet"].mean(),
        "avg_elapsed_time": performance_df["elapsed_time"].mean(),
    }

    unrelaxed_match = pd.to_numeric(performance_df["structure_match_unrelaxed_gt"], errors="coerce")
    pred_match = pd.to_numeric(performance_df["structure_match_pred_gt"], errors="coerce")
    avg_metrics["avg_structure_match_unrelaxed_gt"] = unrelaxed_match.mean()
    avg_metrics["avg_structure_match_pred_gt"] = pred_match.mean()

    print("\n===== Final Average Performance =====")
    for key, value in avg_metrics.items():
        if pd.isna(value):
            print(f"{key}: NaN")
        else:
            print(f"{key}: {value:.6f}")
    print(f"Saved result CSV: {result_path}")


if __name__ == "__main__":
    main()
