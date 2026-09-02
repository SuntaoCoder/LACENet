import copy
import math

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from model.block import LatticeBlock, MessagePassing, SelfInteractionBlock, StructureUpdating
from utils.graph_utils import AtomEmbedding, RadialBasis, vector_norm


MODEL_VARIANTS = {
    "z": "Z",
    "o": "O",
}


def normalize_model_variant(model_variant):
    key = str(model_variant).lower().lstrip("-")
    if key not in MODEL_VARIANTS:
        valid = ", ".join(MODEL_VARIANTS.values())
        raise ValueError(f"Unknown model_variant '{model_variant}'. Valid values: {valid}")
    return MODEL_VARIANTS[key]


class LACENet(nn.Module):
    """Lattice-aware crystal equilibration network.

    The former Z/O branches are now runtime variants:
    - Z: lattice graph -> lattice-to-atom cross update -> atom graph -> atom-to-lattice.
    - O: lattice-to-atom cross update -> lattice graph -> atom graph -> atom-to-lattice.
    """

    def __init__(
        self,
        hidden_channels=512,
        num_layers=3,
        num_rbf=128,
        cutoff=6.0,
        rbf=None,
        envelope=None,
        num_elements=83,
        return_cell=True,
        model_variant="Z",
        message_passing="reference_gated",
        max_ell=3,
    ):
        super().__init__()

        rbf = {"name": "gaussian"} if rbf is None else rbf
        envelope = {"name": "polynomial", "exponent": 5} if envelope is None else envelope
        model_variant = normalize_model_variant(model_variant)

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.rbf_config = rbf
        self.envelope_config = envelope
        self.num_elements = num_elements
        self.return_cell = return_cell
        self.model_variant = model_variant
        self.message_passing = message_passing
        self.max_ell = max_ell

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        self.lattice_scalar_emb = nn.Embedding(3, hidden_channels)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_passing_layers = nn.ModuleList()
        self.message_update_layers = nn.ModuleList()
        self.structure_update_layers = nn.ModuleList()
        self.lattice_blocks = nn.ModuleList()
        self.lattice_feat_mlps = nn.ModuleList()

        for _ in range(num_layers):
            self.message_passing_layers.append(
                MessagePassing(
                    hidden_channels,
                    num_rbf,
                    max_ell=max_ell,
                    variant=message_passing,
                )
            )
            self.message_update_layers.append(SelfInteractionBlock(hidden_channels))
            self.structure_update_layers.append(StructureUpdating(hidden_channels))

            self.lattice_blocks.append(
                LatticeBlock(hidden_channels, hidden_channels, num_rbf, cutoff)
            )
            self.lattice_feat_mlps.append(
                nn.Sequential(
                    nn.Linear(num_rbf * 2, num_rbf),
                    nn.SiLU(),
                    nn.Linear(num_rbf, num_rbf),
                )
            )

        self.inv_sqrt_2 = 1.0 / math.sqrt(2.0)

    def forward(self, data):
        pos = getattr(data, "pos_u", None)
        cell = getattr(data, "cell_u", None)
        z = getattr(data, "x", None)
        if pos is None:
            pos = data.pos
        if cell is None:
            cell = data.cell
        if z is None:
            z = data.atomic_numbers
        z = z.long()

        cell_offsets = data.cell_offsets.float()
        edge_index = data.edge_index
        neighbors = data.neighbors
        batch = data.batch

        atom_scalar = self.atom_emb(z)
        atom_vector = torch.zeros(
            atom_scalar.size(0),
            3,
            atom_scalar.size(1),
            device=atom_scalar.device,
            dtype=atom_scalar.dtype,
        )

        num_graphs = batch[-1].item() + 1
        
        axis_ids = torch.arange(3, device=edge_index.device, dtype=torch.long)
        scalar_base = self.lattice_scalar_emb(axis_ids)
        lattice_scalar_state = scalar_base.unsqueeze(0).expand(num_graphs, -1, -1).contiguous()
        
        lattice_vector_state = torch.zeros(
            num_graphs,
            3,
            3,
            atom_scalar.size(1),
            device=atom_scalar.device,
            dtype=atom_scalar.dtype,
        )

        pos_list = []
        cell_list = []

        for layer_idx in range(self.num_layers):
            edge_feat, edge_udiff, lattice_feat, lattice_udiff = self._build_layer_geometry(
                pos,
                cell,
                cell_offsets,
                edge_index,
                neighbors,
                batch,
                layer_idx,
            )

            lattice_block = self.lattice_blocks[layer_idx]

            if self.model_variant == "O":
                atom_scalar, lattice_scalar_state, atom_vector, lattice_vector_state = lattice_block.distribute_to_atoms(
                    atom_scalar,
                    lattice_scalar_state,
                    atom_vector,
                    lattice_vector_state,
                    lattice_feat,
                    lattice_udiff,
                    batch,
                )
                lattice_scalar_graph, lattice_vector_graph = lattice_block.propagate_lattice_graph(
                    lattice_scalar_state,
                    lattice_vector_state,
                    cell,
                )
                
                atom_scalar, atom_vector = self._atom_graph_update(
                    layer_idx,
                    atom_scalar,
                    atom_vector,
                    edge_index,
                    edge_feat,
                    edge_udiff,
                )
                
            elif self.model_variant == "Z":
                lattice_scalar_graph, lattice_vector_graph = lattice_block.propagate_lattice_graph(
                    lattice_scalar_state,
                    lattice_vector_state,
                    cell,
                )
                
                atom_scalar, lattice_scalar_graph, atom_vector, lattice_vector_graph = lattice_block.distribute_to_atoms(
                    atom_scalar,
                    lattice_scalar_graph,
                    atom_vector,
                    lattice_vector_graph,
                    lattice_feat,
                    lattice_udiff,
                    batch,
                )
                atom_scalar, atom_vector = self._atom_graph_update(
                    layer_idx,
                    atom_scalar,
                    atom_vector,
                    edge_index,
                    edge_feat,
                    edge_udiff,
                )

            lattice_scalar_state, lattice_vector_state = lattice_block.gather_from_atoms(
                atom_scalar,
                lattice_scalar_graph,
                atom_vector,
                lattice_vector_graph,
                batch,
            )

            pos_delta = self.structure_update_layers[layer_idx](atom_vector)
            cell_delta = lattice_block.decode_lattice_delta(lattice_vector_state)

            pos = pos + pos_delta
            cell = cell + cell_delta

            pos_list.append(pos)
            cell_list.append(cell)

        if self.return_cell:
            return pos_list, cell_list
        return pos_list

    def _atom_graph_update(self, layer_idx, atom_scalar, atom_vector, edge_index, edge_feat, edge_udiff):
        delta_scalar, delta_vector = self.message_passing_layers[layer_idx](
            atom_scalar,
            atom_vector,
            edge_index,
            edge_feat,
            edge_udiff,
        )
        atom_scalar = (atom_scalar + delta_scalar) * self.inv_sqrt_2
        atom_vector = atom_vector + delta_vector

        delta_scalar, delta_vector = self.message_update_layers[layer_idx](
            atom_scalar,
            atom_vector,
        )
        atom_scalar = (atom_scalar + delta_scalar) * self.inv_sqrt_2
        atom_vector = atom_vector + delta_vector
        return atom_scalar, atom_vector

    def _build_layer_geometry(self, pos, cell, cell_offsets, edge_index, neighbors, batch, layer_idx):
        j, i = edge_index
        cell_per_edge = cell.repeat_interleave(neighbors, dim=0)
        pos_diff = pos[j] + torch.einsum("ap,apv->av", cell_offsets, cell_per_edge) - pos[i]
        edge_dist = vector_norm(pos_diff, dim=-1)
        edge_udiff = -pos_diff / (edge_dist.unsqueeze(-1) + 1e-9)
        edge_udiff = torch.nan_to_num(edge_udiff)
        edge_feat = self.radial_basis(edge_dist)

        pos_mean = global_mean_pool(pos, batch)
        pos_centered = pos - pos_mean[batch]
        cell_batch = cell[batch]
        cell_norm = vector_norm(cell_batch, dim=-1)
        pos_expand = pos_centered.unsqueeze(1).expand(-1, 3, -1)

        d_para = (pos_expand * cell_batch).sum(dim=-1) / (cell_norm + 1e-9)
        cross = torch.linalg.cross(pos_expand, cell_batch, dim=-1)
        d_perp = vector_norm(cross, dim=-1) / (cell_norm + 1e-9)
        feat_para = self.radial_basis(d_para.reshape(-1)).view(pos.size(0), 3, -1)
        feat_perp = self.radial_basis(d_perp.reshape(-1)).view(pos.size(0), 3, -1)
        lattice_feat = torch.cat([feat_para, feat_perp], dim=-1)
        lattice_feat = self.lattice_feat_mlps[layer_idx](lattice_feat)

        lattice_diff = cell_batch - pos_expand
        lattice_dist = vector_norm(lattice_diff, dim=-1)
        lattice_udiff = -lattice_diff / (lattice_dist.unsqueeze(-1) + 1e-9)
        lattice_udiff = torch.nan_to_num(lattice_udiff)
        return edge_feat, edge_udiff, lattice_feat, lattice_udiff


    def __deepcopy__(self, memo):
        copied = self.__class__(
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            num_rbf=self.num_rbf,
            cutoff=self.cutoff,
            rbf=copy.deepcopy(self.rbf_config, memo),
            envelope=copy.deepcopy(self.envelope_config, memo),
            num_elements=self.num_elements,
            return_cell=self.return_cell,
            model_variant=self.model_variant,
            message_passing=self.message_passing,
            max_ell=self.max_ell,
        )
        copied.load_state_dict(copy.deepcopy(self.state_dict(), memo))
        return copied
