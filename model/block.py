import math
import torch
from torch import nn
from torch_geometric.utils import scatter
from utils.graph_utils import ScaledSiLU, vector_norm
from torch_geometric.nn import global_mean_pool
from e3nn.o3 import Irreps, SphericalHarmonics


class SH_BilinearInvariants(nn.Module):
    def __init__(self, sh_irreps):
        super().__init__()
        self.sh_irreps = sh_irreps
        self.num_inv = sh_irreps.lmax + 1

    def forward(self, sh1, sh2):
        assert sh1.size(1) == self.sh_irreps.dim
        assert sh2.size(1) == self.sh_irreps.dim
        N = sh1.size(0)
        inv = sh1.new_zeros(N, self.sh_irreps.lmax + 1)

        idx = 0
        for mul, ir in self.sh_irreps:
            dim_l = mul * ir.dim
            block1 = sh1[:, idx: idx + dim_l]   # [N, mul*dim_l]
            block2 = sh2[:, idx: idx + dim_l]   # [N, mul*dim_l]
            idx += dim_l

            # true bilinear invariant (can be negative)
            I_l = (block1 * block2).mean(-1)
            inv[:, ir.l] = I_l

        return inv

class LatticeAxisBlock(nn.Module):
    """
    One lattice-basis 'axis node' that exchanges messages with atoms.
    """
    def __init__(self, in_feats, out_feats, num_rbf):
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        # atoms → axis (global aggregation)
        self.atoms_to_axis_scalar_mlp = nn.Sequential(
            nn.Linear(in_feats * 2, out_feats),
            ScaledSiLU(),
            nn.Linear(out_feats, out_feats),
            ScaledSiLU(),
        )
        self.atoms_to_axis_vector_mlp = nn.Linear(in_feats, out_feats, bias=False)

        # axis → atoms (local interaction)
        self.axis_to_atom_scalar_mlp = nn.Sequential(
            nn.Linear(in_feats * 2, out_feats),
            ScaledSiLU(),
            nn.Linear(out_feats, out_feats),
            ScaledSiLU(),
        )
        self.axis_to_atom_vector_mlp = nn.Linear(in_feats, out_feats, bias=False)

        # atom-edge mixing for local interaction
        self.axis_atom_message_proj = nn.Sequential(
            nn.Linear(in_feats, in_feats),
            ScaledSiLU(),
            nn.Linear(in_feats, in_feats * 3),
        )
        self.axis_edge_rbf_proj = nn.Linear(num_rbf, in_feats * 3)

        # update internal axis state (scalar + vector)
        self.axis_state_vector_proj = nn.Linear(out_feats, out_feats * 2, bias=False)
        self.axis_state_scalar_mlp = nn.Sequential(
            nn.Linear(out_feats * 2, out_feats),
            ScaledSiLU(),
            nn.Linear(out_feats, out_feats * 3),
        )

        # predict lattice parameter delta from axis vector state
        self.axis_lattice_delta_mlp = nn.Linear(out_feats, 1, bias=False)

        self.inv_sqrt_3 = 1.0 / math.sqrt(3.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(in_feats)


    def distribute_to_atoms(
        self,
        atom_scalar,          # atom scalar features
        axis_scalar_state,    # per-axis scalar state [G, H]
        atom_vector,          # vec
        axis_vector_state,    # per-axis vector state [G, 3, H]
        axis_edge_feat,       # lattice_feat[:, axis, :]
        axis_edge_dir,        # lattice_udiff[:, axis, :]
        batch,
    ):
        # project atom features and edge features
        atom_proj = self.axis_atom_message_proj(atom_scalar)
        edge_proj = self.axis_edge_rbf_proj(axis_edge_feat)

        m1, m2, m3 = torch.split(
            atom_proj * edge_proj * self.inv_sqrt_3,
            self.in_feats,
            dim=-1,
        )

        # update atom scalar / vector features using axis state
        atom_scalar = m3 + atom_scalar
        atom_vector = (
            m1.unsqueeze(1) * atom_vector
            + m2.unsqueeze(1) * axis_edge_dir.unsqueeze(2)
        )
        atom_vector = atom_vector * self.inv_sqrt_h

        # inject axis state into atoms
        axis_scalar_b = axis_scalar_state[batch]
        axis_vector_b = axis_vector_state[batch]

        new_atom_scalar = (
            self.axis_to_atom_scalar_mlp(torch.cat([atom_scalar, axis_scalar_b], dim=-1))
            + atom_scalar
        )
        new_atom_vector = (
            self.axis_to_atom_vector_mlp(atom_vector + axis_vector_b) + atom_vector
        )

        return new_atom_scalar, axis_scalar_state, new_atom_vector, axis_vector_state

    def collect_from_atoms(
        self,
        atom_scalar,          # atom scalar features
        axis_scalar_state,    # [G, H]
        atom_vector,          # vec
        axis_vector_state,    # [G, 3, H]
        batch,
    ):
        # aggregate over atoms to update axis scalar/vector states
        pooled_scalar = global_mean_pool(atom_scalar, batch)  # [G, H]
        pooled_vector = scatter(
            atom_vector, batch, dim=0,
            reduce="mean", dim_size=axis_vector_state.size(0)
        )  # [G, 3, H]

        delta_scalar = self.atoms_to_axis_scalar_mlp(
            torch.cat([pooled_scalar, axis_scalar_state], dim=-1)
        )
        delta_vector = self.atoms_to_axis_vector_mlp(
            pooled_vector + axis_vector_state
        )

        axis_scalar_state = axis_scalar_state + delta_scalar
        axis_vector_state = axis_vector_state + delta_vector

        # gated refinement of axis state
        v1, v2 = torch.split(
            self.axis_state_vector_proj(axis_vector_state),
            self.out_feats,
            dim=-1,
        )
        scalar_state_h = self.axis_state_scalar_mlp(
            torch.cat(
                [axis_scalar_state,
                 torch.sqrt(torch.sum(v2 ** 2, dim=-2) + 1e-8)],
                dim=-1,
            )
        )
        s1, s2, s3 = torch.split(scalar_state_h, self.out_feats, dim=-1)
        gate = torch.tanh(s3)

        axis_scalar_state = s2 + axis_scalar_state * gate
        axis_vector_state = s1.unsqueeze(1) * v1 + axis_vector_state

        return axis_scalar_state, axis_vector_state

    def decode_axis_delta(self, axis_vector_state):
        # axis_vector_state: [G, 3, H] → [G, 3, 1]
        return self.axis_lattice_delta_mlp(axis_vector_state)

class GaussianFeatureExpansion(nn.Module):
    """Gaussian expansion for heterogeneous lattice-edge scalar features."""

    def __init__(self, num_rbf, cutoff):
        super().__init__()
        self.num_rbf = num_rbf
        cutoff = float(cutoff)
        ranges = torch.tensor(
            [
                [0.0, cutoff],
                [0.0, cutoff],
                [0.0, cutoff],
                [-cutoff * cutoff, cutoff * cutoff],
                [-1.0, 1.0],
            ],
            dtype=torch.float,
        )
        centers = torch.linspace(0.0, 1.0, num_rbf, dtype=torch.float)
        centers = ranges[:, :1] + centers.unsqueeze(0) * (ranges[:, 1:] - ranges[:, :1])
        widths = (ranges[:, 1] - ranges[:, 0]).clamp_min(1e-6) / max(num_rbf - 1, 1)

        self.register_buffer("centers", centers, persistent=False)
        self.register_buffer("widths", widths, persistent=False)

    def forward(self, edge_features):
        # edge_features: [..., F]
        centers = self.centers.to(device=edge_features.device, dtype=edge_features.dtype)
        widths = self.widths.to(device=edge_features.device, dtype=edge_features.dtype)
        diff = edge_features.unsqueeze(-1) - centers
        width_shape = [1] * (diff.dim() - 2) + [widths.size(0), 1]
        expanded = torch.exp(-0.5 * (diff / widths.view(*width_shape)) ** 2)
        return expanded.flatten(start_dim=-2)

class LatticeGraphBlock(nn.Module):
    """
    PaiNN-style message passing over the fully connected graph of the three
    lattice-vector nodes.
    """

    def __init__(self, hidden_channels, num_rbf, cutoff):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.edge_expansion = GaussianFeatureExpansion(num_rbf, cutoff)
        self.edge_proj = nn.Linear(num_rbf * 5, hidden_channels * 3)
        self.scalar_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels * 3),
        )
        self.update = SelfInteractionBlock(hidden_channels)

        self.register_buffer(
            "edge_src",
            torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "edge_dst",
            torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long),
            persistent=False,
        )

        self.inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        self.inv_sqrt_3 = 1.0 / math.sqrt(3.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(hidden_channels)

    def _edge_geometry(self, cell):
        # cell: [G, 3, 3], row axis vectors are lattice-node coordinates.
        src_vec = cell[:, self.edge_src, :]
        dst_vec = cell[:, self.edge_dst, :]
        diff = dst_vec - src_vec
        diff_norm = vector_norm(diff, dim=-1)
        edge_dir = diff / (diff_norm.unsqueeze(-1) + 1e-9)
        edge_dir = torch.nan_to_num(edge_dir)

        src_norm = vector_norm(src_vec, dim=-1)
        dst_norm = vector_norm(dst_vec, dim=-1)
        dot = (src_vec * dst_vec).sum(dim=-1)
        cos = dot / (src_norm * dst_norm + 1e-9)
        cos = torch.clamp(torch.nan_to_num(cos), min=-1.0, max=1.0)

        edge_features = torch.stack([src_norm, dst_norm, diff_norm, dot, cos], dim=-1)
        return edge_features, edge_dir

    def forward(self, axis_scalar_state, axis_vector_state, cell):
        """
        axis_scalar_state: [G, 3, H]
        axis_vector_state: [G, 3, 3, H]
        cell:              [G, 3, 3]
        """
        num_graphs = axis_scalar_state.size(0)
        edge_features, edge_dir = self._edge_geometry(cell)

        edge_rbf = self.edge_expansion(edge_features)
        edge_h = self.edge_proj(edge_rbf)
        scalar_h = self.scalar_proj(axis_scalar_state[:, self.edge_src, :])

        m1, m2, m3 = torch.split(
            scalar_h * edge_h * self.inv_sqrt_3,
            self.hidden_channels,
            dim=-1,
        )

        src_vector = axis_vector_state[:, self.edge_src, :, :]
        vec_msg = (
            m1.unsqueeze(2) * src_vector
            + m2.unsqueeze(2) * edge_dir.unsqueeze(-1)
        )
        vec_msg = vec_msg * self.inv_sqrt_h

        delta_scalar = axis_scalar_state.new_zeros(num_graphs, 3, self.hidden_channels)
        delta_vector = axis_vector_state.new_zeros(num_graphs, 3, 3, self.hidden_channels)
        delta_scalar.index_add_(1, self.edge_dst, m3)
        delta_vector.index_add_(1, self.edge_dst, vec_msg)

        axis_scalar_state = (axis_scalar_state + delta_scalar) * self.inv_sqrt_2
        axis_vector_state = axis_vector_state + delta_vector

        flat_scalar = axis_scalar_state.reshape(num_graphs * 3, self.hidden_channels)
        flat_vector = axis_vector_state.reshape(num_graphs * 3, 3, self.hidden_channels)
        delta_scalar, delta_vector = self.update(flat_scalar, flat_vector)
        delta_scalar = delta_scalar.view(num_graphs, 3, self.hidden_channels)
        delta_vector = delta_vector.view(num_graphs, 3, 3, self.hidden_channels)

        axis_scalar_state = (axis_scalar_state + delta_scalar) * self.inv_sqrt_2
        axis_vector_state = axis_vector_state + delta_vector

        return axis_scalar_state, axis_vector_state

class LatticeBlock(nn.Module):
    """
    Wraps the three lattice axes and handles:
    - lattice → atoms (local interaction)
    - atoms → lattice (global aggregation)
    - lattice delta prediction
    """
    def __init__(self, in_feats, out_feats, num_rbf, cutoff=6.0):
        super().__init__()
        self.axis_blocks = nn.ModuleList(
            [LatticeAxisBlock(in_feats, out_feats, num_rbf) for _ in range(3)]
        )
        self.lattice_graph = LatticeGraphBlock(out_feats, num_rbf, cutoff)
        
        # axis–axis interaction (shared across graphs)
        self.axis_mix_scalar = nn.Linear(3, 3, bias=True)    # mix axis dimension
        self.axis_mix_vector = nn.Linear(3, 3, bias=False)   # mix axis dimension

    def propagate_lattice_graph(self, axis_scalar_state, axis_vector_state, cell):
        return self.lattice_graph(axis_scalar_state, axis_vector_state, cell)
    
    def axis_interaction(self, axis_scalar_state, axis_vector_state): 
        """
        axis_scalar_state: [G, 3, H]
        axis_vector_state: [G, 3, 3, H]
        returns mixed versions with axis–axis coupling.
        """
        # --- mix scalar state over axis dimension ---
        # reshape to [G, H, 3] → apply Linear over last dim → back
        s = axis_scalar_state.permute(0, 2, 1)              # [G, H, 3]
        s = self.axis_mix_scalar(s)                         # [G, H, 3]
        axis_scalar_state = s.permute(0, 2, 1).contiguous() # [G, 3, H]

        # --- mix vector state over axis dimension ---
        # treat (G, C, H) as batch, mix axis dimension
        v = axis_vector_state.permute(0, 2, 3, 1)           # [G, C, H, 3]
        v = self.axis_mix_vector(v)                         # [G, C, H, 3]
        axis_vector_state = v.permute(0, 3, 1, 2).contiguous()  # [G, 3, C, H]

        return axis_scalar_state, axis_vector_state

    def distribute_to_atoms(self, atom_scalar, axis_scalar_state, atom_vector,
                            axis_vector_state, lattice_feat, lattice_udiff, batch):
        updated_scalar_states = []
        updated_vector_states = []
        for axis, axis_block in enumerate(self.axis_blocks):
            s_axis = axis_scalar_state[:, axis, :]
            v_axis = axis_vector_state[:, axis, :, :]
            atom_scalar, s_axis, atom_vector, v_axis = axis_block.distribute_to_atoms(
                atom_scalar,
                s_axis,
                atom_vector,
                v_axis,
                lattice_feat[:, axis, :],
                lattice_udiff[:, axis, :],
                batch,
            )
            updated_scalar_states.append(s_axis.unsqueeze(1))
            updated_vector_states.append(v_axis.unsqueeze(1))

        axis_scalar_state = torch.cat(updated_scalar_states, dim=1)
        axis_vector_state = torch.cat(updated_vector_states, dim=1)
        axis_scalar_state, axis_vector_state = self.axis_interaction(    
                    axis_scalar_state, axis_vector_state
                )
        return atom_scalar, axis_scalar_state, atom_vector, axis_vector_state

    def gather_from_atoms(self, atom_scalar, axis_scalar_state, atom_vector,
                          axis_vector_state, batch):
        updated_scalar_states = []
        updated_vector_states = []
        for axis, axis_block in enumerate(self.axis_blocks):
            s_axis = axis_scalar_state[:, axis, :]
            v_axis = axis_vector_state[:, axis, :, :]
            s_axis, v_axis = axis_block.collect_from_atoms(
                atom_scalar, s_axis, atom_vector, v_axis, batch
            )
            updated_scalar_states.append(s_axis.unsqueeze(1))
            updated_vector_states.append(v_axis.unsqueeze(1))

        axis_scalar_state = torch.cat(updated_scalar_states, dim=1)
        axis_vector_state = torch.cat(updated_vector_states, dim=1)
        
        axis_scalar_state, axis_vector_state = self.axis_interaction(    
                    axis_scalar_state, axis_vector_state
                )
        return axis_scalar_state, axis_vector_state

    def decode_lattice_delta(self, axis_vector_state):
        # concat per-axis deltas along last dim: [G, 3, H] → [G, 3, 3]
        deltas = []
        for axis, axis_block in enumerate(self.axis_blocks):
            v_axis = axis_vector_state[:, axis, :, :]
            deltas.append(axis_block.decode_axis_delta(v_axis))  # [G, 3, 1]
        return torch.cat(deltas, dim=-1).transpose(-1, -2)  # [G, 3, 3] 


class StructureUpdating(nn.Module):
    """Maps atom-level vector embeddings into Cartesian position updates."""

    def __init__(self, hidden_channels):
        super().__init__()
        self.pos_mlp = nn.Linear(hidden_channels, 1, bias=False)
        self.delta_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, atom_vector):
        # atom_vector: [N, 3, H]
        coord_from_vec = self.pos_mlp(atom_vector).squeeze(-1)  # [N, 3]
        return self.delta_scale * coord_from_vec    

    
    
MESSAGE_PASSING_VARIANTS = {
    "reference_gated": "reference_gated",
    "gated": "reference_gated",
    "mp1": "reference_gated",
    "reference_masked": "reference_masked",
    "masked": "reference_masked",
    "mp2": "reference_masked",
}


def normalize_message_passing_variant(variant):
    variant = str(variant).lower().replace("-", "_")
    if variant not in MESSAGE_PASSING_VARIANTS:
        valid = ", ".join(sorted(set(MESSAGE_PASSING_VARIANTS.values())))
        raise ValueError(f"Unknown message_passing variant '{variant}'. Valid values: {valid}")
    return MESSAGE_PASSING_VARIANTS[variant]


class MessagePassing(nn.Module):

    def __init__(
        self,
        hidden_channels,
        edge_feat_channels,
        max_ell=6,
        variant="reference_gated",
    ):
        super(MessagePassing, self).__init__()

        self.hidden_channels = hidden_channels
        self.variant = normalize_message_passing_variant(variant)

        # scalar projections
        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels * 3),
        )
        self.edge_proj = nn.Linear(edge_feat_channels, hidden_channels * 3)

        # SH for directions
        base_irreps = Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = SphericalHarmonics(
            base_irreps, normalize=True, normalization="component"
        )
        self.sh_pair_inv = SH_BilinearInvariants(base_irreps)  # new
        self.inv_norm = nn.LayerNorm(self.sh_pair_inv.num_inv)

        # (L invariants + 1 cosθ) → 3H
        self.edge_inv_proj = nn.Sequential(
            nn.Linear(self.sh_pair_inv.num_inv + 1, hidden_channels * 3),
            ScaledSiLU(),
        )

        self.inv_sqrt_3 = 1.0 / math.sqrt(3.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(hidden_channels)

    def forward(self, node_scalar, node_vector, edge_index, edge_rbf, edge_udiff):
        """
        node_scalar: [N, H]    scalar node features
        node_vector: [N, 3, H] vector node features
        edge_rbf:    [E, R]    scalar edge features
        edge_udiff:  [E, 3]    unit edge directions (j -> i)
        """
        j, i = edge_index
        num_nodes = node_scalar.size(0)

        # ---- radial part ----
        rbf_h = self.edge_proj(edge_rbf)   # [E, 3H]

        # ---- angular part ----
        ref_vec_node = scatter(edge_udiff, index=i, dim=0,
                       dim_size=num_nodes, reduce="mean")      # [N, 3]
        
        # Safe norm calculation to avoid NaN in backward pass when norm is 0
        ref_vec_sq = ref_vec_node.pow(2).sum(dim=-1, keepdim=True)
        ref_vec_norm = torch.sqrt(ref_vec_sq + 1e-9)
        valid_ref_node = ref_vec_norm.squeeze(-1) >= 5e-5
        ref_vec_node = ref_vec_node / ref_vec_norm

        if (~valid_ref_node).any():
            # Placeholder only keeps spherical_harmonics finite. Invalid edges
            # replace this reference with edge_sh below.
            fallback_vec = torch.tensor([1.0, 0.0, 0.0], device=ref_vec_node.device, dtype=ref_vec_node.dtype)
            ref_vec_node = torch.where(valid_ref_node.unsqueeze(-1), ref_vec_node, fallback_vec)

        edge_sh = self.spherical_harmonics(edge_udiff)     # [E, sh_dim]
        ref_sh_node = self.spherical_harmonics(ref_vec_node) # [N, sh_dim]
        ref_sh = ref_sh_node[i]                                  # [E, sh_dim]
        valid_ref_edge = valid_ref_node[i].unsqueeze(-1)
        if self.variant == "reference_gated":
            ref_sh = torch.where(valid_ref_edge, ref_sh, edge_sh)
        edge_ref_inv = self.sh_pair_inv(edge_sh, ref_sh)       # [E, lmax+1]
        edge_ref_inv = self.inv_norm(edge_ref_inv)

        # cosθ
        ref_vec_e = ref_vec_node[i]
        cos_theta = (edge_udiff * ref_vec_e).sum(-1, keepdim=True)  # [E, 1]
        if self.variant == "reference_gated":
            cos_theta = torch.where(valid_ref_edge, cos_theta, torch.ones_like(cos_theta))

        # ---- use invariants to gate rbf_h ----
        # concatenate cosθ and invariants
        geom_feat = torch.cat([cos_theta, edge_ref_inv], dim=-1)     # [E, L+1]
        gate = self.edge_inv_proj(geom_feat)                         # [E, 3H]
        gate = torch.tanh(gate)
        if self.variant == "reference_masked":
            gate = torch.where(valid_ref_edge, gate, torch.zeros_like(gate))
        rbf_h = rbf_h * (1.0 + gate)

        # ---- propagate messages ----
        scalar_proj = self.x_proj(node_scalar)                                      # [N, 3H]
        x_ji1, x_ji2, x_ji3 = torch.split(
            scalar_proj[j] * rbf_h * self.inv_sqrt_3,
            self.hidden_channels,
            dim=-1,
        )

        vec_ji = (
            x_ji1.unsqueeze(1) * node_vector[j]
            + x_ji2.unsqueeze(1) * edge_udiff.unsqueeze(2)
        )
        vec_ji = vec_ji * self.inv_sqrt_h

        delta_vector = scatter(vec_ji, index=i, dim=0, dim_size=node_scalar.size(0))
        delta_scalar = scatter(x_ji3, index=i, dim=0, dim_size=node_scalar.size(0))

        return delta_scalar, delta_vector


class SelfInteractionBlock(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, node_scalar, node_vector):

        vec1, vec2 = torch.split(
            self.vec_proj(node_vector), self.hidden_channels, dim=-1
        )

        scalar_vector_proj = self.xvec_proj(
            torch.cat(
                [node_scalar, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            scalar_vector_proj, self.hidden_channels, dim=-1
        )

        gate = torch.tanh(xvec3)
        delta_scalar = xvec2 * self.inv_sqrt_2 + node_scalar * gate

        delta_vector = xvec1.unsqueeze(1) * vec1

        return delta_scalar, delta_vector
