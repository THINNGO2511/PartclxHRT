"""
Hybrid Analytical + SA Placer

3-stage pipeline:
  1. Differentiable analytical placement (LSE wirelength + density + overlap penalties)
  2. Greedy legalization (resolve remaining overlaps)
  3. Multi-start SA refinement with fast approximate cost

Usage:
    uv run evaluate submissions/hybrid_placer.py
    uv run evaluate submissions/hybrid_placer.py --all
    uv run evaluate submissions/hybrid_placer.py -b ibm01 --vis
"""

import math
import time
import random
import torch
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Net extraction: convert plc net structure → tensor-friendly format
# ---------------------------------------------------------------------------

def extract_nets(benchmark, plc):
    """
    Extract net connectivity into tensor-friendly format.

    Returns:
        net_hard_pins: list of lists of (tensor_idx, x_offset, y_offset)
        net_fixed_x: list of lists of float (fixed pin x positions)
        net_fixed_y: list of lists of float (fixed pin y positions)
        edges: [E, 2] tensor of hard macro pairs with connections
        edge_weights: [E] tensor of connection weights
    """
    n_hard = benchmark.num_hard_macros

    # Build name -> plc index lookup
    name_to_pidx = {}
    for idx, m in enumerate(plc.modules_w_pins):
        name_to_pidx[m.get_name()] = idx

    # Map macro names -> tensor indices
    hard_name_to_tidx = {}
    for tidx, plc_idx in enumerate(benchmark.hard_macro_indices):
        hard_name_to_tidx[plc.modules_w_pins[plc_idx].get_name()] = tidx

    soft_name_to_tidx = {}
    for i, plc_idx in enumerate(benchmark.soft_macro_indices):
        soft_name_to_tidx[plc.modules_w_pins[plc_idx].get_name()] = n_hard + i

    # Extract nets
    net_hard_pins = []
    net_fixed_x = []
    net_fixed_y = []

    # Also build pairwise edge weights for SA
    edge_dict = {}

    for net_name, sink_names in plc.nets.items():
        all_pin_names = [net_name] + sink_names
        hard_pins = []
        fixed_xs = []
        fixed_ys = []

        for pn in all_pin_names:
            parts = pn.split("/")
            macro_name = parts[0]

            if macro_name in hard_name_to_tidx:
                tidx = hard_name_to_tidx[macro_name]
                if pn in name_to_pidx:
                    m = plc.modules_w_pins[name_to_pidx[pn]]
                    if hasattr(m, 'x_offset'):
                        hard_pins.append((tidx, m.x_offset, m.y_offset))
                    else:
                        hard_pins.append((tidx, 0.0, 0.0))
                else:
                    hard_pins.append((tidx, 0.0, 0.0))

            elif macro_name in soft_name_to_tidx:
                tidx = soft_name_to_tidx[macro_name]
                pos = benchmark.macro_positions[tidx]
                if pn in name_to_pidx:
                    m = plc.modules_w_pins[name_to_pidx[pn]]
                    if hasattr(m, 'x_offset'):
                        fixed_xs.append(pos[0].item() + m.x_offset)
                        fixed_ys.append(pos[1].item() + m.y_offset)
                    else:
                        fixed_xs.append(pos[0].item())
                        fixed_ys.append(pos[1].item())
                else:
                    fixed_xs.append(pos[0].item())
                    fixed_ys.append(pos[1].item())

            elif pn in name_to_pidx:
                m = plc.modules_w_pins[name_to_pidx[pn]]
                p = m.get_pos()
                fixed_xs.append(p[0])
                fixed_ys.append(p[1])

        total_pins = len(hard_pins) + len(fixed_xs)
        if total_pins >= 2:
            net_hard_pins.append(hard_pins)
            net_fixed_x.append(fixed_xs)
            net_fixed_y.append(fixed_ys)

            # Build edge weights for SA
            hard_indices = [hp[0] for hp in hard_pins]
            if len(hard_indices) >= 2:
                w = 1.0 / (len(hard_indices) - 1)
                for a in range(len(hard_indices)):
                    for b in range(a + 1, len(hard_indices)):
                        pair = (min(hard_indices[a], hard_indices[b]),
                                max(hard_indices[a], hard_indices[b]))
                        edge_dict[pair] = edge_dict.get(pair, 0) + w

    if edge_dict:
        edges = torch.tensor(list(edge_dict.keys()), dtype=torch.long)
        edge_weights = torch.tensor([edge_dict[e] for e in edge_dict], dtype=torch.float32)
    else:
        edges = torch.zeros(0, 2, dtype=torch.long)
        edge_weights = torch.zeros(0)

    return net_hard_pins, net_fixed_x, net_fixed_y, edges, edge_weights


# ---------------------------------------------------------------------------
# Padded tensor net representation (for vectorized ops)
# ---------------------------------------------------------------------------

def build_padded_nets(net_hard_pins, net_fixed_x, net_fixed_y, canvas_w, canvas_h):
    """
    Convert variable-length nets into padded tensors for vectorized computation.

    Returns dict with:
        - var_pin_idx: [V, max_hard] int64 tensor of hard macro indices (padded with 0)
        - var_pin_xoff: [V, max_hard] float tensor of x offsets
        - var_pin_yoff: [V, max_hard] float tensor of y offsets  
        - var_pin_mask: [V, max_hard] bool tensor (True = valid pin)
        - var_fixed_x: [V, max_fixed] float tensor of fixed pin x positions
        - var_fixed_y: [V, max_fixed] float tensor of fixed pin y positions
        - var_fixed_mask: [V, max_fixed] bool tensor
        - fixed_wl: float, total wirelength of nets with no hard macro pins
        - total_nets: int
        - norm: float, normalization factor
    """
    var_indices = [i for i, hp in enumerate(net_hard_pins) if len(hp) > 0]
    V = len(var_indices)

    if V == 0:
        return None

    max_hard = max(len(net_hard_pins[i]) for i in var_indices)
    max_fixed = max((len(net_fixed_x[i]) for i in var_indices), default=0)
    max_fixed = max(max_fixed, 1)  # At least 1 to avoid empty tensors

    pin_idx = np.zeros((V, max_hard), dtype=np.int64)
    pin_xoff = np.zeros((V, max_hard), dtype=np.float64)
    pin_yoff = np.zeros((V, max_hard), dtype=np.float64)
    pin_mask = np.zeros((V, max_hard), dtype=bool)
    fix_x = np.full((V, max_fixed), -1e9, dtype=np.float64)
    fix_y = np.full((V, max_fixed), -1e9, dtype=np.float64)
    fix_mask = np.zeros((V, max_fixed), dtype=bool)

    for vi, ni in enumerate(var_indices):
        hpins = net_hard_pins[ni]
        for j, (tidx, xo, yo) in enumerate(hpins):
            pin_idx[vi, j] = tidx
            pin_xoff[vi, j] = xo
            pin_yoff[vi, j] = yo
            pin_mask[vi, j] = True
        for j, (fx, fy) in enumerate(zip(net_fixed_x[ni], net_fixed_y[ni])):
            fix_x[vi, j] = fx
            fix_y[vi, j] = fy
            fix_mask[vi, j] = True

    # Fixed-only nets
    fixed_wl = 0.0
    for i, hpins in enumerate(net_hard_pins):
        if len(hpins) == 0 and len(net_fixed_x[i]) >= 2:
            xs = np.array(net_fixed_x[i])
            ys = np.array(net_fixed_y[i])
            fixed_wl += (xs.max() - xs.min()) + (ys.max() - ys.min())

    total_nets = len(net_hard_pins)
    norm = (canvas_w + canvas_h) * total_nets

    return {
        'pin_idx': torch.tensor(pin_idx, dtype=torch.long),
        'pin_xoff': torch.tensor(pin_xoff, dtype=torch.float32),
        'pin_yoff': torch.tensor(pin_yoff, dtype=torch.float32),
        'pin_mask': torch.tensor(pin_mask, dtype=torch.bool),
        'fix_x': torch.tensor(fix_x, dtype=torch.float32),
        'fix_y': torch.tensor(fix_y, dtype=torch.float32),
        'fix_mask': torch.tensor(fix_mask, dtype=torch.bool),
        'fixed_wl': fixed_wl,
        'total_nets': total_nets,
        'norm': norm,
        # Numpy versions for fast_hpwl
        'np_pin_idx': pin_idx,
        'np_pin_xoff': pin_xoff,
        'np_pin_yoff': pin_yoff,
        'np_pin_mask': pin_mask,
        'np_fix_x': fix_x,
        'np_fix_y': fix_y,
        'np_fix_mask': fix_mask,
    }


# ---------------------------------------------------------------------------
# Fast approximate HPWL (vectorized numpy)
# ---------------------------------------------------------------------------

def build_fast_hpwl(net_data):
    """
    Build a vectorized HPWL evaluator from padded net data.
    Returns a function: fast_hpwl(hard_positions_np) -> normalized_hpwl
    """
    if net_data is None:
        return lambda pos: 0.0

    pin_idx = net_data['np_pin_idx']      # [V, max_hard]
    pin_xoff = net_data['np_pin_xoff']
    pin_yoff = net_data['np_pin_yoff']
    pin_mask = net_data['np_pin_mask']
    fix_x = net_data['np_fix_x']          # [V, max_fixed]
    fix_y = net_data['np_fix_y']
    fix_mask = net_data['np_fix_mask']
    fixed_wl = net_data['fixed_wl']
    norm = net_data['norm']

    BIG = 1e18

    def fast_hpwl(pos):
        """pos: [n_hard, 2] numpy array."""
        # Gather hard pin positions: [V, max_hard]
        hx = pos[pin_idx, 0] + pin_xoff
        hy = pos[pin_idx, 1] + pin_yoff
        # Mask invalid pins with extreme values
        hx = np.where(pin_mask, hx, BIG)
        hy = np.where(pin_mask, hy, BIG)
        hx_max = np.where(pin_mask, hx, -BIG)
        hy_max = np.where(pin_mask, hy, -BIG)

        # Fixed pins
        fx = np.where(fix_mask, fix_x, BIG)
        fy = np.where(fix_mask, fix_y, BIG)
        fx_max = np.where(fix_mask, fix_x, -BIG)
        fy_max = np.where(fix_mask, fix_y, -BIG)

        # HPWL per net = (max_x - min_x) + (max_y - min_y)
        all_min_x = np.minimum(hx.min(axis=1), fx.min(axis=1))
        all_max_x = np.maximum(hx_max.max(axis=1), fx_max.max(axis=1))
        all_min_y = np.minimum(hy.min(axis=1), fy.min(axis=1))
        all_max_y = np.maximum(hy_max.max(axis=1), fy_max.max(axis=1))

        wl = ((all_max_x - all_min_x) + (all_max_y - all_min_y)).sum() + fixed_wl
        return wl / norm

    return fast_hpwl


# ---------------------------------------------------------------------------
# Fast approximate density cost (grid-based, numpy)
# ---------------------------------------------------------------------------

def build_fast_density(benchmark):
    """
    Build a fast vectorized density evaluator.
    Returns: fast_density(hard_positions_np) -> density_cost (top-10% avg)
    """
    n_hard = benchmark.num_hard_macros
    grid_rows = benchmark.grid_rows
    grid_cols = benchmark.grid_cols
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    cell_w = cw / grid_cols
    cell_h = ch / grid_rows
    cell_area = cell_w * cell_h
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)

    # Precompute soft macro density (constant)
    soft_density = np.zeros(grid_rows * grid_cols, dtype=np.float64)
    n_soft = benchmark.num_soft_macros
    if n_soft > 0:
        soft_pos = benchmark.macro_positions[n_hard:].numpy()
        soft_sizes = benchmark.macro_sizes[n_hard:].numpy()
        for i in range(n_soft):
            x, y = soft_pos[i]
            sw, sh = soft_sizes[i]
            x0 = max(0, int((x - sw / 2) / cell_w))
            x1 = min(grid_cols - 1, int((x + sw / 2) / cell_w))
            y0 = max(0, int((y - sh / 2) / cell_h))
            y1 = min(grid_rows - 1, int((y + sh / 2) / cell_h))
            for gy in range(y0, y1 + 1):
                for gx in range(x0, x1 + 1):
                    ox0 = max(x - sw / 2, gx * cell_w)
                    ox1 = min(x + sw / 2, (gx + 1) * cell_w)
                    oy0 = max(y - sh / 2, gy * cell_h)
                    oy1 = min(y + sh / 2, (gy + 1) * cell_h)
                    if ox1 > ox0 and oy1 > oy0:
                        soft_density[gy * grid_cols + gx] += (ox1 - ox0) * (oy1 - oy0) / cell_area

    k = max(1, int(0.1 * grid_rows * grid_cols))
    areas = sizes[:, 0] * sizes[:, 1] / cell_area

    def fast_density(pos):
        """pos: [n_hard, 2] numpy array. Vectorized center-of-mass density."""
        density = soft_density.copy()
        gx = np.clip((pos[:, 0] / cell_w).astype(int), 0, grid_cols - 1)
        gy = np.clip((pos[:, 1] / cell_h).astype(int), 0, grid_rows - 1)
        gidx = gy * grid_cols + gx
        np.add.at(density, gidx, areas)
        top_k = np.partition(density, -k)[-k:]
        return top_k.mean()

    return fast_density


def build_fast_congestion(benchmark, net_hard_pins, net_fixed_x, net_fixed_y):
    """
    Build a fast RUDY-like congestion estimator.
    Distributes routing demand uniformly across each net's bounding box.
    Returns: fast_congestion(hard_positions_np) -> congestion_cost (top-5% avg)
    """
    n_hard = benchmark.num_hard_macros
    grid_rows = benchmark.grid_rows
    grid_cols = benchmark.grid_cols
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    cell_w = cw / grid_cols
    cell_h = ch / grid_rows
    total_cells = grid_rows * grid_cols

    # Precompute fixed-only net congestion
    fixed_cong = np.zeros(total_cells, dtype=np.float64)
    variable_net_data = []

    for i, hpins in enumerate(net_hard_pins):
        fxs = net_fixed_x[i]
        fys = net_fixed_y[i]
        if len(hpins) == 0:
            if len(fxs) >= 2:
                xs = np.array(fxs)
                ys = np.array(fys)
                _add_rudy(fixed_cong, xs.min(), ys.min(), xs.max(), ys.max(),
                          cell_w, cell_h, grid_cols, grid_rows, len(fxs))
        else:
            tidxs = np.array([hp[0] for hp in hpins], dtype=np.int64)
            x_offs = np.array([hp[1] for hp in hpins], dtype=np.float64)
            y_offs = np.array([hp[2] for hp in hpins], dtype=np.float64)
            variable_net_data.append((
                tidxs, x_offs, y_offs,
                np.array(fxs, dtype=np.float64) if fxs else None,
                np.array(fys, dtype=np.float64) if fys else None,
                len(hpins) + len(fxs)))

    k_cong = max(1, int(0.05 * total_cells))

    def fast_congestion(pos):
        cong = fixed_cong.copy()
        for tidxs, x_offs, y_offs, fx, fy, n_pins in variable_net_data:
            hx = pos[tidxs, 0] + x_offs
            hy = pos[tidxs, 1] + y_offs
            if fx is not None:
                all_x = np.concatenate([hx, fx])
                all_y = np.concatenate([hy, fy])
            else:
                all_x = hx
                all_y = hy
            _add_rudy(cong, all_x.min(), all_y.min(), all_x.max(), all_y.max(),
                      cell_w, cell_h, grid_cols, grid_rows, n_pins)
        top_k = np.partition(cong, -k_cong)[-k_cong:]
        return top_k.mean()

    return fast_congestion


def _add_rudy(cong_grid, x_min, y_min, x_max, y_max, cell_w, cell_h, grid_cols, grid_rows, n_pins):
    """Add RUDY routing demand for one net to congestion grid."""
    bb_w = x_max - x_min
    bb_h = y_max - y_min
    if bb_w < 1e-6 and bb_h < 1e-6:
        return
    demand = (n_pins - 1) / max(bb_w * bb_h, cell_w * cell_h)
    gx0 = max(0, int(x_min / cell_w))
    gx1 = min(grid_cols - 1, int(x_max / cell_w))
    gy0 = max(0, int(y_min / cell_h))
    gy1 = min(grid_rows - 1, int(y_max / cell_h))
    for gy in range(gy0, gy1 + 1):
        for gx in range(gx0, gx1 + 1):
            cong_grid[gy * grid_cols + gx] += demand


# ---------------------------------------------------------------------------
# Differentiable analytical placement (PyTorch)
# ---------------------------------------------------------------------------

def analytical_place(benchmark, net_data, n_iters=1500, lr=0.05):
    """
    Vectorized differentiable placement using:
    - Log-sum-exp wirelength approximation (batched over all nets)
    - Differentiable overlap penalty with increasing weight

    Returns: [n_hard, 2] numpy array of hard macro positions.
    """
    if net_data is None:
        return benchmark.macro_positions[:benchmark.num_hard_macros].numpy().astype(np.float64)

    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].float()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    # Initialize from current positions
    pos = benchmark.macro_positions[:n_hard].clone().float().requires_grad_(True)

    # Padded net tensors
    pin_idx = net_data['pin_idx']       # [V, max_hard]
    pin_xoff = net_data['pin_xoff']
    pin_yoff = net_data['pin_yoff']
    pin_mask = net_data['pin_mask']     # [V, max_hard] bool
    fix_x = net_data['fix_x']          # [V, max_fixed]
    fix_y = net_data['fix_y']
    fix_mask = net_data['fix_mask']     # [V, max_fixed] bool
    norm = net_data['norm']

    # Replace masked positions with large/small for LSE
    V = pin_idx.shape[0]

    optimizer = torch.optim.Adam([pos], lr=lr)
    lambda_overlap = 0.5
    mask_triu = torch.triu(torch.ones(n_hard, n_hard, dtype=torch.bool), diagonal=1)

    for step in range(n_iters):
        optimizer.zero_grad()

        # --- Vectorized LSE wirelength ---
        gamma = 2.0 + min(6.0, step / n_iters * 6.0)

        # Gather hard pin positions: [V, max_hard, 2]
        hx = pos[pin_idx, 0] + pin_xoff  # [V, max_hard]
        hy = pos[pin_idx, 1] + pin_yoff

        # For LSE: masked positions should not affect max or min
        # Use -inf for max computation (logsumexp), +inf for min (logsumexp of negated)
        LARGE = 1e6
        hx_for_max = torch.where(pin_mask, hx, torch.tensor(-LARGE))
        hy_for_max = torch.where(pin_mask, hy, torch.tensor(-LARGE))
        hx_for_min = torch.where(pin_mask, hx, torch.tensor(LARGE))
        hy_for_min = torch.where(pin_mask, hy, torch.tensor(LARGE))

        fx_for_max = torch.where(fix_mask, fix_x, torch.tensor(-LARGE))
        fy_for_max = torch.where(fix_mask, fix_y, torch.tensor(-LARGE))
        fx_for_min = torch.where(fix_mask, fix_x, torch.tensor(LARGE))
        fy_for_min = torch.where(fix_mask, fix_y, torch.tensor(LARGE))

        # Concatenate hard and fixed: [V, max_hard + max_fixed]
        all_x_max = torch.cat([hx_for_max, fx_for_max], dim=1)
        all_y_max = torch.cat([hy_for_max, fy_for_max], dim=1)
        all_x_min = torch.cat([hx_for_min, fx_for_min], dim=1)
        all_y_min = torch.cat([hy_for_min, fy_for_min], dim=1)

        # LSE approximation of max - min per net
        wl_x = (torch.logsumexp(gamma * all_x_max, dim=1) +
                 torch.logsumexp(-gamma * all_x_min, dim=1)) / gamma
        wl_y = (torch.logsumexp(gamma * all_y_max, dim=1) +
                 torch.logsumexp(-gamma * all_y_min, dim=1)) / gamma
        wl_loss = (wl_x + wl_y).sum() / norm

        # --- Overlap penalty (vectorized) ---
        cx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
        cy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)
        sw = half_w.unsqueeze(1) + half_w.unsqueeze(0)
        sh = half_h.unsqueeze(1) + half_h.unsqueeze(0)
        ox = torch.clamp(sw - cx.abs(), min=0)
        oy = torch.clamp(sh - cy.abs(), min=0)
        overlap_loss = (ox * oy)[mask_triu].sum()

        loss = wl_loss + lambda_overlap * overlap_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pos[:, 0].clamp_(half_w, cw - half_w)
            pos[:, 1].clamp_(half_h, ch - half_h)

        lambda_overlap = min(lambda_overlap * 1.005, 200.0)

    return pos.detach().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Legalization: resolve overlaps with minimum displacement
# ---------------------------------------------------------------------------

def legalize(pos, sizes_np, movable, canvas_w, canvas_h):
    """
    Try two strategies, keep the one producing better placement quality.
    Uses fast HPWL as tie-breaker (lower = better).
    A) Gentle repulsion + greedy spiral fallback
    B) Greedy-only (skip repulsion)
    """
    half_w = sizes_np[:, 0] / 2
    half_h = sizes_np[:, 1] / 2
    gap = 0.05
    orig = pos.copy()

    pos_a = _legalize_repulsion_greedy(orig.copy(), sizes_np, movable, canvas_w, canvas_h, half_w, half_h, gap)
    pos_b = _legalize_greedy_only(orig.copy(), sizes_np, movable, canvas_w, canvas_h, half_w, half_h, gap)

    # Pick the one with less displacement from original positions
    # (proxy for "which preserved more of the original quality")
    # Weight by macro area so displacing large macros is penalized more
    areas = sizes_np[:, 0] * sizes_np[:, 1]
    disp_a = np.sum(areas * np.sum((pos_a - orig) ** 2, axis=1))
    disp_b = np.sum(areas * np.sum((pos_b - orig) ** 2, axis=1))
    return pos_a if disp_a <= disp_b else pos_b


def _legalize_repulsion_greedy(pos, sizes_np, movable, cw, ch, half_w, half_h, gap):
    """Gentle repulsion then greedy fallback."""
    n = len(pos)
    for iteration in range(50):
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        sx = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2 + gap
        sy = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2 + gap
        ox = sx - np.abs(dx)
        oy = sy - np.abs(dy)
        mask = np.triu((ox > 0) & (oy > 0))
        np.fill_diagonal(mask, False)
        pairs = np.argwhere(mask)
        if len(pairs) == 0:
            break
        for i, j in pairs:
            ddx, ddy = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]
            if ox[i, j] < oy[i, j]:
                p = ox[i, j] / 2 + 0.02
                s = 1.0 if ddx >= 0 else -1.0
                if movable[i]: pos[i, 0] += s * p
                if movable[j]: pos[j, 0] -= s * p
            else:
                p = oy[i, j] / 2 + 0.02
                s = 1.0 if ddy >= 0 else -1.0
                if movable[i]: pos[i, 1] += s * p
                if movable[j]: pos[j, 1] -= s * p
        pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
        pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)
    return _legalize_greedy_only(pos, sizes_np, movable, cw, ch, half_w, half_h, gap)


def _legalize_greedy_only(pos, sizes_np, movable, cw, ch, half_w, half_h, gap):
    """Greedy sequential: largest first, spiral search for conflicts."""
    n = len(pos)
    sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
    sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2
    order = sorted(range(n), key=lambda i: -sizes_np[i, 0] * sizes_np[i, 1])
    placed = np.zeros(n, dtype=bool)
    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue
        if placed.any():
            da = np.abs(pos[idx, 0] - pos[:, 0])
            db = np.abs(pos[idx, 1] - pos[:, 1])
            cf = (da < sep_x[idx] + gap) & (db < sep_y[idx] + gap) & placed
            cf[idx] = False
            if not cf.any():
                placed[idx] = True
                continue
        else:
            placed[idx] = True
            continue
        orig = pos[idx].copy()
        step = max(sizes_np[idx, 0], sizes_np[idx, 1]) * 0.2
        best_p, best_d = orig.copy(), float('inf')
        for r in range(1, 150):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = np.clip(orig[0] + dxm * step, half_w[idx], cw - half_w[idx])
                    cy = np.clip(orig[1] + dym * step, half_h[idx], ch - half_h[idx])
                    da = np.abs(cx - pos[:, 0])
                    db = np.abs(cy - pos[:, 1])
                    cf = (da < sep_x[idx] + gap) & (db < sep_y[idx] + gap) & placed
                    cf[idx] = False
                    if cf.any():
                        continue
                    d = (cx - orig[0]) ** 2 + (cy - orig[1]) ** 2
                    if d < best_d:
                        best_d, best_p, found = d, np.array([cx, cy]), True
            if found:
                break
        pos[idx] = best_p
        placed[idx] = True
    return pos


# ---------------------------------------------------------------------------
# SA refinement with fast approximate cost
# ---------------------------------------------------------------------------

def sa_refine(pos, benchmark, fast_hpwl, fast_density, fast_congestion,
              edges_np, edge_weights_np,
              movable, sizes_np, n_iters=30000, seed=42):
    """
    Simulated annealing refinement using fast approximate cost.

    Uses wirelength + density as proxy (no congestion — too slow to approximate
    accurately, and Will showed WL-only is already competitive).
    """
    rng = random.Random(seed)
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    half_w = sizes_np[:, 0] / 2
    half_h = sizes_np[:, 1] / 2
    sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
    sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2
    gap = 0.05

    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return pos

    # Build neighbor lists from edges
    neighbors = [[] for _ in range(n_hard)]
    if len(edges_np) > 0:
        for i, j in edges_np:
            neighbors[i].append(j)
            neighbors[j].append(i)

    pos = pos.copy()

    def check_overlap(idx):
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        overlaps = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
        overlaps[idx] = False
        return overlaps.any()

    # Combined cost: match official proxy weights (1.0*WL + 0.5*density + 0.5*congestion)
    # But scale to roughly similar magnitudes for SA acceptance
    use_congestion = fast_congestion is not None
    # Evaluate congestion less frequently (it's slower)
    cong_cache = [0.0]
    cong_step = [0]

    def cost(force_cong=False):
        wl = fast_hpwl(pos)
        den = fast_density(pos)
        if use_congestion and (force_cong or cong_step[0] % 50 == 0):
            cong_cache[0] = fast_congestion(pos)
        cong_step[0] += 1
        return 1.0 * wl + 0.5 * den + 0.5 * cong_cache[0]

    current_cost = cost()
    best_pos = pos.copy()
    best_cost = current_cost

    T_start = max(cw, ch) * 0.15
    T_end = max(cw, ch) * 0.001

    stagnation = 0
    early_stop_threshold = max(3000, n_iters // 5)

    for step in range(n_iters):
        if stagnation > early_stop_threshold:
            break

        frac = step / n_iters
        T = T_start * (T_end / T_start) ** frac

        # Reheat on stagnation
        if stagnation > 300:
            T = T * 5
            stagnation = 0

        move_type = rng.random()
        i = rng.choice(movable_idx)
        old_x, old_y = pos[i, 0], pos[i, 1]

        if move_type < 0.4:
            # SHIFT
            shift = T * (0.3 + 0.7 * (1 - frac))
            pos[i, 0] = np.clip(pos[i, 0] + rng.gauss(0, shift), half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(pos[i, 1] + rng.gauss(0, shift), half_h[i], ch - half_h[i])
        elif move_type < 0.7:
            # MOVE TOWARD NEIGHBOR
            if neighbors[i]:
                j = rng.choice(neighbors[i])
                alpha = rng.uniform(0.05, 0.35)
                pos[i, 0] = np.clip(pos[i, 0] + alpha * (pos[j, 0] - pos[i, 0]),
                                     half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + alpha * (pos[j, 1] - pos[i, 1]),
                                     half_h[i], ch - half_h[i])
            else:
                continue
        elif move_type < 0.9:
            # SWAP with neighbor
            if neighbors[i]:
                cands = [j for j in neighbors[i] if movable[j]]
                if not cands:
                    continue
                j = rng.choice(cands)
            else:
                j = rng.choice(movable_idx)
            if i == j:
                continue
            old_jx, old_jy = pos[j, 0], pos[j, 1]
            pos[i, 0] = np.clip(old_jx, half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(old_jy, half_h[i], ch - half_h[i])
            pos[j, 0] = np.clip(old_x, half_w[j], cw - half_w[j])
            pos[j, 1] = np.clip(old_y, half_h[j], ch - half_h[j])
            if check_overlap(i) or check_overlap(j):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                stagnation += 1
                continue
            new_cost = cost()
            delta = new_cost - current_cost
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                pos[i, 0] = old_x; pos[i, 1] = old_y
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                stagnation += 1
            continue
        else:
            # RANDOM SHIFT (exploration)
            shift = max(cw, ch) * 0.1 * (1 - frac)
            pos[i, 0] = np.clip(pos[i, 0] + rng.uniform(-shift, shift), half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(pos[i, 1] + rng.uniform(-shift, shift), half_h[i], ch - half_h[i])

        # Check overlap for single-macro moves
        if check_overlap(i):
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            stagnation += 1
            continue

        new_cost = cost()
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_pos = pos.copy()
                stagnation = 0
            else:
                stagnation += 1
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            stagnation += 1

    return best_pos


# ---------------------------------------------------------------------------
# Multi-start SA (run multiple seeds, keep best)
# ---------------------------------------------------------------------------

def _run_sa_worker(args):
    """Worker for parallel SA."""
    (pos, benchmark_data, fast_hpwl_data, edges_np, edge_weights_np,
     movable, sizes_np, n_iters, seed) = args

    # Reconstruct fast functions (can't pickle closures)
    net_hard_pins, net_fixed_x, net_fixed_y, canvas_w, canvas_h = fast_hpwl_data
    fast_hpwl = build_fast_hpwl(net_hard_pins, net_fixed_x, net_fixed_y, canvas_w, canvas_h)

    grid_rows, grid_cols, n_hard = benchmark_data['grid_rows'], benchmark_data['grid_cols'], benchmark_data['n_hard']

    # Simplified density for SA workers
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows

    def fast_density(p):
        density = np.zeros(grid_rows * grid_cols, dtype=np.float64)
        for i in range(n_hard):
            x, y = p[i]
            w, h = sizes_np[i]
            gx = int(np.clip(x / cell_w, 0, grid_cols - 1))
            gy = int(np.clip(y / cell_h, 0, grid_rows - 1))
            density[gy * grid_cols + gx] += (w * h) / (cell_w * cell_h)
        k = max(1, int(0.1 * grid_rows * grid_cols))
        top_k = np.partition(density, -k)[-k:]
        return top_k.mean()

    result = sa_refine(pos, type('B', (), benchmark_data)(), fast_hpwl, fast_density,
                       edges_np, edge_weights_np, movable, sizes_np, n_iters, seed)
    score = fast_hpwl(result)
    return result, score


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

def _load_plc(name):
    """Load PlacementCost object for a benchmark."""
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root).replace("\\", "/"))
        return plc
    ng45 = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile"}
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            pb = str(base / "netlist.pb.txt").replace("\\", "/")
            pl = str(base / "initial.plc").replace("\\", "/")
            _, plc = load_benchmark(pb, pl)
            return plc
    return None


class HybridPlacer:
    """
    3-stage hybrid placer:
      1. Analytical placement (differentiable wirelength + overlap penalty)
      2. Legalization (greedy, minimum displacement)
      3. SA refinement (fast approximate cost, multi-start)
    """

    def __init__(self, sa_iters=30000, sa_starts=4, seed=42, time_budget=600):
        self.sa_iters = sa_iters
        self.sa_starts = sa_starts
        self.seed = seed
        self.time_budget = time_budget

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_start = time.time()
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n_hard].numpy()

        # --- Extract net connectivity ---
        plc = _load_plc(benchmark.name)
        if plc is None:
            return benchmark.macro_positions.clone()

        net_hard_pins, net_fixed_x, net_fixed_y, edges, edge_weights = extract_nets(benchmark, plc)
        net_data = build_padded_nets(net_hard_pins, net_fixed_x, net_fixed_y, cw, ch)

        fast_hpwl = build_fast_hpwl(net_data)
        fast_den = build_fast_density(benchmark)
        fast_cong = build_fast_congestion(benchmark, net_hard_pins, net_fixed_x, net_fixed_y)
        edges_np = edges.numpy() if len(edges) > 0 else np.zeros((0, 2), dtype=np.int64)
        ew_np = edge_weights.numpy() if len(edge_weights) > 0 else np.zeros(0, dtype=np.float32)

        def approx_proxy(p):
            return 1.0 * fast_hpwl(p) + 0.5 * fast_den(p) + 0.5 * fast_cong(p)

        # --- Stage 1: Legalize initial placement ---
        print(f"    Stage 1: Legalization...", end="", flush=True)
        t1 = time.time()
        hard_pos = benchmark.macro_positions[:n_hard].numpy().copy().astype(np.float64)
        hard_pos = legalize(hard_pos, sizes_np, movable, cw, ch)
        print(f" [{time.time()-t1:.1f}s]")

        # --- Stage 2: SA refinement from multiple seeds ---
        print(f"    Stage 2: SA refinement ({self.sa_starts} seeds × {self.sa_iters} iters, budget={self.time_budget}s)...", end="", flush=True)
        t2 = time.time()

        best_pos = hard_pos.copy()
        best_cost = approx_proxy(hard_pos)

        for s in range(self.sa_starts):
            if time.time() - t2 > self.time_budget:
                break
            result = sa_refine(
                hard_pos.copy(), benchmark, fast_hpwl, fast_den, fast_cong,
                edges_np, ew_np, movable, sizes_np,
                n_iters=self.sa_iters, seed=self.seed + s * 17
            )
            score = approx_proxy(result)
            if score < best_cost:
                best_cost = score
                best_pos = result

        print(f" [{time.time()-t2:.1f}s]")

        # --- Build final placement ---
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)

        elapsed = time.time() - t_start
        print(f"    Total: {elapsed:.1f}s  approx_proxy={best_cost:.4f}")

        return full_pos