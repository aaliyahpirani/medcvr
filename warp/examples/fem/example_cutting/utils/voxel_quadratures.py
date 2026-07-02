# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools

import warp as wp
import warp.fem as fem

__all__ = ["neural_quadrature", "regular_quadrature"]


@wp.kernel
def instantiate_regular_quadrature(
    clip: bool,
    cell_indices: wp.array2d(dtype=int),
    vertex_sdf: wp.array(dtype=float),
    cell_alphas: wp.array2d(dtype=float),
    regular_coords: wp.array(dtype=wp.vec3),
    regular_weights: wp.array(dtype=float),
    cell_qp_coords: wp.array2d(dtype=wp.vec3),
    cell_qp_weights: wp.array2d(dtype=float),
    active_cells: wp.array(dtype=int),
):
    """A warp kernel that fills in per-cell quadrature points, clipping the exterior ones if requested.
    
    Args:
        clip: if True, clip the quadrature points that are outside of the cell
        cell_indices: contains the 8 vertex indices for the current cell
        vertex_sdf: contains the eight vertex SDF values for the current cell
        cell_alphas: contains the 8 alphas for the current cell, used for interpolation of the SDF at the quadrature points
        regular_coords: contains the quadrature points for the current cell in the cubes local coordinate system
        regular_weights: contains the weights for the quadrature points
        cell_qp_coords: contains the coordinates of the quadrature points for the current cell
        cell_qp_weights: contains the weights of the quadrature points for the current cell
        active_cells: contains the active cells for the current cell
    Returns:
        No returns, fills in cell_qp_coords, cell_qp_weights and active_cells
    """

    i = wp.tid() # create a thread index 
    vidx = cell_indices[i] # get the vertex indices for the current cell 
    alphas = cell_alphas[i] # get the alphas for the current cell

    # test if active
    min_sdf = float(1.0e8) # initialize the minimum SDF to a large value 
    for k in range(cell_indices.shape[1]):
        min_sdf = wp.min(min_sdf, vertex_sdf[vidx[k]]) # find the minimum SDF for the current cell
    active_cells[i] = wp.where(min_sdf <= 0.0, 1, 0) # if the minimum is less than or equal to 0, set the active cell to 1, otherwise it is inactive

    for j in range(regular_coords.shape[0]): # for each quadrature point
        coords = regular_coords[j] # get the coordinates of the current quadrature point
        cell_qp_coords[i, j] = coords # store the coordinates of the current quadrature point in cell_qp_coords
        cell_qp_weights[i, j] = regular_weights[j] # store the weight of the current quadrature point in cell_qp_weights

        if clip:
            # Clip quadrature -- disable exterior qps

            x = coords[0] # get the x coordindate
            y = coords[1] # get the y coordinate
            sdf = ( # interpolate the SDF at the current qudrature point 
                (1.0 - x) * (1.0 - y) * vertex_sdf[vidx[0]] * alphas[0]
                + (x) * (1.0 - y) * vertex_sdf[vidx[1]] * alphas[1]
                + (1.0 - x) * (y) * vertex_sdf[vidx[2]] * alphas[2]
                + (x) * (y) * vertex_sdf[vidx[3]] * alphas[3]
            )

            if cell_indices.shape[1] > 4: # if the cell has more than 4 vertices
                z = coords[2]
                sdf = (1.0 - z) * sdf + z * ( # interpolate the SDF at the current quadrature point
                    (1.0 - x) * (1.0 - y) * vertex_sdf[vidx[4]] * alphas[4]
                    + (x) * (1.0 - y) * vertex_sdf[vidx[5]] * alphas[5]
                    + (1.0 - x) * (y) * vertex_sdf[vidx[6]] * alphas[6]
                    + (x) * (y) * vertex_sdf[vidx[7]] * alphas[7]
                )

            if sdf > 0.0: # if the SDF is greater than 0, set the weight of the qudrature point to 0
                cell_qp_weights[i, j] = 0.0


def regular_quadrature(cell_vtx, sdf, cell_alpha, clip=True, order=2):
    """
    Takes in cell vertices, SDF values and alphas, and returns quadrature points, weights and active cells. 
    
    Args:
        cell_vtx (np.ndarray): the vertices of each cell 
        sdf (np.ndarray): the SDF values at each vertex
        cell_alpha (np.ndarray): the alphas for each cell
    
    Returns:
        qc (wp.array): the quadrature points
        qw (wp.array): the quadrature weights
        active_cells (wp.array): the active cells
    """

    cell_vtx = wp.array(cell_vtx, dtype=int) 
    sdf = wp.array(sdf, dtype=float)
    cell_alpha = wp.array(cell_alpha, dtype=float)

    # if the cell has 8 vertices, use the cube quadrature points, otherwise use the square ones 
    if cell_vtx.shape[1] == 8:
        reg_points, reg_weights = fem.geometry.element.Cube().instantiate_quadrature(
            order=order, family=fem.Polynomial.GAUSS_LEGENDRE
        )
    else:
        reg_points, reg_weights = fem.geometry.element.Square().instantiate_quadrature(
            order=order, family=fem.Polynomial.GAUSS_LEGENDRE
        ) # generates template quadrature points and weights 

    n_qp = len(reg_weights)
    reg_qp = wp.array(reg_points, dtype=wp.vec3)
    reg_qw = wp.array(reg_weights, dtype=float)

    qc = wp.empty(shape=(cell_vtx.shape[0], n_qp), dtype=wp.vec3) # empty array to store quadrature points, weights, and active cells
    qw = wp.empty(shape=(cell_vtx.shape[0], n_qp), dtype=float)
    active_cells = wp.empty(shape=(cell_vtx.shape[0]), dtype=int)

    wp.launch(
        instantiate_regular_quadrature,
        dim=cell_vtx.shape[0],
        inputs=[clip, cell_vtx, sdf, cell_alpha, reg_qp, reg_qw],
        outputs=[qc, qw, active_cells],
    )

    return qc, qw, active_cells


@functools.cache
def _load_model(model_path: str):
    import torch

    model = torch.jit.load(model_path)
    model.eval()
    return model


def infer_quadrature(model, cell_vtx, sdf, cell_alpha):
    """Inferred quadrature points from MLP"""
    import torch

    cell_sdf = sdf[cell_vtx]

    qc, qw = model(cell_sdf * cell_alpha)

    qc = qc.float().flip(dims=(2,))  # flip because FC cube corners are z-major

    min_sdf, _ = torch.min(cell_sdf, dim=1)
    active_cells = torch.where(min_sdf < 0, 1, 0)

    if qc.shape[-1] == 2:
        qc = torch.cat((qc, torch.zeros_like(qc[..., -1]).unsqueeze(-1)), dim=2)

    qc = qc.contiguous()
    qw = (qw + 1.0e-8).contiguous()

    return qc, qw, active_cells


def neural_quadrature(model_path, cell_vtx, sdf, cell_alpha):
    import torch

    # convert to torch and run inference
    device = str(wp.get_device())

    sdf = torch.tensor(sdf, device=device, dtype=torch.float32)
    cube = torch.tensor(cell_vtx, device=device, dtype=torch.int)
    cell_alpha = torch.tensor(cell_alpha, device=device, dtype=torch.float32)

    model = _load_model(model_path)
    qc, qw, active_cells = infer_quadrature(model, cube, sdf, cell_alpha)

    qc_wp = wp.clone(wp.from_torch(qc, dtype=wp.vec3, requires_grad=False))
    qw_wp = wp.clone(wp.from_torch(qw, dtype=wp.float32, requires_grad=False))
    active_cells = wp.clone(wp.from_torch(active_cells.int(), dtype=wp.int32, requires_grad=False))

    return qc_wp, qw_wp, active_cells
