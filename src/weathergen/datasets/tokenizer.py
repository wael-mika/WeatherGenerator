# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings

import astropy_healpix as hp
import numpy as np
import torch

from weathergen.datasets.utils import (
    healpix_verts_rots,
    r3tos2,
)


class Tokenizer:
    """
    Base class for tokenizers.
    """

    def __init__(self, healpix_level: int, healpix_level_target: int | None = None):
        ref = torch.tensor([1.0, 0.0, 0.0])

        self.healpix_level = healpix_level
        self.hl_source = healpix_level
        self.hl_target = healpix_level_target if healpix_level_target is not None else healpix_level

        self.num_healpix_cells_source = 12 * 4**self.hl_source
        self.num_healpix_cells_target = 12 * 4**self.hl_target

        self.size_time_embedding = 6

        verts00_src, verts00_rots_src = healpix_verts_rots(self.hl_source, 0.0, 0.0)
        verts10_src, verts10_rots_src = healpix_verts_rots(self.hl_source, 1.0, 0.0)
        verts11_src, verts11_rots_src = healpix_verts_rots(self.hl_source, 1.0, 1.0)
        verts01_src, verts01_rots_src = healpix_verts_rots(self.hl_source, 0.0, 1.0)
        vertsmm_src, vertsmm_rots_src = healpix_verts_rots(self.hl_source, 0.5, 0.5)
        self.hpy_verts_source = [
            verts00_src.to(torch.float32),
            verts10_src.to(torch.float32),
            verts11_src.to(torch.float32),
            verts01_src.to(torch.float32),
            vertsmm_src.to(torch.float32),
        ]
        self.hpy_verts_rots_source = [
            verts00_rots_src.to(torch.float32),
            verts10_rots_src.to(torch.float32),
            verts11_rots_src.to(torch.float32),
            verts01_rots_src.to(torch.float32),
            vertsmm_rots_src.to(torch.float32),
        ]

        if self.hl_target != self.hl_source:
            verts00_tgt, verts00_rots_tgt = healpix_verts_rots(self.hl_target, 0.0, 0.0)
            verts10_tgt, verts10_rots_tgt = healpix_verts_rots(self.hl_target, 1.0, 0.0)
            verts11_tgt, verts11_rots_tgt = healpix_verts_rots(self.hl_target, 1.0, 1.0)
            verts01_tgt, verts01_rots_tgt = healpix_verts_rots(self.hl_target, 0.0, 1.0)
            vertsmm_tgt, vertsmm_rots_tgt = healpix_verts_rots(self.hl_target, 0.5, 0.5)
            self.hpy_verts_target = [
                verts00_tgt.to(torch.float32),
                verts10_tgt.to(torch.float32),
                verts11_tgt.to(torch.float32),
                verts01_tgt.to(torch.float32),
                vertsmm_tgt.to(torch.float32),
            ]
            self.hpy_verts_rots_target = [
                verts00_rots_tgt.to(torch.float32),
                verts10_rots_tgt.to(torch.float32),
                verts11_rots_tgt.to(torch.float32),
                verts01_rots_tgt.to(torch.float32),
                vertsmm_rots_tgt.to(torch.float32),
            ]
        else:
            verts00_tgt = verts00_src
            verts10_tgt = verts10_src
            verts11_tgt = verts11_src
            verts01_tgt = verts01_src
            vertsmm_tgt = vertsmm_src
            verts00_rots_tgt = verts00_rots_src
            verts10_rots_tgt = verts10_rots_src
            verts11_rots_tgt = verts11_rots_src
            verts01_rots_tgt = verts01_rots_src
            vertsmm_rots_tgt = vertsmm_rots_src
            self.hpy_verts_target = self.hpy_verts_source
            self.hpy_verts_rots_target = self.hpy_verts_rots_source

        self.hpy_verts = self.hpy_verts_target

        transforms = [
            ([verts10_tgt, verts11_tgt, verts01_tgt, vertsmm_tgt], verts00_rots_tgt),
            ([verts00_tgt, verts11_tgt, verts01_tgt, vertsmm_tgt], verts10_rots_tgt),
            ([verts00_tgt, verts10_tgt, verts01_tgt, vertsmm_tgt], verts11_rots_tgt),
            ([verts00_tgt, verts11_tgt, verts10_tgt, vertsmm_tgt], verts01_rots_tgt),
            ([verts00_tgt, verts10_tgt, verts11_tgt, verts01_tgt], vertsmm_rots_tgt),
        ]

        self.verts_local = []
        for _verts, rot in transforms:
            # Compute local coordinates
            verts = torch.stack(_verts)
            # shape: <healpix, 4, 3>
            verts = verts.transpose(0, 1)
            # Batch multiplication by the 3x3 rotation matrices.
            # shape: <healpix, 3, 3> @ <healpix, 4, 3> -> <healpix, 4, 3>
            # Needs to transpose first to <healpix, 3, 4> then transpose back.
            t1 = torch.bmm(rot, verts.transpose(-1, -2)).transpose(-2, -1)
            t2 = ref - t1
            self.verts_local.append(t2.flatten(1, 2))

        self.hpy_verts_local_target = torch.stack(self.verts_local).transpose(0, 1)

        # add local coords wrt to center of neighboring cells
        # (since the neighbors are used in the prediction)
        num_healpix_cells = 12 * 4**self.hl_target
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(
                np.arange(num_healpix_cells), 2**self.hl_target, order="nested"
            ).transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        self.hpy_nctrs_target = (
            vertsmm_tgt[temp.flatten()]
            .reshape((num_healpix_cells, 8, 3))
            .transpose(1, 0)
            .to(torch.float32)
        )

    def compute_source_centroids(self, source_tokens_cells: list[torch.Tensor]) -> torch.Tensor:
        source_means = [
            (
                self.hpy_verts_source[-1][i].unsqueeze(0).repeat(len(s), 1)
                if len(s) > 0
                else torch.tensor([])
            )
            for i, s in enumerate(source_tokens_cells)
        ]
        source_means_lens = [len(s) for s in source_means]
        # merge and split to vectorize computations
        source_means = torch.cat(source_means)
        # TODO: precompute also source_means_r3 and then just cat
        source_centroids = torch.cat(
            [source_means.to(torch.float32), r3tos2(source_means).to(torch.float32)], -1
        )
        source_centroids = torch.split(source_centroids, source_means_lens)

        return source_centroids

    def get_size_time_embedding(self) -> int:
        """
        Get size of time embedding
        """
        return self.size_time_embedding
