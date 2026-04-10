# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Shim pkg_resources before pytorch_wavelets tries to import it.
# pytorch_wavelets uses pkg_resources.resource_stream to load filter .npz files,
# but pkg_resources was removed in setuptools>=72. This shim provides a minimal
# pkg_resources module using the modern importlib.resources API.
import importlib.resources as _ilr
import io
import sys
import types

if "pkg_resources" not in sys.modules:
    _pkg_resources_shim = types.ModuleType("pkg_resources")

    def _resource_stream(package: str, resource_name: str) -> io.BytesIO:
        data = _ilr.files(package).joinpath(resource_name).read_bytes()
        return io.BytesIO(data)

    _pkg_resources_shim.resource_stream = _resource_stream  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = _pkg_resources_shim

from .loss_module_physical import LossPhysical
from .loss_module_spectral import LossSpectralWFCL
from .loss_module_ssl import LossLatentSSLStudentTeacher

__all__ = [LossPhysical, LossLatentSSLStudentTeacher, LossSpectralWFCL]
