"""Ordered slab dicts to keyword arguments for ``uniaxial_reflectivity``.

Each slab becomes one ``layers`` row (thickness, roughness; columns 1 and 2 are
zeroed for bookkeeping) and one 3x3 ``tensor`` row. In ``tjf4x4`` the kernel
then uses ``epsilon = conj(I - 2 * tensor)``. The **input** ``tensor`` diagonal
is the X-ray pair :math:`\\delta + i\\beta` per principal axis so that
:math:`\\tilde{n} = 1 - \\delta + i\\beta` matches the intended Henke-style
linearization used in that port.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, TypeGuard, cast, get_args

import numpy as np

from refloxide.pxr.tjf4x4 import uniaxial_reflectivity

index = Literal["nxx", "nyy", "nzz"]

sld_component = Literal[
    "delta", "beta", "delta_xx", "beta_xx", "delta_yy", "beta_yy", "delta_zz", "beta_zz"
]

type SLDConst = complex | np.ndarray
type SLDDispersive = Callable[..., SLDConst]
type SLDLike = SLDConst | SLDDispersive


def _is_material_key(key: str) -> TypeGuard[index | sld_component]:
    return key in get_args(index) or key in get_args(sld_component)


def resolve_sld(sld: SLDLike, *, energy: float | None = None, **kwargs):
    match sld:
        case complex() as z:
            return z
        case np.ndarray() as arr:
            return arr
        case _ if callable(sld) and not isinstance(sld, type):
            if not energy:
                raise ValueError("No energy to evaluate sld callable.")
            return cast("SLDDispersive", sld)(energy, **kwargs)
        case _:
            raise TypeError(f"unsupported sld type: {type(sld)}")


class Material(Enum):
    """Material Catagory."""

    SCALAR = "scalar"
    UNIAXIAL = "uniaxial"
    BIAXIAL = "biaxial"

    def tensor(self, index_tensor: SLDLike, **kwargs) -> np.ndarray:
        tensor_dict = self.tensor_dict(index_tensor, **kwargs)
        return np.array(
            [
                [1 - tensor_dict["nxx"], 0, 0],
                [0, 1 - tensor_dict["nyy"], 0],
                [0, 0, 1 - tensor_dict["nzz"]],
            ],
            dtype=np.complex128,
        )

    def tensor_dict(self, index_tensor: SLDLike, **kwargs) -> dict[index, complex]:
        index_tensor = self.resolve(index_tensor, **kwargs)
        match self:
            case Material.SCALAR:
                iso = self.isotropic(index_tensor)
                return {
                    "nxx": iso,
                    "nyy": iso,
                    "nzz": iso,
                }
            case Material.UNIAXIAL:
                # Assert the index is well behaved
                match index_tensor.shape:
                    case (3, 3):
                        return {
                            "nxx": index_tensor[0, 0] / 2 + index_tensor[1, 1] / 2,
                            "nyy": index_tensor[0, 0] / 2 + index_tensor[1, 1] / 2,
                            "nzz": index_tensor[2, 2],
                        }
                    case (3,):
                        return {
                            "nxx": index_tensor[0] / 2 + index_tensor[1] / 2,
                            "nyy": index_tensor[0] / 2 + index_tensor[1] / 2,
                            "nzz": index_tensor[2],
                        }
                    case (2, 2):
                        return {
                            "nxx": index_tensor[0, 0],
                            "nyy": index_tensor[0, 0],
                            "nzz": index_tensor[1, 1],
                        }
                    case (2,):
                        return {
                            "nxx": index_tensor[0],
                            "nyy": index_tensor[0],
                            "nzz": index_tensor[1],
                        }
                    case _:
                        raise ValueError(
                            f"""
                            Cannot confirm tenbsor shape got: {index_tensor.shape},
                            expected (3,3), (3,), (2,2), or (2,)
                            """
                        )
            case Material.BIAXIAL:
                return {
                    "nxx": index_tensor[0, 0],
                    "nyy": index_tensor[1, 1],
                    "nzz": index_tensor[2, 2],
                }

    def isotropic(self, index_tensor: SLDLike, **kwargs) -> complex:
        index_tensor = self.resolve(index_tensor, **kwargs)
        match index_tensor:
            case complex():
                return index_tensor
            case np.ndarray():
                return np.sum(index_tensor) / 3.0
            case _:
                raise TypeError(
                    "index_tensor must be a complex number or a numpy array,"
                    + f"got {type(index_tensor)}",
                )

    def resolve(self, index_of_refraction: SLDLike, **kwargs) -> np.ndarray:
        sld = resolve_sld(index_of_refraction, **kwargs)
        match sld:
            case complex() as z:
                return np.array([z, z, z], dtype=np.complex128)
            case np.ndarray():
                return sld
            case _:
                raise TypeError("Index of refraction not castable to complex dtype")

    def get(
        self, index_tensor: np.ndarray | complex, key: index | sld_component
    ) -> complex | float:
        tensor_dict = self.tensor_dict(index_tensor)
        match key:
            case "nxx" | "nyy" | "nzz":
                return tensor_dict[key]
            case "delta_xx":
                return 1.0 - tensor_dict["nxx"].real
            case "beta_xx":
                return tensor_dict["nxx"].imag
            case "delta_yy":
                return 1.0 - tensor_dict["nyy"].real
            case "beta_yy":
                return tensor_dict["nyy"].imag
            case "delta_zz":
                return 1.0 - tensor_dict["nzz"].real
            case "beta_zz":
                return tensor_dict["nzz"].imag
            case "delta":
                return 1.0 - self.isotropic(index_tensor).real
            case "beta":
                return self.isotropic(index_tensor).imag
            case _:
                raise KeyError(f"Invalid key: {key}")


@dataclass
class Layer:
    """One slab in stack order (fronting first, backing last).

    Parameters
    ----------
    thickness, roughness:
        Thickness (ångströms) and Nevot-Croce sigma (ångströms) for the interface
        into this slab. Omit or zero for semi-infinite claddings.
    delta_xx, beta_xx, delta_yy, beta_yy, delta_zz, beta_zz:
        Diagonal :math:`\\delta_{ii},\\beta_{ii}` with
        :math:`\\tilde{n}_{ii}=1-\\delta_{ii}+i\\beta_{ii}`.
    delta, beta:
        Shared in-plane :math:`\\delta,\\beta` for ``xx`` and ``yy`` when the six
        per-axis keys are absent.
    delta_zz, beta_zz:
        Out-of-plane pair; default to the same ``delta``/``beta`` as in-plane.
    nxx, nyy, nzz:
        Optional diagonal indices; when given, set
        :math:`\\delta = 1 - \\mathrm{Re}(\\tilde{n})`,
        :math:`\\beta = \\mathrm{Im}(\\tilde{n})` per axis.
    """

    thickness: float
    roughness: float
    material: Material
    sld: SLDLike

    def __getattribute__(
        self, name: str | index | sld_component, /
    ) -> complex | float | Material | np.ndarray:
        if _is_material_key(name):
            material = super().__getattribute__("material")
            sld = super().__getattribute__("sld")
            return material.get(sld, name)
        else:
            return super().__getattribute__(name)

    def __getitem__(self, key: str) -> float | None:
        return getattr(self, key)

    def tensor(self, **kwargs) -> np.ndarray:
        return self.material.tensor(self.sld, **kwargs)

    def slab(self, **kwargs):
        match self.sld:
            case complex() | np.ndarray():
                return np.array([self.thickness, self.delta, self.beta, self.roughness])
            case _:
                concrete = resolve_sld(self.sld, **kwargs)
                bound = replace(self, sld=concrete)
                return np.array(
                    [bound.thickness, bound.delta, bound.beta, bound.roughness]
                )


def stack_tensor(
    layers: list[Layer],
    *,
    energy: float | None = None,
    kwargs: list[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    layer_kwargs: list[Mapping[str, Any]] = (
        [{} for _ in layers] if kwargs is None else kwargs
    )
    return np.array(
        [
            layer.tensor(energy=energy, **kw)
            for layer, kw in zip(layers, layer_kwargs, strict=True)
        ],
        dtype=np.complex128,
    )


def stack_slabs(
    layers: list[Layer],
    *,
    energy: float | None = None,
    kwargs: list[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    layer_kwargs: list[Mapping[str, Any]] = (
        [{} for _ in layers] if kwargs is None else kwargs
    )
    return np.array(
        [
            layer.slab(energy=energy, **kw)
            for layer, kw in zip(layers, layer_kwargs, strict=True)
        ],
        dtype=np.float64,
    )


def reflectivity(
    layers: list[Layer],
    q: np.ndarray,
    energy: float,
    kwargs: list[Mapping[str, Any]] | None = None,
):
    refl, _trans, *components = uniaxial_reflectivity(
        q=q,
        layers=stack_slabs(layers, energy=energy, kwargs=kwargs),
        tensor=stack_tensor(layers, energy=energy, kwargs=kwargs),
        energy=energy,
    )
    return refl[:, 0, 0], refl[:, 1, 1], components


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from periodictable.xsf import index_of_refraction

    from refloxide.pxr.plugin.structure import MaterialSLD

    # callable test
    def ps_sld(e):
        return index_of_refraction("C8H8", density=1, energy=e * 1e-3)

    def si_sld(e):
        return index_of_refraction("Si", density=2.33, energy=e * 1e-3)

    # Simplified structure format
    vac = Layer(
        thickness=0.0,
        roughness=0.0,
        material=Material("scalar"),
        sld=complex(1, 0),
    )
    ps = Layer(
        thickness=200.0,
        roughness=5.0,
        material=Material("uniaxial"),
        sld=ps_sld,
    )
    si = Layer(
        thickness=0,
        roughness=0.5,
        material=Material("scalar"),
        sld=si_sld,
    )
    # refnx structure format
    vac_refnx = MaterialSLD("", 0, 250, name="vacuum")(0, 0)
    ps_refnx = MaterialSLD("C8H8", 1.0, 250, name="polystyrene")(200, 5.0)
    si_refnx = MaterialSLD("Si", 2.33, 250, name="silicon")(0, 0.5)
    stack = vac_refnx | ps_refnx | si_refnx
    q = np.linspace(0.001, 0.25, 1000)
    refl_p, refl_s, *_ = reflectivity(
        layers=[vac, ps, si],
        q=q,
        energy=250,
    )
    refl_refnx_p, refl_refnx_s, *_ = stack.reflectivity(q=q, energy=250)
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))
    ax[0].plot(q, refl_s, label="refloxide s", c="C0")
    ax[1].plot(q, refl_p, label="refloxide p", c="C1")
    ax[0].plot(q, refl_refnx_p, label="refnx s", ls="--", c="k")
    ax[1].plot(q, refl_refnx_s, label="refnx p", ls="--", c="k")
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].legend()
    ax[1].legend()
    plt.show()
