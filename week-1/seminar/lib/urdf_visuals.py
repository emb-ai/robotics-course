from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np


class VisualEntry(NamedTuple):
    mesh_path: Path
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


def _parse_xyz(elem: ET.Element | None) -> tuple[float, float, float]:
    if elem is None:
        return (0.0, 0.0, 0.0)
    s = (elem.get("xyz") or "0 0 0").strip().split()
    return (float(s[0]), float(s[1]), float(s[2])) if len(s) >= 3 else (0.0, 0.0, 0.0)


def _parse_rpy(elem: ET.Element | None) -> tuple[float, float, float]:
    if elem is None:
        return (0.0, 0.0, 0.0)
    s = (elem.get("rpy") or "0 0 0").strip().split()
    return (float(s[0]), float(s[1]), float(s[2])) if len(s) >= 3 else (0.0, 0.0, 0.0)


def _parse_rgba(elem: ET.Element | None) -> tuple[float, float, float, float]:
    if elem is None:
        return (0.5, 0.5, 0.5, 1.0)
    s = (elem.get("rgba") or "0.5 0.5 0.5 1.0").strip().split()
    return (float(s[0]), float(s[1]), float(s[2]), float(s[3])) if len(s) >= 4 else (0.5, 0.5, 0.5, 1.0)


def parse_urdf_visuals(urdf_path: Path) -> dict[str, list[VisualEntry]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    path_prefix = urdf_path.parent

    materials: dict[str, tuple[float, float, float, float]] = {}
    for mat in root.findall(".//material"):
        name = mat.get("name")
        if name is None:
            continue
        color = mat.find("color")
        materials[name] = _parse_rgba(color)

    result: dict[str, list[VisualEntry]] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name is None:
            continue
        entries: list[VisualEntry] = []
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            geom = visual.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue
            filename = mesh.get("filename")
            if not filename:
                continue
            mesh_path = path_prefix / filename.strip()
            mat_ref = visual.find("material")
            mat_name = mat_ref.get("name") if mat_ref is not None else None
            color_elem = mat_ref.find("color") if mat_ref is not None else None
            if color_elem is not None:
                rgba = _parse_rgba(color_elem)
            elif mat_name and mat_name in materials:
                rgba = materials[mat_name]
            else:
                rgba = (0.5, 0.5, 0.5, 1.0)
            entries.append(
                VisualEntry(
                    mesh_path=mesh_path,
                    origin_xyz=_parse_xyz(origin),
                    origin_rpy=_parse_rpy(origin),
                    rgba=rgba,
                )
            )
        if entries:
            result[name] = entries
    return result


def origin_to_matrix(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T
