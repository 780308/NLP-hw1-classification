"""Explicit channel registry for Stage-I representation comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from .extractors import GROUP_SLICES


@dataclass(frozen=True)
class Representation:
    identifier: str
    groups: tuple[str, ...]

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(index for group in self.groups for index in range(*GROUP_SLICES[group]))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.channels), 7, 7)

    @property
    def flatten_dim(self) -> int:
        return len(self.channels) * 49


INITIAL_REPRESENTATIONS = (
    Representation("REP-001-HOG", ("hog",)),
    Representation("REP-002-HOG-LBP", ("hog", "lbp")),
    Representation("REP-003-ROOTSIFT", ("rootsift",)),
    Representation("REP-004-ROOTSIFT-LBP", ("rootsift", "lbp")),
    Representation("REP-005-ROOTSIFT-LBP-HSV", ("rootsift", "lbp", "hsv")),
)
RAW_REPRESENTATION_ID = "REP-000-RAW64"


def get_representation(identifier: str, fusion_groups: tuple[str, ...] | None = None) -> Representation:
    for representation in INITIAL_REPRESENTATIONS:
        if representation.identifier == identifier:
            return representation
    if identifier == "REP-006-FUSION" and fusion_groups:
        if len(set(fusion_groups)) != len(fusion_groups):
            raise ValueError("Fusion groups must be unique")
        return Representation(identifier, fusion_groups)
    raise KeyError(identifier)
