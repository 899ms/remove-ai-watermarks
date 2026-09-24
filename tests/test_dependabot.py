"""Regression tests for Dependabot compatibility constraints."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_dependabot_numpy_ignore_mirrors_the_lock_constraint() -> None:
    constraints = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["uv"]["constraint-dependencies"]
    (numpy,) = (r for r in map(Requirement, constraints) if r.name == "numpy")
    (upper,) = (spec.version for spec in numpy.specifier if spec.operator == "<")

    config = Path(".github/dependabot.yml").read_text()
    numpy_ignore = config.split('dependency-name: "numpy"', maxsplit=1)[1].split("dependency-name:", maxsplit=1)[0]
    assert f'- ">={upper}"' in numpy_ignore
