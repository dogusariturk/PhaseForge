"""Exercise the installed CLI with two manufactured sublattice systems."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from pycalphad import Database

from phaseforge_uq import BayesianTdbFit, load_bundle
from validation.prototype.cef_fixtures import fixtures


def configuration(fixture):
    """Build independent fixed-site observations with known formula units."""
    rng = np.random.default_rng(2026100721)
    sites = rng.uniform(0.06, 0.94, (60, 2))
    temperatures = np.tile([600.0, 900.0, 1200.0], 20)
    center = fixture.nominal_parameters
    scale = np.r_[np.full(9, 6000.0), 5.0]
    truth = center + rng.normal(size=10) * scale * 0.2
    energy = fixture.independent_energy(sites, temperatures, truth) + rng.normal(0, 80.0, 60)
    fit = dict(
        design=fixture.design_matrix(sites, temperatures).tolist(),
        energies=energy.tolist(),
        offset=fixture.ideal_offset(sites, temperatures).tolist(),
        observation_kind="fixed_configuration",
        energy_units="J/mol_atoms",
        coefficient_ids=fixture.parameter_names,
        coefficient_units=[p["units"] for p in fixture.parameters],
        prior_mean=center.tolist(),
        prior_scale=scale.tolist(),
        noise_scale=150.0,
        sigma_bounds=[1e-4, 5000.0],
        grid_size=801,
    )
    metadata = dict(
        parameters=fixture.parameters,
        phase_model=dict(
            phase=fixture.name, constituents=[list(s) for s in fixture.species], site_ratios=list(fixture.site_ratios)
        ),
    )
    # Rebuild PARAMETER records from their identities, retaining independent TDB declarations.
    header = fixture.render_tdb().split("PARAMETER", 1)[0]
    lines = []
    for i, parameter in enumerate(fixture.parameters[:9]):
        expression = "{{" + parameter["id"] + "}}"
        if i == 4:
            expression += "+({{" + fixture.parameters[9]["id"] + "}})*T"
        lines.append(f"PARAMETER {parameter['tdb_identifier']} 298.15 {expression}; 3000 N !")
    config = dict(schema_version=1, fit=fit, metadata=metadata, tdb_template="model.tdb.in", sample_count=256, seed=72)
    return config, header + "\n".join(lines) + "\n"


@pytest.mark.parametrize("name", list(fixtures()))
def test_cli_fit_export_relocate_predict(tmp_path, name):
    """CLI output preserves joint draws, unit conversion and native TDB parsing."""
    fixture = fixtures()[name]
    config, template = configuration(fixture)
    source = tmp_path / "input"
    source.mkdir()
    (source / "fit.json").write_text(json.dumps(config))
    (source / "model.tdb.in").write_text(template)
    command = [
        sys.executable,
        "-m",
        "phaseforge_uq.cli",
        str(source / "fit.json"),
        "--output",
        str(tmp_path / "bundle"),
    ]
    process = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    shutil.move(tmp_path / "bundle", tmp_path / "relocated")
    tdb = tmp_path / "relocated/mean.tdb"
    payload, metadata, arrays = load_bundle(tdb)
    assert fixture.name in Database(str(tdb)).phases  # Standard reader skips UQ comments.
    fit = BayesianTdbFit(**config["fit"])
    draws, sigmas = fit.sample(256, seed=72)
    np.testing.assert_array_equal(arrays["joint_draws"], draws)
    np.testing.assert_array_equal(arrays["sigmas"], sigmas)
    np.testing.assert_array_equal(arrays["mean"], fit.mean)
    query_sites = np.array([[0.31, 0.72], [0.69, 0.22]])
    query_temperature = np.array([750.0, 1050.0])
    expected = np.stack([fixture.independent_energy(query_sites, query_temperature, row) for row in draws])
    recovered = np.stack(
        [fixture.independent_energy(query_sites, query_temperature, row) for row in arrays["joint_draws"]]
    )
    np.testing.assert_array_equal(expected, recovered)
    assert metadata["observation_kind"] == "fixed_configuration"
    assert metadata["inference"]["prior_mean"] == config["fit"]["prior_mean"]
    assert metadata["quadrature"]["normalized_moment_error"] < 1e-4
    assert "PARAMETER" in payload
    before = {p.name: p.read_bytes() for p in tdb.parent.iterdir()}
    with pytest.raises(ValueError, match="overwrite"):
        fit.export(fixture.render_tdb(fit.mean), config["metadata"], tdb.parent)
    assert before == {p.name: p.read_bytes() for p in tdb.parent.iterdir()}


@pytest.mark.parametrize("kind", ["internal_equilibrium", "multiphase", "unknown"])
def test_unsupported_observation_semantics(kind):
    """The public API never treats relaxed observations as fixed sites."""
    config, _ = configuration(next(iter(fixtures().values())))
    config["fit"]["observation_kind"] = kind
    with pytest.raises(NotImplementedError, match="fixed_configuration"):
        BayesianTdbFit(**config["fit"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("energy_units", "eV/atom"),
        ("grid_size", 32),
        ("prior_scale", [0.0] * 10),
        ("coefficient_ids", ["duplicate"] * 10),
    ],
)
def test_invalid_contract(field, value):
    """Reject ambiguous units, invalid priors, and duplicate parameter names."""
    config, _ = configuration(next(iter(fixtures().values())))
    config["fit"][field] = value
    with pytest.raises(ValueError):
        BayesianTdbFit(**config["fit"])


def test_prediction_intervals_and_bad_export(tmp_path):
    """Prediction adds the fixed baseline, distinguishes noise, and checks maps."""
    fixture = next(iter(fixtures().values()))
    config, _ = configuration(fixture)
    fit = BayesianTdbFit(**config["fit"])
    rows = np.zeros((1, 10))
    np.testing.assert_array_equal(fit.intervals(rows, offset=[123.0]), [[123.0, 123.0]])
    predictive = fit.intervals(rows, offset=[123.0], predictive=True)
    assert predictive[0, 0] < 123.0 < predictive[0, 1]
    for level in [0, 1, float("nan")]:
        with pytest.raises(ValueError):
            fit.intervals(rows, level=level)
    for count in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            fit.sample(count)
    config["metadata"]["parameters"].reverse()
    with pytest.raises(ValueError, match="mapping order"):
        fit.export(fixture.render_tdb(fit.mean), config["metadata"], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
