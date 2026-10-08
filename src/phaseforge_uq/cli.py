"""JSON-configured fitting command for explicit fixed-site model designs."""

import argparse
import json
import re
from pathlib import Path

from phaseforge_uq.fit import BayesianTdbFit


def fit_config(config_path, output_dir):
    """Fit one JSON configuration and export the rendered mean TDB bundle."""
    path = Path(config_path)
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Expected configuration schema_version 1")
    allowed = {"schema_version", "fit", "metadata", "tdb_template", "sample_count", "seed"}
    if set(config) - allowed:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(set(config) - allowed)))
    fit = BayesianTdbFit(**config["fit"])
    template = (path.parent / config["tdb_template"]).read_text()
    tokens = re.findall(r"\{\{([^{}]+)\}\}", template)
    if set(tokens) != set(fit.coefficient_ids):
        raise ValueError("Template placeholders must match fitted coefficient IDs exactly")
    values = dict(zip(fit.coefficient_ids, fit.mean))
    text = re.sub(r"\{\{([^{}]+)\}\}", lambda m: format(values[m.group(1)], ".17g"), template)
    if "{{" in text or "}}" in text:
        raise ValueError("Unresolved template placeholder")
    return fit.export(
        text, config["metadata"], output_dir, count=config.get("sample_count", 20000), seed=config.get("seed", 20261007)
    )


def main(argv=None):
    """Run the optional PhaseForge Bayesian fit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON input; template paths are relative to this file")
    parser.add_argument("--output", required=True, type=Path, help="Directory for new TDB/JSON/NPZ files")
    args = parser.parse_args(argv)
    try:
        result = fit_config(args.config, args.output)
    except (ValueError, TypeError, KeyError, OSError, NotImplementedError) as exc:
        parser.exit(2, f"phaseforge-uq: {exc}\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
