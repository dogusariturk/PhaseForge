"""Version1 TDB-comment pointer bundle; no pycalphad roundtrip serialization.

write_bundle(mean_tdb_text, metadata, arrays, output_dir, tdb_name='mean.tdb')
  -> new TDB Path. metadata['parameters'] is an ordered list of coefficient maps.
arrays: joint_draws(n,p), mean(p,), sigmas(n,), coefficient_ids Unicode(p,).
load_bundle(tdb_path) -> (comment-stripped TDB text, metadata dict, arrays dict).

The mean may be an analytic posterior mean and need not equal the finite sample
mean. Sigmas are strictly positive per-draw scalar noise parameters; their
scientific units/likelihood meaning should be recorded by the calling model.
Checksums detect accidental inconsistency, not malicious replacement of a whole
bundle. Missing pointers are explicit errors, including after TDB roundtrips.
"""

from pathlib import Path, PurePosixPath, PureWindowsPath
import hashlib
import json
import re
import io
import zipfile
import numpy as np

VERSION = 1
PREFIX = "$ PHASEFORGE_UQ_VERSION "
POINTER_KEYS = ("PHASEFORGE_UQ_VERSION", "PHASEFORGE_UQ_METADATA", "PFUQ_SHA256")


def pointer_header(metadata_name, digest):
    lines = ["$ PHASEFORGE_UQ_VERSION 1", "$ PHASEFORGE_UQ_METADATA " + metadata_name, "$ PFUQ_SHA256 " + digest]
    if any(len(line) > 78 for line in lines):
        raise BundleError("Pointer line exceeds78 characters")
    return "\n".join(lines) + "\n"


RESERVED = {"schema_version", "payload_sha256", "npz_path", "npz_sha256", "coefficient_ids", "array_shapes"}


class BundleError(ValueError):
    pass


class MissingDirectiveError(BundleError):
    pass


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def strip_tdb_comments(text):
    """Remove $ comments, retaining all non-comment payload bytes/line endings."""
    out = []
    for line in text.splitlines(keepends=True):
        if "$" not in line:
            out.append(line)
            continue
        before = line.split("$", 1)[0]
        if not before.strip():
            continue
        ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        out.append(before + ending)
    return "".join(out)


def _relative(base, name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise BundleError("Invalid relative path")
    p = PurePosixPath(name)
    if p.is_absolute() or PureWindowsPath(name).drive or any(part in (".", "..") for part in name.split("/")):
        raise BundleError("Absolute/traversal paths forbidden")
    target = base.joinpath(*p.parts).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError:
        raise BundleError("Symlink path escapes bundle directory") from None
    return target


def _mapping(meta):
    pars = meta.get("parameters")
    if not isinstance(pars, list) or not pars:
        raise BundleError("Ordered parameters list required")
    ids = []
    identities = []
    for p in pars:
        if not isinstance(p, dict):
            raise BundleError("Parameter map must be object")
        for key in ["id", "phase", "parameter_type"]:
            if not isinstance(p.get(key), str) or not p[key].strip():
                raise BundleError("Nonempty " + key + " required")
        ca = p.get("constituent_array")
        if (
            not isinstance(ca, list)
            or not ca
            or any(
                not isinstance(sub, list) or not sub or any(not isinstance(e, str) or not e.strip() for e in sub)
                for sub in ca
            )
        ):
            raise BundleError("Full ordered constituent_array required")
        if (
            not isinstance(p.get("parameter_order"), int)
            or isinstance(p["parameter_order"], bool)
            or p["parameter_order"] < 0
        ):
            raise BundleError("Nonnegative parameter_order required")
        term = p.get("polynomial_term")
        units = p.get("units")
        if term not in ("constant", "T"):
            raise BundleError("Unsupported polynomial term")
        if units != ("J/mol_formula_unit" if term == "constant" else "J/mol_formula_unit/K"):
            raise BundleError("Coefficient units do not match polynomial term")
        ids.append(p["id"])
        # Sublattice position and constituent order are part of identity; never sort.
        identity = (
            p["phase"].upper(),
            p["parameter_type"].upper(),
            tuple(tuple(e.upper() for e in sub) for sub in ca),
            p["parameter_order"],
            term,
        )
        identities.append(identity)
    if len(set(ids)) != len(ids) or len(set(identities)) != len(identities):
        raise BundleError("Duplicate coefficient ID or full parameter identity")
    if "coefficient_ids" in meta and meta["coefficient_ids"] != ids:
        raise BundleError("Metadata column order mismatch")
    return ids


def _arrays(arrays, ids):
    if not isinstance(arrays, dict) or not {"joint_draws", "mean", "sigmas", "coefficient_ids"} <= arrays.keys():
        raise BundleError("Missing required arrays")
    a = {k: np.asarray(val) for k, val in arrays.items()}
    p = len(ids)
    draws = a["joint_draws"]
    if draws.ndim != 2 or draws.shape[0] < 1 or draws.shape[1] != p:
        raise BundleError("joint_draws must be nonempty n by p")
    if a["mean"].shape != (p,) or a["sigmas"].shape != (draws.shape[0],):
        raise BundleError("Mean/sigma dimensions inconsistent")
    labels = a["coefficient_ids"]
    if labels.dtype.kind != "U" or labels.shape != (p,) or labels.tolist() != ids:
        raise BundleError("NPZ coefficient column labels mismatch")
    for key, val in a.items():
        if key == "coefficient_ids":
            continue
        if val.dtype.kind not in "fiu" or not np.all(np.isfinite(val)):
            raise BundleError("Nonfinite or unsupported numeric array: " + key)
    if np.any(a["sigmas"] <= 0):
        raise BundleError("Sigmas must be positive")
    return a


def _json_unique(pairs):
    result = {}
    for key, val in pairs:
        if key in result:
            raise BundleError("Duplicate JSON key: " + key)
        result[key] = val
    return result


def semantic_validate(tdb_text, metadata, mean):
    """Require exact mapped CEF identities and complete constant/T coefficient maps.

    Fixed unmapped parameters/functions may remain. Mapped expressions must be
    explicit A+B*T on every active Piecewise branch; no hidden named functions,
    P dependence or higher powers are accepted by protocolv1.
    """
    import sympy as sp
    from pycalphad import Database, variables as v

    try:
        db = Database(tdb_text)
    except Exception as exc:
        raise BundleError("Mean TDB cannot be parsed") from exc
    phase_model = metadata.get("phase_model")
    if phase_model is not None:
        if not isinstance(phase_model, dict):
            raise BundleError("phase_model must be object")
        name = str(phase_model.get("phase", "")).upper()
        if name not in db.phases:
            raise BundleError("phase_model phase missing")
        phase = db.phases[name]
        actual = [{s.name for s in sub} for sub in phase.constituents]
        declared = phase_model.get("constituents")
        if (
            not isinstance(declared, list)
            or len(declared) != len(actual)
            or any(
                not isinstance(sub, list) or {str(e).upper() for e in sub} != actual[i]
                for i, sub in enumerate(declared)
            )
        ):
            raise BundleError("phase_model sublattice constituents mismatch")
        try:
            ratios = np.asarray(phase_model["site_ratios"], float)
        except (KeyError, TypeError, ValueError) as exc:
            raise BundleError("phase_model site_ratios invalid") from exc
        if (
            ratios.shape != (len(phase.sublattices),)
            or not np.all(np.isfinite(ratios))
            or not np.allclose(ratios, np.array(phase.sublattices, float), rtol=0, atol=1e-12)
        ):
            raise BundleError("phase_model site_ratios mismatch")
    T = sp.sympify(v.T)
    groups = {}
    for index, p in enumerate(metadata["parameters"]):
        phase = p["phase"].upper()
        ca = tuple(tuple(e.upper() for e in sub) for sub in p["constituent_array"])
        key = (phase, p["parameter_type"].upper(), ca, p["parameter_order"])
        if phase not in db.phases:
            raise BundleError("Mapped phase missing")
        declared = db.phases[phase].constituents
        if len(ca) != len(declared) or any(
            any(e not in {spc.name for spc in declared[i]} for e in sub) for i, sub in enumerate(ca)
        ):
            raise BundleError("Mapped sublattice constituent/position invalid")
        groups.setdefault(key, {})[p["polynomial_term"]] = float(mean[index])
    parameters = db._parameters.all()
    for key, terms in groups.items():
        phase, typ, ca, order = key
        hits = [
            p
            for p in parameters
            if p["phase_name"] == phase
            and p["parameter_type"] == typ
            and p["parameter_order"] == order
            and tuple(tuple(e.name for e in sub) for sub in p["constituent_array"]) == ca
        ]
        if len(hits) != 1:
            raise BundleError("Mapped full parameter identity missing or nonunique")
        expr = sp.sympify(hits[0]["parameter"])
        branches = expr.args if isinstance(expr, sp.Piecewise) else [(expr, True)]
        active = []
        for branch, condition in branches:
            if isinstance(expr, sp.Piecewise) and condition == sp.true and branch == 0:
                continue
            if branch.free_symbols - {T}:
                raise BundleError("Mapped parameter has unsupported named function or variable")
            try:
                poly = sp.Poly(sp.expand(branch), T)
            except sp.PolynomialError as exc:
                raise BundleError("Mapped parameter is not A+B*T") from exc
            if poly.degree() > 1:
                raise BundleError("Mapped parameter has unsupported temperature power")
            actual = {"constant": float(poly.nth(0)), "T": float(poly.nth(1))}
            for term, val in actual.items():
                if term not in terms and abs(val) > 1e-12:
                    raise BundleError("Incomplete constant/T mapping")
                expected = terms.get(term, 0.0)
                if abs(val - expected) > 1e-8 + 1e-12 * max(abs(val), abs(expected)):
                    raise BundleError("Mapped mean does not reproduce TDB coefficient")
            active.append(actual)
        if not active:
            raise BundleError("No active mapped parameter expression")
    return True


def write_bundle(mean_tdb_text, metadata, arrays, output_dir, tdb_name="mean.tdb"):
    """Write new files; refuse existing targets and preexisting UQ directives."""
    if not isinstance(mean_tdb_text, str):
        raise BundleError("Mean TDB must be text")
    if any(any(key in line for key in POINTER_KEYS) for line in mean_tdb_text.splitlines()):
        raise BundleError("Input already contains UQ directive")
    if not isinstance(metadata, dict):
        raise BundleError("Metadata must be object")
    if "schema_version" in metadata and (
        type(metadata["schema_version"]) is not int or metadata["schema_version"] != VERSION
    ):
        raise BundleError("Unsupported schema version")
    ids = _mapping(metadata)
    a = _arrays(arrays, ids)
    root = Path(output_dir).resolve()
    if Path(tdb_name).name != tdb_name or not tdb_name.lower().endswith(".tdb"):
        raise BundleError("TDB filename must be a simple .tdb basename")
    tdb = _relative(root, tdb_name)
    stem = Path(tdb_name).stem
    npz = _relative(root, stem + ".uq.npz")
    js = _relative(root, stem + ".uq.json")
    if any(p.exists() for p in [tdb, npz, js]):
        raise BundleError("Refusing to overwrite existing bundle files")
    payload = strip_tdb_comments(mean_tdb_text)
    if not payload.strip():
        raise BundleError("Empty mean TDB payload")
    semantic_validate(payload, metadata, a["mean"])
    pointer_header(js.name, "0" * 64)  # validate length before creating files
    root.mkdir(parents=True, exist_ok=True)
    # Copy metadata rather than changing the caller's objects.
    meta = json.loads(json.dumps(metadata, allow_nan=False))
    meta.update(
        schema_version=VERSION,
        payload_sha256=_digest(payload.encode()),
        npz_path=npz.name,
        coefficient_ids=ids,
        array_shapes={k: list(v.shape) for k, v in a.items()},
    )
    np.savez_compressed(npz, **a)
    meta["npz_sha256"] = _digest(npz.read_bytes())
    js.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    header = pointer_header(js.name, _digest(js.read_bytes()))
    # Preserve original ordinary comments; payload integrity excludes comments.
    tdb.write_text(header + mean_tdb_text, encoding="utf-8")
    return tdb


def load_bundle(tdb_path):
    """Load/validate bundle or fail explicitly; never infer missing sidecars."""
    tdb = Path(tdb_path)
    root = tdb.parent.resolve()
    _relative(root, tdb.name)  # reject an external primary-file symlink too
    try:
        text = tdb.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise BundleError("Cannot read TDB") from exc
    found = {key: [] for key in POINTER_KEYS}
    for line in text.splitlines():
        if any(key in line for key in POINTER_KEYS):
            if len(line) > 78:
                raise BundleError("Pointer line exceeds78 characters")
            match = re.fullmatch(r"\$ (PHASEFORGE_UQ_VERSION|PHASEFORGE_UQ_METADATA|PFUQ_SHA256) (\S+)", line.strip())
            if match is None:
                raise BundleError("Malformed UQ directive")
            found[match[1]].append(match[2])
    if not any(found.values()):
        raise MissingDirectiveError(
            "UQ pointer missing; a conventional TDB roundtrip may discard comments. Restore the intact bundle; never infer sidecars."
        )
    if any(len(vals) != 1 for vals in found.values()):
        raise BundleError("Missing or duplicate UQ directive")
    if found["PHASEFORGE_UQ_VERSION"] != ["1"]:
        raise BundleError("Unsupported pointer version")
    expected_digest = found["PFUQ_SHA256"][0]
    if not re.fullmatch("[0-9a-f]{64}", expected_digest):
        raise BundleError("Malformed metadata checksum")
    mp = _relative(root, found["PHASEFORGE_UQ_METADATA"][0])
    try:
        raw = mp.read_bytes()
    except OSError as exc:
        raise BundleError("Missing metadata file") from exc
    if _digest(raw) != expected_digest:
        raise BundleError("Metadata checksum mismatch")
    try:
        meta = json.loads(
            raw,
            object_pairs_hook=_json_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(BundleError("Nonfinite JSON value")),
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise BundleError("Invalid/truncated metadata JSON") from exc
    if not isinstance(meta, dict) or type(meta.get("schema_version")) is not int or meta["schema_version"] != VERSION:
        raise BundleError("Unsupported metadata version")
    payload = strip_tdb_comments(text)
    if _digest(payload.encode()) != meta.get("payload_sha256"):
        raise BundleError("Comment-free TDB payload checksum mismatch")
    ids = _mapping(meta)
    npz = _relative(root, meta.get("npz_path"))
    try:
        rawnpz = npz.read_bytes()
    except OSError as exc:
        raise BundleError("Missing NPZ file") from exc
    if _digest(rawnpz) != meta.get("npz_sha256"):
        raise BundleError("NPZ checksum mismatch")
    try:
        with np.load(io.BytesIO(rawnpz), allow_pickle=False) as data:
            if len(data.files) != len(set(data.files)):
                raise BundleError("Duplicate NPZ keys")
            arrays = {k: data[k] for k in data.files}
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise BundleError("Invalid/truncated NPZ") from exc
    arrays = _arrays(arrays, ids)
    if meta.get("array_shapes") != {k: list(v.shape) for k, v in arrays.items()}:
        raise BundleError("Recorded array dimensions mismatch")
    semantic_validate(payload, meta, arrays["mean"])
    return payload, meta, arrays
