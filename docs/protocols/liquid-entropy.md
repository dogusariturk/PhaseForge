# Standard liquid entropy protocol

Adopted for the ORCHID PhaseForge workflow on 2026-11-03.

Liquid entropy assessment is a required step when constructing or refining a liquid CALPHAD model. The calculation follows Hong and Liu, Physical Review Research 7, L012030 (2025), https://doi.org/10.1103/PhysRevResearch.7.L012030. This protocol establishes the workflow requirement. The estimator is available as the `MLIPentropy` command (see [Estimator](#estimator-mlipentropy) below), with a pair (quasi-chemical) form of the chemical-mixing term.

## Calculation and acceptance

1. Retain the vibrational, configurational and electronic contributions separately. Record missing terms explicitly rather than treating them as zero. Preserve the potential identity, code revision, trajectory hashes, composition, temperature, pressure, atom count, equilibration and production settings, sampling interval and analysis settings.
2. Calculate alloy liquids and their pure-liquid references at matching temperatures, pressure and compatible simulation settings. Verify liquid behavior using diffusion and structural diagnostics. Retain failed or incomplete states with their rejection reasons.
3. At one representative composition and temperature for a new chemical system, compare 108 and 864 atoms with matched settings, normally 100 ps production. Accept 108 atoms for the broader campaign only if 100*abs(S108-S864)/abs(S864) < 5 for vibrational entropy. This acceptance does not validate electronic, configurational or excess entropy. Use an absolute tolerance for excess quantities near zero.
4. Check trajectory-length, spectral-window, DOS normalization and negative spectral weight sensitivity. Retain block statistics and distinguish sampling uncertainty from potential error and methodological uncertainty. Check electronic convergence and snapshot variability separately.
5. Validate the configurational convention against a controlled ideal-mixing limit before exporting total excess entropy. Declare how chemical mixing enters the estimator so that ideal mixing is included exactly once. Alternative pair selections remain separate sensitivity cases unless a calibrated uncertainty model justifies combining them.

## Mapping to CALPHAD

Use a common molar-atom basis. For a binary liquid with x = x_B,

    S_ideal = -R*((1-x)*ln(1-x) + x*ln(x))
    S_ex = S_liquid - (1-x)*S_A_liquid - x*S_B_liquid - S_ideal

For L_k(T) = a_k + b_k*T and the convention (x_A-x_B)^k,

    S_ex_model = -x*(1-x)*sum_k b_k*(1-2*x)^k

The equiatomic value constrains b_0 alone. Off-equiatomic compositions constrain the remaining combinations. Preserve covariance introduced by shared pure-liquid references. Compare multiple temperatures to test whether the assumed temperature dependence is adequate.

Use validated entropy observations as soft likelihood constraints with documented uncertainties. Do not infer hard parameter bounds from the spread between unvalidated methods or from sampling errors alone. Retain resolved component estimates as diagnostics when total entropy is not ready for fitting. Do not count total entropy, its components, or derived ratios as independent evidence from the same observations.

## Required handoff to PhaseForge+

Archive component values, reference subtractions, uncertainty assumptions, covariance, acceptance checks and an explicit fit-readiness decision with reasons. Compare predicted entropy for the reference assessment and candidate models before a separate entropy-constrained refinement. Preserve the phase-assemblage protocol, non-observation constraints and existing search records. Record the entropy contribution to the objective separately.

For reference, the Cu-Zr campaign's accepted finite-size benchmark at Cu50Zr50 and 2178 K is 95.1415 versus 95.5639 J/mol-atom/K for 108 versus 864 atoms, a 0.442% difference. This benchmark supports the vibrational cell-size choice only.

## Estimator: MLIPentropy

`MLIPentropy` computes the excess entropy of an alloy liquid with any number of species from NVT production trajectories of the alloy and of the pure liquid of every species at the same temperature. Trajectories are read with ASE (extxyz, ASE `.traj` or any other format that stores positions and velocities). Units are J/mol-atom/K.

    S_ex = S_vib,ex + S_chem,ex

The electronic term is not computed and is reported as missing, not as zero.

**Vibrational term.** The mass-weighted velocity autocorrelation function of each state is windowed (Hann, `-tmax`, default 3 ps) and transformed to a vibrational density of states normalised to 3N, which is converted with the quantum harmonic entropy per mode. S_vib,ex is the alloy value minus the composition-weighted pure-liquid values.

**Chemical term.** The species labels are treated on the neighbour graph of each frame, two atoms being neighbours when closer than the first minimum of the total radial distribution function (default; `-cutoff=partial` uses the partial minima, `-rc=` a fixed value). Let y_ij be the fraction of ordered bonds with species i at one end and j at the other (a k x k matrix), w_i = sum_j y_ij the bond-end fraction of species i, and zbar the mean coordination. The pair (quasi-chemical, or Bethe) approximation to the entropy of the labels, with the site and bond frequencies as constraints (Guggenheim 1944; Kikuchi 1951), adds to each site entropy the entropy of each bond less that of its two ends:

    S_chem/R = -sum_i x_i ln x_i - (zbar/2) sum_ij y_ij ln[ y_ij / (w_i w_j) ]
    S_chem,ex = S_chem - S_ideal = -(zbar/2) R I

where I >= 0 is the mutual information between the species at the two ends of a bond. Three properties follow, and the test suite checks each of them. For random labels on any graph, y_ij = w_i w_j, so I = 0 and S_chem is the ideal value -R sum_i x_i ln x_i exactly; ideal mixing is therefore counted once. Any chemical short-range order gives S_chem,ex < 0. The expression is exact for a one-dimensional Markov chain (zbar = 2) and on a Bethe lattice. Referring the bond statistics to the bond-end fractions w_i rather than to x_i keeps the coupling between species and coordination number out of the chemical term; the x_i-referenced form is reported as a sensitivity case. This term is a correction of the chemical-mixing term of Hong and Liu, Phys. Rev. Research 7, L012030 (2025), https://doi.org/10.1103/PhysRevResearch.7.L012030. With `-hongliu`, the tool also evaluates Hong-Liu Eqs. 15-16 on the same graphs of two-species alloys, for comparison; these values do not enter S_ex.

**Checks and uncertainties.** Before any excess entropy is reported, the species labels of each alloy are permuted at random on every MD neighbour graph. The pair term must then give |S_chem,ex| below `-ideal_tol` (default 0.05 J/mol-atom/K). If the check fails, the tool warns and marks every alloy as held. Standard errors come from `-nblock` blocks (default 5). The uncertainty of S_chem,ex combines the block standard error with the largest change over the cutoff cases (base, base -/+ `-rc_delta`, and the other RDF-minimum convention). The covariance between alloy states introduced by the shared pure-liquid references is retained. The bonds sampled for every species pair are reported, and the tool warns when a pair would have fewer than 1000 bonds under random mixing. Mean kinetic temperatures more than 5% from `-temp` are flagged.

**Mapping.** S_ex is fitted by generalised least squares, with that covariance, to S_ex = -sum_{i<j} x_i x_j sum_k b_k^ij (x_i - x_j)^k, the temperature derivative of the Redlich-Kister excess with L_k = a_k + b_k*T. Species are in alphabetical order. For a binary this is the expression given above. The order is the largest the compositions determine, at most 2 (`-order=` to set it). No ternary interaction terms are fitted, and when the compositions do not determine the parameters the tool reports S_ex and its covariance for use in the fit directly.

    MLIPentropy -temp=2200 -dt=2 -ref=Cr_liq.traj -ref=Ni_liq.traj \
                -alloy=Cr75Ni25.traj -alloy=Cr50Ni50.traj -alloy=Cr25Ni75.traj [-hongliu] -o=entropy.json

`-dt` is the time between saved frames in fs. `-skip` discards leading frames, `-stride` and `-rdf_stride` thin the frames used for the neighbour graphs and the RDFs, and `-seed` fixes the label permutations. The JSON output records the components, references, uncertainties, cutoffs, k x k bond matrices and per-pair bond counts, the ideal-mixing check, the b_k mapping with its covariance, the settings and SHA-256 hashes of the inputs and of the tool. Tests: `python -m pytest tests` (numpy, ase, pytest).
