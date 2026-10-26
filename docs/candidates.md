# Candidate products

An FFA frequency sweep now writes a versioned common candidate file,
`<prefix>_ffa_results.h5`. Searches operate on **one already dedispersed DM
trial**. Loki does not perform a DM search or distill candidates across DMs.

## Search configuration

```toml
[input]
timeseries = "trial.tim"
# Optional overrides when input metadata is missing or incorrect:
# dm = 42.5
# start_mjd = 60000.0
# barycentric = false

[search]
snr_min = 7.0

[candidates]
cluster = true
grouping = "transitive" # alternative: "anchored"
radius = 1.0
phase_radius = 0.05
retain_detections = true
flag_harmonics = false
remove_harmonics = false
harmonic_denominator_max = 100
harmonic_phase_distance = 1.0
harmonic_snr_distance = 3.0
max_candidates = 0
memory_gb = 1.0
# temporary_directory = "/scratch/loki"
```

CLI equivalents include `--no-cluster`, `--cluster-grouping transitive|anchored`, `--cluster-radius`, `--cluster-phase-radius`,
`--no-retain-detections`, `--flag-harmonics`, `--remove-harmonics`,
`--max-candidates`, `--candidate-memory-gb`, `--candidate-tempdir`,
`--dm`, `--start-mjd` and `--barycentric true|false`.
Harmonic removal requires harmonic flagging. Zero maximum candidates means
unlimited. Clustering is enabled by default; raw retention is enabled by default.

Python uses `libloki.configs.CandidateOptions` and
`libloki.io.ObservationMetadata`:

```python
from loki import libloki

options = libloki.configs.CandidateOptions()
options.grouping = "transitive"
options.radius = 1.0
options.phase_radius = 0.05
options.retain_detections = False
sweep = libloki.ffa.FFAFreqSweep(cfg, candidates=options)
sweep.execute(ts.ts_e, ts.ts_v, "results", "trial", observation=ts.observation)
```

Array-based callers provide observation metadata explicitly; omitted DM, start
MJD and timing frame remain unknown. Values from `.tim` and `.dat`/`.inf`
headers are available through `TimeSeries.observation`. A known DM of zero is
distinct from an unknown DM. CLI overrides take precedence over header values.

## Clustering and harmonic selection

The production minimum S/N is 7, configurable through `search.snr_min` or
`--snr-min`. This threshold reduces noise triggers; it does not deduplicate
responses to a real source.

Every FFA score passing `search.snr_min` remains a detection, including separate
boxcar-width trials. Candidate processing changes neither scores nor the search
grid. Detections are ordered by decreasing S/N; ties use ascending frequency
derivative vectors, narrower duty cycle, higher bin count, then detection ID.
The default `grouping = "transitive"` computes connected components of the
pairwise phase-track graph (friends of friends). Any detection can connect two
components, including a weak detection that would not be a representative.
Each final component is represented by its strongest original trial, with the
same deterministic tie breakers. Final cluster IDs are assigned in decreasing
representative S/N order after all bridges are resolved.

Two detections have an edge when the span (maximum minus minimum) of their
unwrapped phase residual is at most
`max(phase_radius, radius * min(width_a/nbins_a, width_b/nbins_b))` cycles.
`radius` defaults to 1 pulse width; `phase_radius` defaults to 0.05 cycles.
Constant phase offsets cancel. All polynomial extrema, including interior
stationary points, are evaluated, so acceleration and higher derivatives
participate in the comparison. For constant acceleration and midpoint time t,
`delta_phase(t) = delta_f*t + delta_fdot*t*t/2`, where `fdot = -f*accel/c`.
Equal frequencies with incompatible acceleration tracks do not get a direct
edge. Mixed offsets use the full phase polynomial, not separately normalized
frequency and acceleration cuts. Edges cross chunk/band boundaries and bin counts.

A transitive link limit is not a component-diameter bound. Chains can join
nearby independent pulsars; a component describes a connected response family,
not a guarantee of one source. `clusters/phase_extent` records the maximum
phase span from the representative to any member, in cycles, making long
chains visible. A later profile/secondary-peak verification step can distinguish
blended signals. Larger radii can increase bridging. No phase-only rule can
separate identical derivative tracks differing only in pulse phase.

Exact derivative-track duplicates are compressed in the neighbor index, while
all their raw detections and membership counts are preserved. The strongest
trial supplies the representative; the widest recorded duty cycle supplies
the union of that track's possible neighbor edges. A later weaker, wider trial
therefore triggers another neighbor query. This compression exactly preserves
the full per-width graph; it does not discard weaker widths before clustering.
EP inputs that already retain only the winning width use the same interface.

Set `grouping = "anchored"` to restore strongest-representative suppression:
each trial can join only an existing representative, so weak trials cannot
bridge components. This rule can leave several representatives on one response.
Both modes use the same edge predicate. `phase_radius = 0` disables the minimum
phase-cycle floor; `cluster = false` disables grouping entirely and keeps each
detection as its own representative.

Representative parameters and scores are never averaged or added. The default
uses riptide-style transitive grouping with full Taylor phase tracks rather
than frequency alone. Harmonic relationships are handled separately.

Harmonic processing is a separate opt-in pass over representatives. It finds
the closest positive rational frequency ratio with denominator up to the
configured limit, excluding 1:1. It compares the full scaled phase tracks in
units of the faster candidate's pulse width and requires S/N agreement with
`S_parent / sqrt(p*q)`. Stronger already-flagged children do not become parents.
The stored parent is the stronger representative; it is not a claim about which
observed peak is the physical fundamental. All candidates in a file have the
same DM, so no DM-distance rejection is applied. With no DM discrimination,
inspect harmonic associations before choosing to remove them.

Removal affects the folding list, while `/clusters` retains flagged rows.
The optional count limit is applied last, in decreasing S/N order. Neither
mechanism alters `/detections`.

## Common HDF5 schema 1.0

Root attributes include `schema_version`, `producer`, `algorithm`, `complete`,
`param_names`, parameter-model/order/units descriptions, and counts
`n_detections`, `n_clusters`, `n_candidates`. The initial algorithm is `ffa` and
the model is `kinematic_taylor`.

| Group | Meaning |
| --- | --- |
| `/observation` | Observational attributes; missing optional attributes mean unknown |
| `/processing` | Resolved candidate options |
| `/search` | Original search configuration/attributes, region grids and timer statistics |
| `/detections` | All passing trials, S/N sorted; omitted when raw retention is disabled |
| `/clusters` | All representatives, member counts, relationships and selection decisions |
| `/candidates` | Selected representatives ready for downstream folding |

Candidate tables are columnar. `id` is the representative/detection ID;
`cluster_id` is a zero-based row index into `/clusters`. Cluster IDs are not
renumbered after filtering. IDs/counts use uint64; `parent_cluster_id` uses
int64 with -1 meaning no parent. `source_region` and `source_index` identify an
FFA chunk and its flattened score coordinate. FFA detection IDs encode
`(source_region << 32) | source_index`, independent of buffer flush timing.
`/search/regions/region_NNNN` records each region's parameter limits, grid
counts, scoring widths and fold-bin count. Identity is tied to a particular search plan, not guaranteed across different
grids or chunk plans.

`param_sets` is `(N, len(param_names))` in native Loki parameter order
`[..., jerk, accel, freq]`. Higher derivatives remain in this array.
`param_errors` has the same shape, with validity indicated by
`has_parameter_errors`; FFA currently supplies no parameter uncertainties.
`f0`, `f1`, `f2` are frequency, frequency derivative and second frequency
derivative in Hz, Hz/s and Hz/s². Other columns include `snr`, `width`, `nbins`,
`duty_cycle`, `member_count`, `phase_extent`, `parent_cluster_id`, `harmonic_numerator`,
`harmonic_denominator`, and `selected`. Nonapplicable relationship fields are
zero/-1. On raw detection rows, only `cluster_id` encodes membership; `phase_extent` is zero and cluster
counts, relationships and selection must be read from `/clusters`.

`/observation` records `searched_nsamps`, `original_nsamps`, `tsamp`, `tobs`,
`reference_time_seconds`, source/input identity, and optionally `dm`,
`start_mjd`, `reference_mjd`, `barycentric`. Unknown original sample count is
zero. The reference is the midpoint of the **searched** interval, including
when the CLI truncates the original series to a power of two. Frequency-only
trials have no evolving derivatives, so the midpoint convention is equivalent
to their original frequency reference.

For kinematic acceleration and jerk, the conversion is:

```text
F0 = frequency
F1 = -acceleration * F0 / 299792458
F2 = -jerk * F0 / 299792458
reference_mjd = start_mjd + tobs / (2 * 86400)
```

The final file is atomically renamed into place only after search, clustering,
selection and serialization succeed. `complete=1` is set explicitly before
publication. An abandoned raw writer does not finalize in its destructor.

External S/N sorting uses bounded batches and two-way merging, including when
raw retention is disabled. Temporary files may be as large as several copies
of the raw detections. The graph index (or representative index in anchored mode) has an explicit
memory limit;
exhaustion fails without dropping detections or replacing an existing final
result. The candidate host budget is reserved within the CPU search cap, so it
must be smaller than `max_process_memory_gb`. CUDA planning still treats the
search cap as device memory; candidate processing uses a separate host budget.
With both clustering and harmonic flagging disabled, singleton results stream
directly without allocating a representative index.
Runtime/allocator/HDF5 overhead is additional to the accounted buffers.

## PulsarX conversion

```bash
loki-candidates trial_ffa_results.h5 trial.cands
# Also available without relying on the installed entry point:
python -m loki.candidates trial_ffa_results.h5 trial.cands
```

Output has exactly seven fields per data line:

```text
#id DM accel F0 F1 F2 S/N
```

Rows are S/N sorted, and IDs remain traceable to the HDF5 product. Conversion
uses a disk-backed SQLite sort with a bounded page cache. Optional switches are
`--table candidates|clusters|detections`, `--snr-min`, `--max-candidates`, `--dm`,
`--pepoch`, and `--timing-frame topocentric|barycentric`. A requested detections
table must have been retained. Output replacement is atomic.

The converter prints the `--pepoch` value to pass to either folding tool:

```bash
psrfold_fil2 --candfile trial.cands --pepoch <reported-reference-MJD> \
    --template <fold-template> -f observation.fil -o folded
```

PulsarX ignores comment headers when reading candidates: `#Pepoch` is
informational and does not set the folding epoch. Pass `--pepoch` explicitly,
especially for truncated searches or folding a longer original observation.
`--pepoch` overrides the supplied native reference MJD; it does not propagate
parameters to another epoch.

The default folding frame is topocentric. Barycentric candidates require
explicit `--timing-frame barycentric` and a compatible downstream workflow;
Loki does not convert timing frames. Unknown frames require an explicit choice.
Unknown DM requires `--dm`. Unknown epoch requires `--pepoch` when derivatives
are nonzero. Nonzero terms above jerk cannot be represented by this format and
are rejected rather than silently truncated.

## Migration and EP adapter contract

Existing flat FFA files remain readable through `loki.candidates.CandidateFile`;
missing observational metadata can be supplied during export. The new root
`snr`, `param_sets`, `width` and `nbins` names are transitional hard links to
**selected candidates**. Readers needing all raw detections must now open
`/detections`. They must check the schema rather than assume the old root names
still mean all passing trials.

The shared C++ API is in `loki/search/candidates.hpp`. It accepts normalized
Taylor trials and provides `cluster_candidates` and `phase_track_distance`.
It runs on host memory without an `Exec` parameter. Future EP adapters must
provide the normalized track at the common epoch, original score, winning
boxcar width/bin count, and stable run/leaf identity. EP supplies one trial per
leaf, while FFA supplies all widths. EP's current maximum-over-width output does
not preserve the winning width; later integration must explicitly carry or
recover it. Chebyshev/orbital model normalization and legacy EP file decoding
remain part of that integration, not this FFA change.
