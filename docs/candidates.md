# Candidate products

FFA and candidate-enabled Taylor EP sweeps write versioned common candidate
files, `<prefix>_ffa_results.h5` and `<prefix>_ep_results.h5`. Searches operate
on **one already dedispersed DM trial**. Loki does not perform a DM search or
distill candidates across DMs.

## Switching candidate processing off

Candidate processing is on by default. Turn it off to get the output of
earlier versions:

- TOML: `[candidates] enabled = false`;
- CLI: `loki search ffa --no-candidates` or `loki search ep --no-candidates`;
- Python/C++: `CandidateOptions.enabled = False` (`enabled = false`).

With processing off, nothing is grouped, selected or sorted, and no host
memory is reserved for it: the whole `max_process_memory_gb` goes to the
search. FFA writes the flat detection file `<prefix>_ffa_results.h5` of
earlier versions, one root row per passing trial with the search attributes
at the root. It also carries two additive items, `detection_id` and
`/regions`. This flat file is also the spool candidate processing starts
from, so `FFAFreqSweep.publish_candidates` can publish it later. EP writes
only its run diagnostics, as the constructor without candidates does; Taylor
runs still save their winning `width`. `publish_candidates` refuses options
with processing off.

## EP candidates

Taylor EP sweeps can publish the same common candidate tables after every
independent run has finished and the run files have been merged. CPU runs
remain independent and may execute concurrently. Postprocessing reads their
final leaves across all runs/chunks, applies `snr_min`, then uses the same
phase-track grouping and selection as FFA. Every surviving leaf contributes
one winning boxcar width. There is no expansion into multiple width trials,
and no early suppression of a weaker run's leaves before transitive closure.

The result remains `<prefix>_ep_results.h5`. Existing `/chunks/.../runs/...`
diagnostics and search attributes are preserved alongside `/observation`,
`/processing`, `/search`, `/detections`, `/clusters` and `/candidates`.
`retain_detections = false` removes the common raw table; the original run
diagnostics remain available. Root compatibility arrays refer to selected
candidates, as for FFA.

The candidate score is the existing run `scores` value: maximum boxcar S/N of
the final reintegrated fold. `scores_ep` remains the pruning-score diagnostic.
Candidate options and S/N selection do not change pruning thresholds, survivor
decisions or RNG streams. Width identity is recovered from the final folds
using the same backend scorer; the first configured width wins an exact tie.
Each run now saves this identity in `width`; a chunk saves `scoring_widths`.

This increment supports full-observation polynomial Taylor searches
(acceleration, jerk and snap) with RFI control disabled. Common-product
construction rejects active RFI, Chebyshev and circular-model requests. The
existing diagnostic constructor remains available for other supported searches.
Harvesting and masking are outside this candidate path.

In Python, pass `candidates=CandidateOptions()` to opt in; `candidates=None`
retains the existing diagnostic-only API. In C++, use the overload whose
second argument is `CandidateOptions`. The CLI enables candidate processing
for supported Taylor searches using the existing `[candidates]` table and
the same flags as FFA, including `--no-cluster`. Other CLI models retain
diagnostic output and report that common candidates are unsupported.

For a Taylor search configured in `ep.toml`, with output directory `results`
and prefix `search`, the CLI search and conversion are:

```bash
loki search ep -c ep.toml
loki-candidates results/search_ep_results.h5 results/search.cands
```

Clustering is enabled by default. The converter exports `/candidates`, the
selected representatives, unless a different table is requested. See
[PulsarX conversion](#pulsarx-conversion) for metadata and folding-epoch
requirements.

```python
options = libloki.configs.CandidateOptions()
options.memory_gb = 0.25
sweep = libloki.prune.EPFreqSweep(
    cfg, ref_segs=[2, 4], show_progress=False,
    candidates=options, backend="cuda",
)
sweep.execute(ts_e, ts_v, "results", "search", observation=observation)
```

Completed Taylor runs already report their kinematic parameters at the full
observation midpoint, independent of the starting reference segment. The
adapter normalizes these reported terms to frequency derivatives without
applying a second epoch shift. Native value/scale pairs are retained in the
run diagnostics; their scales are not labeled as statistical uncertainties.
The common `has_parameter_errors` flag remains false.

`/search/sources/run_path` and `chunk_id` map each `source_region` to its
original run. `source_index` is the original final-leaf row, and the uint64
identity is `(source_region << 32) | source_index`. Source ordinals follow
sorted run paths, independent of writer completion order. Both indices have
explicit overflow checks. Every common row retains its original score,
width, bin count and native parameters.

Publication is atomic. If postprocessing fails after the search completes,
the previous product is preserved and completed runs are kept in
`<prefix>_ep_spool.h5`. Retry without searching again:

```python
libloki.prune.EPFreqSweep.publish_candidates(
    "results/search_ep_spool.h5", "results/search_ep_results.h5",
    candidates=options, observation=observation,
)
```

The default retry threshold comes from the spool's saved `snr_min`; an explicit
override affects postprocessing only. Supply the original observation metadata
again, as with FFA retry publication. Incomplete spools are rejected. The
common Python reader and PulsarX converter accept supported EP products with
the same DM/frame/epoch requirements; nonzero derivatives above jerk are
still rejected by PulsarX export.

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
`--no-retain-detections`, `--flag-harmonics` / `--no-flag-harmonics`,
`--remove-harmonics` / `--no-remove-harmonics`,
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
`has_parameter_errors`; neither FFA nor Taylor EP currently supplies statistical
parameter uncertainties.
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

If the search finishes but postprocessing fails (the candidate memory budget,
a full scratch disk, an HDF5 error), the search's detections are kept as
`<prefix>_ffa_spool.h5` beside the intended output and the error is
rethrown. Publish them without searching again, with larger budgets if needed,
passing the same observation metadata as the search:

```python
options = libloki.configs.CandidateOptions()
options.memory_gb = 8.0
libloki.ffa.FFAFreqSweep.publish_candidates(
    "out/run_ffa_spool.h5", "out/run_ffa_results.h5",
    candidates=options, observation=observation,
)
```

(C++: `pipelines::FFAFreqSweep::publish_candidates`.) A search that fails
part-way leaves nothing behind.

External S/N sorting uses bounded batches and two-way merging, including when
raw retention is disabled. Each sorted record is a full 184-byte `Candidate`,
and the two-way merge rewrites every record about log2(N / 65536) times for N
detections. Plan the scratch disk (`temporary_directory`, by default beside
the output) for a peak of roughly the raw spool + 2 x 184 B x N + the final
file: about 37 GB for 10^8 detections, with several hundred GB written over
the run. The graph index (or representative index in anchored mode) has an explicit
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
# The same converter accepts a common Taylor EP product:
loki-candidates trial_ep_results.h5 trial_ep.cands
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

## Migration and shared candidate contract

### Behaviour changes for existing FFA users

- **Default `snr_min` is 7, was 6** (C++ `FFASearchConfig`, Python, both TOML
  templates). It is the FFA detection threshold, so a search relying on the
  default writes fewer detections. Keep the old behaviour with
  `[search] snr_min = 6.0`. Candidate-enabled Taylor EP applies `snr_min`
  to final reintegrated scores after all runs finish. Its pruning uses the
  independent `target_snr`, which is unchanged. Diagnostic-only EP output
  retains all final leaves without this candidate filter.
- **Root datasets now mean the selected candidates.** `snr`, `param_sets`,
  `width` and `nbins` are hard links to `/candidates`: one row per selected
  representative, sorted by S/N, not one row per passing trial. Every passing
  trial is in `/detections` (when `retain_detections`).
- **Types and names.** `width` and `nbins` are `uint32` (were `uint16`). Root
  `detection_id` is gone: use `/detections/id` and `/candidates/id`. The search
  attributes (`ffa_version`, `config_toml`, `tsamp`, `nsamps`, `tobs`, `f_min`,
  `f_max`, `snr_min`, `ducy_max`, `wtsp`, `eta`, `bseg_*`, ...) moved to the
  `/search` group; the root keeps `param_names`, `complete` and the schema
  attributes.
- **CPU memory.** The candidate budget `candidates.memory_gb` (default 1 GB) is
  reserved inside `max_process_memory_gb`. A CPU search with
  `max_process_memory_gb <= memory_gb` now fails at construction; lower
  `[candidates] memory_gb` or raise the search cap. Candidate-enabled Taylor
  EP uses the same reservation. CUDA keeps its device search cap and a
  separate host candidate budget.

### Reading old files

Existing flat FFA files remain readable through `loki.candidates.CandidateFile`;
missing observational metadata can be supplied during export. Readers must
check `schema_version` rather than assume the root names still mean all
passing trials.

Legacy EP files without saved winning widths are not accepted as common
candidate products. Width identity is required for duty-aware grouping and
cannot be inferred from a maximum-over-width score alone. Completed current
Taylor spools can be published using `EPFreqSweep.publish_candidates`.

### Shared candidate contract

The shared C++ API is in `loki/search/candidates.hpp`. It accepts normalized
Taylor trials and provides `cluster_candidates` and `phase_track_distance`.
It runs on host memory without an `Exec` parameter. Both adapters provide
midpoint-referenced frequency derivatives, original scores and native
parameters, boxcar width/bin count, and stable source identity. FFA supplies
every passing width trial; Taylor EP supplies one winning-width trial per
final leaf across completed runs. Taylor EP recovers that width from final
folds without overwriting reported scores or changing pruning decisions.

Common EP candidates currently require an inactive RFI configuration and
polynomial Taylor models through snap. Chebyshev/orbital normalization and
RFI harvest/masking integration are outside this path. PulsarX export supports
terms through jerk and rejects nonzero higher derivatives.
