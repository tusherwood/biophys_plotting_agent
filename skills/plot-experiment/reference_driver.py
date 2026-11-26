"""
Reference plotting driver for a compiled/annotated biophysics experiment.

This is the TEMPLATE the plot-experiment skill adapts per experiment. When generating a driver:
  - copy biophys_plot_toolkit.py next to this driver in the analysis directory (so the import
    below resolves locally and the analysis stays self-contained / reproducible),
  - set COMPILED_DIR / COULTER_DIR and FIG_DIR for this experiment,
  - review the printed plan and set ROLE_OVERRIDES for any [CONFIRM] columns.

The toolkit reads whatever hand-added annotation columns exist and classifies each into a role
(boolean / categorical / time / ordered / continuous / label / structural). `build_plan` proposes
plots from those roles; `autoplot` runs them. Every column becomes a grouping/comparison axis; a
time column is used to order samples sequentially. Use the explicit combinators (see the bottom of
main) for full control.

Run it with the same Python env that has: numpy, pandas, matplotlib, openpyxl, python-pptx, Pillow,
scipy (p-values; skipped with a warning without it) (+ pyarrow, and vqvae_cache_io.py beside the toolkit, when CROP_CACHE is on).
"""
from pathlib import Path

import biophys_plot_toolkit as tk

# ---------------------------------------------------------------------------
# Per-experiment configuration  (EDIT THESE)
# ---------------------------------------------------------------------------
# The loaders read the raw biophys_helpers outputs directly (no reorg step). Point each at the
# timestamped output dir it needs (set one to None if that half of the experiment doesn't exist):
#   iFXM   -> compile_experiment.py's '<...>_compiled/' dir      (holds experiment_data.xlsx)
#   Coulter-> annotate_coulter_samples.py's '<...>_coulter_sample_annotation/' dir
# EXP_NAME names the deck and MUST keep the analysis dir's date prefix
# (e.g. '2026-09-22_fl5_wnki_conc-curves'). The driver lives in that dir, so derive it; if the dir
# has no YYYY-MM-DD / YYYYMMDD prefix, hardcode the dated name instead.
EXP_NAME     = Path(__file__).resolve().parent.name
COMPILED_DIR = Path(r"C:\path\to\<...>_compiled")                    # iFXM (or None)
COULTER_DIR  = Path(r"C:\path\to\<...>_coulter_sample_annotation")   # Coulter (or None)
# COMPILED_DIR = Path("/Users/you/.../<...>_compiled")              # macOS
# COULTER_DIR  = Path("/Users/you/.../<...>_coulter_sample_annotation")

OUT_ROOT = COMPILED_DIR.parent if COMPILED_DIR else COULTER_DIR.parent
FIG_DIR  = OUT_ROOT / f"{EXP_NAME}_fig"
PPTX_OUT = OUT_ROOT / f"{EXP_NAME}_figures.pptx"
# Default second deck: per-sample per-cell props vs run time (drift within each run), kept in its
# own fig dir so it doesn't mix into the main deck. iFXM paired samples only.
TIME_FIG_DIR  = OUT_ROOT / f"{EXP_NAME}_vs_time_fig"
TIME_PPTX_OUT = OUT_ROOT / f"{EXP_NAME}_props_vs_time.pptx"
# Default third deck: QC — cell-count waterfall, per-sample medians in acquisition order (drift of
# repeated conditions), and pairing-confidence ridges / scatters. Own fig dir, never the main deck.
QC_FIG_DIR  = OUT_ROOT / f"{EXP_NAME}_qc_fig"
QC_PPTX_OUT = OUT_ROOT / f"{EXP_NAME}_qc.pptx"
# Column giving the acquisition order for the drift plot (e.g. a run index / timestamp), or None
# to use the metadata row order.
QC_ORDER_COL = None
# Default export (iFXM): the plotted cells (gated + post-outlier-rejection) as an xlsx in
# experiment_data.xlsx's layout — one sheet per sample, mass / volume / density columns only.
PROCESSED_XLSX = OUT_ROOT / "processed_data" / f"{EXP_NAME}_processed.xlsx"
# Optional (Claude asks each run): the compiled VQ-VAE crop cache cropped to the same cells, beside
# PROCESSED_XLSX (+ <stem>_metadata.parquet). Needs vqvae_cache_io.py copied next to the toolkit.
CROP_CACHE      = False
PROCESSED_CACHE = OUT_ROOT / "processed_data" / f"{EXP_NAME}_vqvae_cache.pt"
# Claude writes OUT_ROOT / f"{EXP_NAME}_analysis_notes.md" next to the deck (data structure +
# plotting choices) so a future dataset can be analyzed the same way.

# Paired runs use the matched pair_ block (pair_mass_pg, pair_volume_fl, pair_cell_density_g_per_mL
# — density is already absolute, computed by the hdf5 pipeline itself); mass-only / volume-only
# runs fall back to the standalone mass_/vol_ blocks automatically, so they plot too (density and
# scatter need pairing and are skipped for those samples).

# Axis labels for each property (also selects which props to plot). Trim to what exists.
IFXM_LABELS    = dict(tk.IFXM_PROPS)        # mass / density / vol
COULTER_LABELS = dict(tk.COULTER_PROPS)     # volume

# 2-D per-cell scatters with marginal histograms (iFXM only). (prop_x, prop_y, xlabel, ylabel)
SCATTER_PAIRS = [
    ("mass", "density", "Buoyant mass (pg)", "Total density (g/cm^3)"),
    ("vol",  "mass",    "Volume (fL)",       "Buoyant mass (pg)"),
    ("vol",  "density", "Volume (fL)",       "Total density (g/cm^3)"),
]

# Pin any column's role after reviewing the printed plan (resolves every [CONFIRM] flag).
# e.g. {"dose_uM": "ordered", "plate": "categorical"}. Leave empty to accept the inferred roles.
ROLE_OVERRIDES = {}

# Grid-search heatmaps (two columns varied jointly, e.g. osmolarity x drug dose). EMPTY by default:
# render_plan lists any grid-like column pairs as suggestions — add one here only after the user
# confirms. Each entry is (x_col, y_col), or {"cols": (x_col, y_col), "show_repeats": True} to also
# show repeated control samples individually beside the grid (the grid cell shows their mean).
GRID_PAIRS = []

# Statistics: the CONFIRMED reference (control) value per grouping column, e.g.
# {"drug_name": "dmso", "dose_uM": 0}. Each entry adds, per prop: stats on that column's
# replicate_summary and compare_groups box, a fold_change forest (% change in median vs the
# reference, 95% hierarchical-bootstrap CI) and a quantile_shift plot; tables go to
# FIG_DIR/stats/*.csv. Comparisons are made within strata (other crossed columns, e.g. time).
# render_plan lists suggested references — add one here only after the user confirms it.
REFERENCES = {}

# Split cells by a per-cell annotation (only if the user asked; e.g. the sickle_annotations sheet
# written by annotate_sickle_cells.py): {"sheet": "sickle_annotations", "col": "sickle_class",
# "reference": "normal"} ("reference" optional — adds stats vs that class). None = off.
CELL_SPLIT = None


def main() -> None:
    print(f"=== plotting {EXP_NAME} ===")
    coulter = tk.load_coulter(COULTER_DIR) if COULTER_DIR else []
    ifxm = ifxm_paired = []
    if COMPILED_DIR:
        # Cells arrive already gated by compile_experiment.py (AND of every gate; older xlsx files
        # are gated by the loaders instead), so a paired sample has the same cells in both.
        ifxm = tk.load_ifxm(COMPILED_DIR)
        # row-aligned variant for scatter / props-vs-time (same cells, plus per-cell time_h)
        ifxm_paired = tk.load_ifxm_paired(COMPILED_DIR)
    print(f"  coulter samples: {len(coulter)} | ifxm samples: {len(ifxm)}")

    # --- Outlier rejection (Claude asks each run; data is loaded verbatim otherwise) -------
    # Recommended default: Tukey fences, per sample, on mass/density/vol (+ coulter volume),
    # JOINT across props (default joint=True): a paired cell outside ANY prop's fence is dropped
    # from every plot, so all of a sample's plots show the same cells. Single pass — fences come
    # from the untrimmed cells. Unpaired mass-only / volume-only samples are trimmed on their one
    # prop. Apply the same spec to both iFXM record sets (they hold identical paired cells, so
    # they get identical masks). method 'mad'|'iqr'|'percentile' (per-prop dict allowed);
    # scope 'per_sample'|'pooled'; log=True for log-normal mass/volume. Delete if declined.
    #   OUTLIER_SPEC = dict(method="iqr", props=["mass", "density", "vol"], scope="per_sample")
    #   ifxm        = tk.reject_outliers(ifxm, **OUTLIER_SPEC)
    #   ifxm_paired = tk.reject_outliers(ifxm_paired, **OUTLIER_SPEC)
    #   coulter     = tk.reject_outliers(coulter, method="iqr", props=["volume"], scope="per_sample")

    # --- Processed-data export (default, iFXM): exactly the cells the plots show --------------
    # One sheet per sample (same sheet names), pair_cell_id + pair_mass_pg / pair_volume_fl /
    # pair_cell_density_g_per_mL (unpaired: mass_mass_pg / vol_cell_id + vol_volume_fl), plus
    # metadata + README sheets. Pass the load_ifxm records after reject_outliers. Reloads with
    # tk.load_ifxm(PROCESSED_XLSX), the file path, not its folder.
    if ifxm:
        tk.export_processed(ifxm, COMPILED_DIR, PROCESSED_XLSX)
    # --- Cropped crop cache (if the user said yes): the images of exactly those cells ---------
    # Same records as export_processed. Writes PROCESSED_CACHE + its _metadata.parquet, the pair
    # VAEImagingAnalysis' run_pipeline.py trains on (--data <processed_data dir>).
    if ifxm and CROP_CACHE:
        tk.export_processed_cache(ifxm, COMPILED_DIR, PROCESSED_CACHE)

    # --- iFXM: inspect the annotation schema, show the plan, run it -----------------------
    if ifxm:
        roles = tk.infer_roles(ifxm, overrides=ROLE_OVERRIDES)
        plan = tk.build_plan(ifxm, "ifxm", roles=roles, props=list(IFXM_LABELS),
                             scatter_pairs=SCATTER_PAIRS, grid_pairs=GRID_PAIRS,
                             references=REFERENCES)
        print(tk.render_plan(plan))       # review; set ROLE_OVERRIDES for any [CONFIRM] columns
        tk.autoplot(ifxm, plan, "ifxm", FIG_DIR, prop_labels=IFXM_LABELS,
                    paired_records=ifxm_paired)

    # --- Per-cell split (CELL_SPLIT): each sample split by a per-cell class, e.g. sickle ------
    # Joined on pair_cell_id <-> cell_id_source per sample (data_schema.md "Per-cell ids"); paired
    # samples only. The split column then groups / is compared like any annotation column, within
    # strata of the sample-level columns (e.g. sickle vs normal at each drug x time).
    if ifxm and CELL_SPLIT:
        col = CELL_SPLIT["col"]
        ann = tk.attach_cell_annotations(ifxm, COMPILED_DIR, sheet=CELL_SPLIT["sheet"],
                                         columns=[col])
        split = tk.split_by_cell_col(ann, col)
        sroles = tk.infer_roles(split, overrides=ROLE_OVERRIDES)
        sref = CELL_SPLIT.get("reference")
        for prop, label in IFXM_LABELS.items():
            tk.compare_groups(split, prop, label, f"ifxm-{col}", FIG_DIR, group_col=col,
                              roles=sroles, stats_ref=sref)
            tk.replicate_summary(split, prop, label, f"ifxm-{col}", FIG_DIR, group_col=col,
                                 roles=sroles, reference=sref)
            if sref is not None:
                tk.fold_change(split, prop, label, f"ifxm-{col}", FIG_DIR, group_col=col,
                               roles=sroles, reference=sref)

    # --- Coulter --------------------------------------------------------------------------
    if coulter:
        croles = tk.infer_roles(coulter, overrides=ROLE_OVERRIDES)
        cplan = tk.build_plan(coulter, "coulter", roles=croles, props=list(COULTER_LABELS),
                              grid_pairs=GRID_PAIRS, references=REFERENCES)
        print(tk.render_plan(cplan))      # includes the per-sample cell-count table
        tk.autoplot(coulter, cplan, "coulter", FIG_DIR, prop_labels=COULTER_LABELS)

    # --- Explicit combinators (comment out autoplot above and drive it yourself) ----------
    # Each groups/compares/orders by ANY column name; roles= is optional (inferred if omitted).
    #   tk.plot_grouped(ifxm, "mass", "Buoyant mass (pg)", "ifxm", FIG_DIR, group_col="is_activated")
    #   tk.compare_groups(ifxm, "mass", "Buoyant mass (pg)", "ifxm", FIG_DIR, group_col="is_activated")
    #   tk.timecourse_by(ifxm, "mass", "Buoyant mass (pg)", "ifxm", FIG_DIR, series_col="is_activated")
    #   tk.scatter_by(ifxm_paired, "mass", "density", "Mass", "Density", "ifxm", FIG_DIR,
    #                 group_col="is_activated")
    #   tk.facet(ifxm, "mass", "Buoyant mass (pg)", "ifxm", FIG_DIR, facet_col="media")
    #   tk.cross_groups(ifxm, "mass", "Buoyant mass (pg)", "ifxm", FIG_DIR,
    #                   cols=("is_activated", "media"))   # crossing on request
    #   tk.grid_heatmap(ifxm, "density", "Density (g/mL)", "ifxm", FIG_DIR,
    #                   cols=("osm_mOsmL", "dose_uM"), show_repeats=True)   # grid search, confirmed
    #   tk.replicate_summary(ifxm, "mass", "Buoyant mass (pg)", "ifxm", FIG_DIR,
    #                        group_col="drug_name", reference="dmso")       # dots = sample medians
    #   tk.fold_change(ifxm, "mass", "Buoyant mass (pg)", "ifxm", FIG_DIR,
    #                  group_col="drug_name", reference="dmso")            # forest, 95% CI
    #   tk.quantile_shift(ifxm, "mass", "Buoyant mass (pg)", "ifxm", FIG_DIR,
    #                     group_col="drug_name", reference="dmso")         # decile shift vs ref
    #   tk.stats_table(tk.compare_to_reference(ifxm, "mass", "drug_name", "dmso"))  # the numbers
    #
    # Hand-rolled figures: legends must not cover the data. Draw the data first, then
    #   tk.place_legend(ax, handles=[...], loc="upper right")   # moves itself if that spot is busy
    #   tk._save(fig, "my_fig.png", FIG_DIR)                    # also re-checks every legend

    # First slide(s): per-sample cell counts before gating / after gating / after outlier rejection,
    # from the same records the ridges and boxes used. Copy the printed tables into the notes.
    count_tables = {label: recs for label, recs in (("iFXM", ifxm), ("Coulter", coulter)) if recs}
    for label, recs in count_tables.items():
        print(tk.render_cell_counts(recs, title=f"Cells per sample — {label}"))
    tk.save_pptx(FIG_DIR, PPTX_OUT, count_tables=count_tables)

    # --- Default second deck: per-cell mass/vol/density vs run time, one slide per sample ---
    # Every paired sample; trend 'binned_median' (bins=) | 'rolling_mean' | 'rolling_median'
    # (window=) | None. Narrow with samples=[...] / props=(...) if the user asks.
    if ifxm_paired:
        tk.props_vs_time(ifxm_paired, "ifxm", TIME_FIG_DIR,
                         props=[p for p in ("mass", "vol", "density") if p in IFXM_LABELS],
                         prop_labels=IFXM_LABELS, trend="binned_median")
        tk.save_pptx(TIME_FIG_DIR, TIME_PPTX_OUT)

    # --- Default third deck: QC (cell-count waterfall, acquisition-order drift, pairing
    # confidence). Same records as the main deck (after rejection); pairing confidence needs the
    # load_ifxm_paired records.
    if ifxm:
        tk.qc_report(ifxm, "ifxm", QC_FIG_DIR, paired_records=ifxm_paired,
                     props=list(IFXM_LABELS), prop_labels=IFXM_LABELS, roles=roles,
                     order_col=QC_ORDER_COL)
    if coulter:
        tk.qc_report(coulter, "coulter", QC_FIG_DIR, props=list(COULTER_LABELS),
                     prop_labels=COULTER_LABELS, roles=croles, order_col=QC_ORDER_COL)
    tk.save_pptx(QC_FIG_DIR, QC_PPTX_OUT)


if __name__ == "__main__":
    main()
