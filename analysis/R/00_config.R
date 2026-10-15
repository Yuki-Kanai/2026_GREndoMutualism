## ---------------------------------------------------------------------------
## GRHD reproduction -- analysis layer (R)
## 00_config.R : every path and every constant lives here.
##
## Paths default to the repository layout (<repo>/analysis, <repo>/simulation)
## and can be overridden with the environment variables GRHD_ANA_DIR,
## GRHD_SIM_DIR and GRHD_RAW_DIR.
##
## The simulation directory is treated as READ-ONLY.  Nothing in this analysis
## ever writes into it.
## ---------------------------------------------------------------------------

## The directory holding this file: notebooks set R_DIR, R/10_build_cache.R
## sets `here`, and both source this file from there.
.cfg_dir <- if (exists("R_DIR")) R_DIR else if (exists("here")) here else "R"

ANA_DIR <- Sys.getenv(
  "GRHD_ANA_DIR",
  normalizePath(file.path(.cfg_dir, ".."), mustWork = FALSE)
)
SIM_DIR <- Sys.getenv(
  "GRHD_SIM_DIR",
  normalizePath(file.path(ANA_DIR, "..", "simulation"), mustWork = FALSE)
)

## Simulation output: one parquet per run (<run>_cells.parquet /
## <run>_vectors.parquet), produced from the per-block files under data/raw by
## simulation/scripts/consolidate_parquet.py (plain row concatenation).  Not
## part of the repository; see the top-level README for how to obtain it.
RAW_DIR     <- Sys.getenv("GRHD_RAW_DIR", file.path(SIM_DIR, "data", "consolidated"))
SIM_TAB_DIR <- file.path(SIM_DIR, "tables")           # the Python pipeline's tables
SIM_CFG_DIR <- file.path(SIM_DIR, "config")

CACHE_DIR <- file.path(ANA_DIR, "cache")              # tidy intermediates (this repo)
FIG_DIR   <- file.path(ANA_DIR, "figures")
TAB_DIR   <- file.path(ANA_DIR, "tables")

for (d in c(CACHE_DIR, FIG_DIR, TAB_DIR)) {
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
}

## --- model constants (mirror of <SIM_DIR>/config/base.yaml) -----------------
## Kept here only so that figures can draw reference lines.  Every *measured*
## quantity is read from the data, never from this block.
M_METABOLITES <- 10L        # m
K_REACTIONS   <- 4L         # k
R_H_DEFAULT   <- 24L        # r^H of the single host used in Fig.3-Fig.5
MASTER_SEED   <- 20260916L

## --- analysis constants (mirror of base.yaml, "stats" and "tolerances") ----
BOOT_B       <- 2000L                  # bootstrap replicates
SCATTER_N    <- 300L                   # scatter subsample per r
HDR_LEVELS   <- c(25, 50, 75)          # HDR contour masses (%)
KDE_GRID     <- 256L                   # HDR integration grid
KDE_BW_FIG3C <- 2.0e-3                 # fixed isotropic bandwidth, Fig.3C
KDE_BW_FIG4C <- 0.2                    # fixed isotropic bandwidth, Fig.4C
## How the two numbers above are read.  "mass_h" is the MASS::kde2d /
## ggdensity convention, where the bandwidth spans FOUR standard deviations
## (sigma = bw / 4); "sigma" treats them as the standard deviation itself,
## which is what the Python pipeline did and which makes the HDR bands four
## times wider.  The sigma actually used is recorded in every figure's
## .meta.json.
KDE_BW_CONVENTION <- "mass_h"
HDR_R_VALUES <- c(5L, 10L, 20L, 40L)   # the four panels of Fig.3C / Fig.4C
TOL_ABS      <- 1.0e-8                 # saturation test: |mu_S - |c^H|_1| <=
TOL_REL      <- 1.0e-6                 #   TOL_ABS + TOL_REL * |c^H|_1
TRIM_PROP    <- 0.05                   # two-sided trimmed mean

## How a Greek VARIABLE (mu, alpha, ...) is drawn in a label.  R's plotmath
## takes `mu` from the Adobe Symbol font, which has no italic face, so
## italic(mu) is a silent no-op -- hence the two alternatives below.
##
##   "face"  the literal Greek code point (U+03BC) wrapped in italic(), so the
##           LABEL FONT'S OWN ITALIC FACE draws it.  Sans-serif, right weight,
##           right superscript position.  Needs a family that has an italic
##           face and covers Greek; Arial, Liberation Sans, Nimbus Sans, Lato
##           and FreeSans all do.  THIS IS THE DEFAULT.
##   "math"  the Unicode MATHEMATICAL ITALIC block (U+1D6FC..U+1D71B).  No sans
##           font here carries it, so the glyph falls back to a SERIF maths
##           face and sits low in an otherwise sans label.  This is what made
##           the Greek look serif; kept only as an escape hatch.
##   "none"  upright Greek from the plotmath symbol font.
##
## Overridable from the shell with GRHD_GREEK=math (etc.).
GREEK_MODE <- Sys.getenv("GRHD_GREEK", "face")

## --- design grids ----------------------------------------------------------
## Pre-evolution (Fig.3, Fig.5D/5E, Fig.6A/6B): r = 1..100, N_S = 300, N_c = 100.
PRE_R_GRID <- 1:100

## Post-evolution (Fig.4, Fig.5A).  The original MILP design grid had 15 values
## of r with N_S = 300 each; the grid is still being filled in, so more values
## of r (with their own, smaller N_S) keep appearing in data/raw/post_paper.
##
##   POST_R_SELECTION = "all"          every r present on disk (default).
##                                     Sample size per r is unequal and is
##                                     printed and plotted, never hidden.
##                      "design_grid"  only the 15 original values, i.e. the
##                                     frozen set the first analysis used.
##
## After the simulation adds blocks, refresh the cache with
##   Rscript R/10_build_cache.R --parts post,r5 --force
POST_R_GRID      <- c(1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 60, 80, 100)
POST_R_SELECTION <- "all"

## Apply the selection above to a post-evolution table.
post_select <- function(d) {
  if (identical(POST_R_SELECTION, "design_grid")) d[d$r %in% POST_R_GRID, ] else d
}

## Fig.5D / 5E follow the paper and restrict to 5 <= r <= 25; the all-r version
## is produced as well.
R3_R_RANGE <- c(5L, 25L)

## --- Fig.6C (R5): the two host designs -------------------------------------
## A : r^H = round(2.4 m)   -- host ability held fixed relative to m (control)
## B : r^H = 24 (all m)     -- the setting of the original paper
## The two coincide at m = 10, where the block directory is the main run.
R5_N_HOSTS <- c(A = 5L, B = 10L)

r_H_of <- function(design, m) {
  ifelse(design == "A", as.integer(round(2.4 * m)), 24L)
}

## Directory holding the post-evolution cells of one (design, m, host).
r5_block_dir <- function(design, m, host) {
  rH <- r_H_of(design, m)
  if (m == 10L && rH == 24L) {
    return(file.path(RAW_DIR, if (host == 0L) "post_paper" else sprintf("post_paper_h%d", host)))
  }
  tag <- sprintf("post_paper_m%d_rH%d", m, rH)
  if (host != 0L) tag <- sprintf("%s_h%d", tag, host)
  file.path(RAW_DIR, tag)
}
