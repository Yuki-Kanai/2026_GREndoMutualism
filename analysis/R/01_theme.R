## ---------------------------------------------------------------------------
## 01_theme.R : one look for every figure, and one way to save them.
##
## The palette and the panel grammar follow the manuscript figures, which use
## the Okabe-Ito colour-blind safe palette throughout:
##
##   cells / symbionts        black, low alpha, RASTERISED
##   mean (or median) curve   vermillion  #D55E00      ("Mean" in the legend)
##   bounded metabolites      sky blue    #56B4E9
##   excreted metabolites     vermillion  #D55E00
##   reference lines          grey        (mu^H_0, r = m-1)
##   HDR probability bands    greys, 75 % darkest, over vermillion points
##   r gradient (Fig.5A)      viridis
##
## Every figure is saved three times:
##   <name>.pdf        vector, for the manuscript
##   <name>.png        raster preview (dpi = FIG_DPI)
##   <name>.meta.json  provenance: which cache files and which constants it used
##
## Panels with tens of thousands of points rasterise their point layer, so the
## PDF stays small and opens instantly.  Use raster_points() / points_cells(),
## never a bare geom_point(), whenever a layer can carry more than a few
## thousand points.
## ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
  library(jsonlite)
})

## Okabe-Ito -----------------------------------------------------------------
cbp <- c(grey = "#999999", orange = "#E69F00", skyblue = "#56B4E9",
         green = "#009E73", yellow = "#F0E442", blue = "#0072B2",
         vermillion = "#D55E00", purple = "#CC79A7")

## Semantic roles, so a colour change happens in one place.
COL <- list(
  point     = "black",                     # individual cells / symbionts
  mean      = unname(cbp["vermillion"]),   # the "Mean" curve of the manuscript
  mean2     = unname(cbp["blue"]),         # a second curve when two are compared
  bounded   = unname(cbp["blue"]),         # Fig.6B, bounded metabolites
  excreted  = unname(cbp["vermillion"]),   # Fig.6B, excreted metabolites
  accent    = unname(cbp["green"]),
  purple    = unname(cbp["purple"]),
  ref       = "grey55",                    # mu^H_0, r = m-1, (m-1)/2
  hdr_point = unname(cbp["vermillion"]),   # points under the HDR bands
  hdr_fill  = c(`25` = "#E3E3E3", `50` = "#AFAFAF", `75` = "#6E6E6E"),
  hdr_line  = "black"
)

## ---------------------------------------------------------------------------
## Font
## ---------------------------------------------------------------------------
## This matters more than it looks.  The graphics device's DEFAULT sans family
## has no italic face, so `italic(r)` silently renders upright -- every maths
## variable comes out roman.  A family with a real italic face fixes it.
## The family also has to carry Greek in that italic face, because that is how
## mu^H is drawn (see GREEK_MODE below).
## Override with GRHD_FONT="Nimbus Sans" (or whatever is installed).
## ---------------------------------------------------------------------------
## Italic Greek
## ---------------------------------------------------------------------------
## R's plotmath draws `mu` from the Adobe Symbol font, which has no italic
## face, so italic(mu) renders identically to mu -- every Greek variable comes
## out upright.  Two ways round it, selected by GREEK_MODE in R/00_config.R:
##
##   "face"  emit the LITERAL Greek code point (U+03BC) inside italic().  It is
##           then ordinary text, so the label family's own italic face draws
##           it: sans-serif, matching weight, superscript in the right place.
##           Every sans family installed here (Liberation Sans, Nimbus Sans,
##           Lato, FreeSans, DejaVu Sans) carries Greek in all four faces.
##   "math"  the Unicode MATHEMATICAL ITALIC block (U+1D6FC..U+1D71B), whose
##           code points ARE italic letters.  Only DejaVu Serif / FreeSerif /
##           DejaVu Math TeX Gyre carry that block here, so the glyph falls
##           back to a SERIF face and sits low -- this is what made mu look
##           serif.  Kept only for a machine with no italic Greek at all.
##   "none"  upright Greek.
GREEK_LETTER <- c(
  mu = "μ", alpha = "α", rho = "ρ",
  beta = "β", gamma = "γ", lambda = "λ",
  kappa = "κ", sigma = "σ", tau = "τ",
  phi = "ϕ", omega = "ω", theta = "θ")
GREEK_MATH_ITALIC <- c(
  mu = "\U0001d707", alpha = "\U0001d6fc", rho = "\U0001d70c",
  beta = "\U0001d6fd", gamma = "\U0001d6fe", lambda = "\U0001d706",
  kappa = "\U0001d705", sigma = "\U0001d70e", tau = "\U0001d70f",
  phi = "\U0001d719", omega = "\U0001d714", theta = "\U0001d703")
## Delta is deliberately NOT here: the difference operator is upright by
## convention (ISO 80000-2), and so are the label letters H and S.

## Everything below asks fontconfig what is installed.  Windows has no
## fc-list, and there the probe would answer "nothing is installed" for every
## family and silently downgrade the Greek -- so on a machine without
## fontconfig we stop probing and trust the request instead (see below).
HAVE_FC <- nzchar(Sys.which("fc-list"))
.fc <- function(pattern, prop = "family") {
  if (!HAVE_FC) return(character(0))
  tryCatch(suppressWarnings(
    system2("fc-list", c(shQuote(pattern), prop),
            stdout = TRUE, stderr = FALSE)), error = function(e) character(0))
}
font_available <- function(family) length(.fc(paste0(":family=", family))) > 0
## Does this family have an ITALIC (or oblique) face that ALSO carries Greek
## mu?  That is the whole requirement for GREEK_MODE = "face".
font_italic_greek <- function(family) {
  st <- .fc(paste0(":family=", family, ":charset=3bc"), "style")
  any(grepl("Italic|Oblique", st))
}
## Is a code point available in ANY installed font (i.e. reachable by fallback)?
charset_available <- function(cp) length(.fc(paste0(":charset=", cp))) > 0

## Preference order for the label font.  Arial first because that is what the
## figures are meant to look like; Liberation Sans and Arimo are metrically
## identical to Arial and are what is actually installed on Linux.  Nimbus Sans
## and Helvetica are the Helvetica metrics PNAS also accepts -- put one of them
## first, or set GRHD_FONT, to switch the whole figure set over in one line.
FONT_PREFS <- c("Arial", "Liberation Sans", "Arimo", "Helvetica",
                "Nimbus Sans", "FreeSans", "Lato", "DejaVu Sans")
pick_font <- function(prefs = FONT_PREFS, need_italic_greek = FALSE) {
  for (f in prefs) {
    if (!font_available(f)) next
    if (need_italic_greek && !font_italic_greek(f)) next
    return(f)
  }
  ""   # give up: the device default, and italics will not render
}

FONT_FAMILY <- Sys.getenv("GRHD_FONT", "")
if (!nzchar(FONT_FAMILY) && !HAVE_FC) {
  ## No fontconfig: Windows, where the real Arial is installed and R's device
  ## finds it by name.  Nothing to probe, so take the first preference.
  FONT_FAMILY <- if (.Platform$OS.type == "windows") "Arial" else ""
}
if (!nzchar(FONT_FAMILY) && HAVE_FC) {
  FONT_FAMILY <- if (identical(GREEK_MODE, "math")) {
    ## The maths-italic block is only in the DejaVu/Free SERIF faces, and only
    ## the DejaVu family keeps the superscript metrics right on the fallback.
    pick_font(c("DejaVu Sans", "FreeSerif", "DejaVu Serif"))
  } else {
    pick_font(need_italic_greek = identical(GREEK_MODE, "face"))
  }
}
if (!nzchar(FONT_FAMILY)) {
  warning("no font with an italic face was found; italic() will render upright")
}

## Resolve the requested Greek mode against what the machine can actually do.
## Without fontconfig there is nothing to check, so the request stands: a real
## Arial has italic Greek, and answering "no" on a failed probe would quietly
## put the labels back in the serif maths face.
GREEK_MODE_USED <- GREEK_MODE
if (HAVE_FC && identical(GREEK_MODE, "face") && !font_italic_greek(FONT_FAMILY)) {
  GREEK_MODE_USED <- if (charset_available("1d707")) "math" else "none"
  warning(sprintf("%s has no italic Greek; GREEK_MODE falls back to '%s'",
                  FONT_FAMILY, GREEK_MODE_USED))
}
if (HAVE_FC && identical(GREEK_MODE, "math") && !charset_available("1d707")) {
  GREEK_MODE_USED <- "none"
  warning("no font carries U+1D707; falling back to upright Greek")
}
## Kept so older code and the .meta.json reader keep working.
GREEK_ITALIC_OK <- GREEK_MODE_USED %in% c("face", "math")

## mx() is expression(), with every Greek variable made italic.  Write plotmath
## as usual -- mu^H, alpha, rho -- and wrap it in mx() instead of expression().
## Under "face" the symbol becomes italic("<greek letter>") so the label font's
## italic face draws it; under "math" it becomes a maths-italic code point;
## under "none" mx() is exactly expression().
.greekify <- function(x, mode) {
  if (is.symbol(x)) {
    nm <- as.character(x)
    ## single-bracket: [[ ]] errors on an absent name, [ ] returns NA
    if (mode == "face") {
      g <- GREEK_LETTER[nm]
      return(if (is.na(g)) x else call("italic", unname(g)))
    }
    g <- GREEK_MATH_ITALIC[nm]
    return(if (is.na(g)) x else unname(g))
  }
  if (is.call(x)) {
    for (i in seq_along(x)) x[[i]] <- .greekify(x[[i]], mode)
    return(x)
  }
  x
}
mx <- function(...) {
  es <- as.list(substitute(list(...)))[-1]
  if (identical(GREEK_MODE_USED, "none")) return(as.expression(es))
  as.expression(lapply(es, .greekify, mode = GREEK_MODE_USED))
}

FIG_DPI     <- 1000    # the png preview
RASTER_DPI  <- 1000    # the rasterised point layers inside the pdf

## Below this many points, rasterising only blurs a layer that was already a
## small, fast vector PDF -- Fig.3C/4C's four 300-cell HDR panels are the
## case this was written for. At or above it (everything else here is in the
## thousands to tens of thousands), ggrastr keeps the file openable. Checked
## with `nrow(data)` at construction time, so it is per PANEL, not per figure
## (a facet's own point count, not the unfiltered table behind it).
RASTER_MIN_N <- 500

## ---------------------------------------------------------------------------
## Type size
## ---------------------------------------------------------------------------
## PNAS asks for 6-12 pt lettering in figures.  BASE_PT is the axis-title size
## of a single panel; axis tick labels and legends come out 1.5 pt smaller.
## Panels that sit in a grid inside one figure use BASE_PT_PANEL so the text
## does not crowd.
BASE_PT       <- 10
BASE_PT_PANEL <- 9
BASE_PT_MIN   <- 6      # PNAS floor; nothing may be set below this

## geom_text()/annotate() take `size` in MILLIMETRES, not points.  Always write
## TXT(8) rather than a bare number, so annotation sizes are in the same units
## as everything else.
TXT <- function(pt) {
  stopifnot(pt >= BASE_PT_MIN)
  pt / ggplot2::.pt
}

## ---------------------------------------------------------------------------
## PNAS figure sizes, in mm (width x height)
## ---------------------------------------------------------------------------
## PNAS accepts three widths; these are the sizes the journal quotes.  Pass
## save_fig(..., size = "small") and the figure comes out at exactly one of
## them, with the preset recorded in the .meta.json.  A multi-panel row that
## needs a PNAS *width* but not the full height uses
## width = PNAS_WIDTH[["large"]] with its own height.
PNAS_SIZE  <- list(small  = c(90,  60),    #  9 x  6 cm -- one column
                   medium = c(110, 110),   # 11 x 11 cm -- 1.5 columns
                   large  = c(180, 220),   # 18 x 22 cm -- two columns, full page
                   ## The assembled main-text figures: full width, and the
                   ## height Yuki set on 2026-09-23.  "main" is the default for
                   ## every main figure; "main_tall" is Fig.6, whose third
                   ## panel needs the extra centimetre.
                   main      = c(180, 50),   # 18 x 5 cm
                   main_tall = c(180, 60),   # 18 x 6 cm
                   ## Square individual panels, for a figure hand-assembled
                   ## next to non-square schematics (e.g. Fig.5A next to the
                   ## 5B/5C artwork) -- square_panels() (aspect.ratio = 1) on
                   ## the plot itself is still needed so the PANEL, not just
                   ## the canvas, comes out square. 45mm is the smallest that
                   ## still fits a one-line axis title/subtitle at BASE_PT;
                   ## 40mm clips both.
                   square_45 = c(45, 45))
PNAS_WIDTH <- c(small = 90, medium = 110, large = 180)

## Height of an assembled main-text figure, in mm.  Change these two numbers
## and every main figure changes with them.  For a one-off height, pass
## save_fig(..., width = PNAS_WIDTH[["large"]], height = 55) instead.
FIG_H_MAIN      <- PNAS_SIZE$main[2]        # 50 mm = 5 cm
FIG_H_MAIN_TALL <- PNAS_SIZE$main_tall[2]   # 60 mm = 6 cm

theme_grhd <- function(base_size = BASE_PT) {
  cowplot::theme_half_open(base_size, font_family = FONT_FAMILY) +
    theme(
      plot.title       = element_text(size = base_size, face = "plain"),
      legend.key.size  = unit(3.2, "mm"),
      legend.text      = element_text(size = base_size - 1.5),
      legend.title     = element_text(size = base_size - 1.5),
      axis.title       = element_text(size = base_size),
      axis.text        = element_text(size = base_size - 1.5),
      strip.background = element_blank(),
      strip.text       = element_text(size = base_size - 0.5, hjust = 0.03)
    )
}

## ggplot2 3.5 renamed the "legend inside the panel" API.  One helper so the
## notebooks read the same on either version.
legend_inside <- function(x, y) {
  if (utils::packageVersion("ggplot2") >= "3.5.0") {
    theme(legend.position = "inside", legend.position.inside = c(x, y))
  } else {
    theme(legend.position = c(x, y))
  }
}

## Panel tags (A, B, C ...) placed in the margin, so they never sit on top of
## a rotated y axis title.  Apply to an assembled patchwork with `& tag_theme()`.
TAG_PT <- 12   # panel labels (A, B, ...), bold, in every assembled figure

tag_theme <- function(size = TAG_PT) {
  th <- theme(plot.tag = element_text(size = size, face = "bold"),
              plot.margin = margin(3, 3, 3, 3))   # the tag gets its own row in the margin
  if (utils::packageVersion("ggplot2") >= "3.5.0") {
    th <- th + theme(plot.tag.location = "margin", plot.tag.position = "topleft")
  }
  th
}

## ---------------------------------------------------------------------------
## Rasterised point layers
## ---------------------------------------------------------------------------
## A panel of 30,000 points is 30,000 vector objects in a PDF.  ggrastr turns
## the LAYER (and only the layer -- axes, lines and text stay vector) into an
## image at `dpi`.  Lower dpi = smaller file, coarser dots.
HAVE_GGRASTR <- requireNamespace("ggrastr", quietly = TRUE)
raster_points <- function(layer, dpi = RASTER_DPI) {
  if (HAVE_GGRASTR) ggrastr::rasterise(layer, dpi = dpi) else layer
}

## The standard cell / symbiont scatter layer of the manuscript. Rasterises
## only if this call's own `data` carries at least RASTER_MIN_N points; pass
## `raster = TRUE`/`FALSE` to override the count-based default.
points_cells <- function(data, mapping, dpi = RASTER_DPI, size = 0.25,
                         alpha = 0.25, colour = COL$point, shape = 16,
                         raster = nrow(data) >= RASTER_MIN_N) {
  layer <- geom_point(data = data, mapping = mapping, colour = colour,
                      size = size, alpha = alpha, shape = shape)
  if (raster) raster_points(layer, dpi = dpi) else layer
}

## ---------------------------------------------------------------------------
## Axes
## ---------------------------------------------------------------------------
## The manuscript uses a LINEAR r axis (0, 25, 50, 75, 100).  That is the
## default here.  scale_x_r(log = TRUE) gives the log version, which is only
## worth using when a grid is very unevenly spaced.
scale_x_r <- function(rmax = 100, log = FALSE,
                      name = expression(italic(r)~"(symbiont genome size)")) {
  if (log) {
    list(scale_x_log10(name = name, breaks = c(1, 2, 5, 10, 20, 50, 100),
                       labels = c("1", "2", "5", "10", "20", "50", "100"),
                       limits = c(0.9, rmax * 1.05)),
         annotation_logticks(sides = "b", size = 0.2, short = unit(0.4, "mm"),
                             mid = unit(0.7, "mm"), long = unit(1.1, "mm")))
  } else {
    scale_x_continuous(name = name, breaks = seq(0, 100, 25),
                       limits = c(0, rmax), expand = expansion(mult = 0.02))
  }
}

## The symbiont-free reference line, with the manuscript's label.
hline_mu0 <- function(mu_H_0, label = "Symbiont-free growth", label_x = 1,
                      size = TXT(7), vjust = -0.55, hjust = 0) {
  l <- list(geom_hline(yintercept = mu_H_0, colour = COL$ref, linewidth = 0.4))
  if (!is.null(label)) {
    l <- c(l, list(annotate("text", x = label_x, y = mu_H_0, label = label,
                            size = size, vjust = vjust, hjust = hjust,
                            colour = "black", family = FONT_FAMILY)))
  }
  l
}

## ---------------------------------------------------------------------------
## Highest-density-region bands (Fig.3C, Fig.4C)
## ---------------------------------------------------------------------------
## `cont` is what hdr_contours() returns: columns level / piece / x / y (plus
## any facet column).  The bands are drawn opaque, outermost (75 %) first, so
## the nested regions read as a filled relief over the points underneath.
## Draw these BEFORE the point layer: the points belong on top, and the bands
## are semi-transparent so the points under them still read.
geom_hdr_bands <- function(cont, linewidth = 0.22, alpha = 0.65) {
  d <- cont[order(-cont$level), , drop = FALSE]
  d$level_f <- factor(d$level, levels = c(75, 50, 25),
                      labels = c("75%", "50%", "25%"))
  ## ggplot draws polygon groups in the sorted order of the group value, so the
  ## group name has to carry the painting order: 75 % first (bottom), 25 % last
  ## (on top).  Without the rank prefix the outermost band paints over the
  ## inner ones and the panel shows a single flat blob.
  d$grp <- sprintf("%d_%s_%s", as.integer(d$level_f), d$piece, d$level)
  list(
    geom_polygon(data = d, aes(x, y, group = grp, fill = level_f),
                 colour = COL$hdr_line, linewidth = linewidth, alpha = alpha),
    scale_fill_manual(values = c(`75%` = COL$hdr_fill[["75"]],
                                 `50%` = COL$hdr_fill[["50"]],
                                 `25%` = COL$hdr_fill[["25"]]),
                      name = "Probs.", drop = FALSE)
  )
}

## Facet labels drawn INSIDE the panel instead of on a strip, so a grid of
## panels can be packed tight.  `labels` is a data frame carrying the facet
## column(s) and a `label` column.
##
## `parse = TRUE` reads the `label` column as plotmath, which is how a label
## gets an italic variable in it: write "italic(r) == 5", not "r = 5".  Use
## lab_r() below rather than writing that string by hand.
facet_label_inside <- function(labels, size = TXT(8), hjust = NULL, vjust = 1.5,
                               spacing = 1.2, corner = c("topleft", "topright"),
                               parse = FALSE) {
  corner <- match.arg(corner)
  xpos <- if (corner == "topleft") -Inf else Inf
  if (is.null(hjust)) hjust <- if (corner == "topleft") -0.25 else 1.25
  list(
    geom_text(data = labels, aes(x = xpos, y = Inf, label = label),
              inherit.aes = FALSE, hjust = hjust, vjust = vjust, size = size,
              colour = "black", family = FONT_FAMILY, parse = parse),
    theme(strip.text = element_blank(), strip.background = element_blank(),
          panel.spacing = unit(spacing, "mm"))
  )
}

## "r = 13" with an ITALIC r, as a plotmath string for facet_label_inside(...,
## parse = TRUE).  A variable is italic everywhere else in these figures, and
## the panel labels were the one place it was not.
lab_r <- function(r) sprintf("italic(r) == %s", format(r))

## ---------------------------------------------------------------------------
## Axis lines on every panel of a facet grid
## ---------------------------------------------------------------------------
## facet_wrap() draws an axis only on the outer edge of the grid, so the inner
## panels of a 2 x 2 float with no frame at all.  The manuscript gives every
## panel its own L-shaped axis line and keeps the ticks and the tick labels on
## the OUTSIDE only.
##
## ggplot2 3.5's `axes = "all"` is close but puts ticks on every panel too, and
## there is no per-panel tick theme to switch them back off -- so the lines are
## drawn as segments at the panel edge instead, which facet_wrap() then repeats
## in every panel for free.  Add it LAST, so the lines sit over the data.
##
## The segments lie exactly on the panel boundary, so clipping takes the outer
## half of each; `linewidth` is therefore twice the theme's axis.line width, and
## the visible weight comes out the same.  (Clipping has to stay on: the HDR
## contours are computed on a grid wider than the panel and rely on it.)
panel_axis_lines <- function(linewidth = 1, colour = "black") {
  list(
    annotate("segment", x = -Inf, xend = Inf, y = -Inf, yend = -Inf,
             linewidth = linewidth, colour = colour),
    annotate("segment", x = -Inf, xend = -Inf, y = -Inf, yend = Inf,
             linewidth = linewidth, colour = colour)
  )
}

## Square panels: the x and y axes come out the same length, whatever the
## canvas is.  Spare canvas becomes margin rather than stretching the panel.
square_panels <- function(ratio = 1) theme(aspect.ratio = ratio)

## Axis labels reused across panels -----------------------------------------
## Convention: a VARIABLE is italic, a descriptive super/subscript is upright.
## So mu^H is "mu" (a variable) with an upright "H" (a label, meaning "host").
##
## Greek goes through mx(), which swaps the plotmath symbol for the Unicode
## mathematical-italic code point (see above).  Set GREEK_ITALIC = FALSE in
## R/00_config.R to get upright Greek back.
LAB_MU_H      <- mx(mu^H)
LAB_MU_S      <- mx(mu^S)
LAB_DMU_H     <- mx(Delta*mu^H)
LAB_MU_H_POST <- mx(mu^H~"after evolution")
LAB_MU_S_POST <- mx(mu^S~"after evolution")

## ---------------------------------------------------------------------------
## save_fig() : write pdf + png + meta.json under FIG_DIR
## ---------------------------------------------------------------------------
save_fig <- function(p, name, width = 90, height = 60, units = "mm",
                     size = NULL, meta = list(), dir = FIG_DIR) {
  if (!is.null(size)) {
    stopifnot(size %in% names(PNAS_SIZE))
    width  <- PNAS_SIZE[[size]][1]
    height <- PNAS_SIZE[[size]][2]
  }
  ## A 4.5 x 4 "figure" is almost always centimetres typed into a millimetre
  ## argument; it is smaller than its own text and the PDF device fails with an
  ## unrelated-looking font error, so catch it here instead.
  if (units == "mm" && (width < 15 || height < 15))
    stop(sprintf("save_fig(\"%s\"): width = %g, height = %g are in MILLIMETRES -- did you mean %g x %g?",
                 name, width, height, width * 10, height * 10), call. = FALSE)
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  pdf_path <- file.path(dir, paste0(name, ".pdf"))
  png_path <- file.path(dir, paste0(name, ".png"))
  ## Explicit white background.  cowplot's theme leaves plot.background blank,
  ## and ggsave then takes that literally and writes a TRANSPARENT figure --
  ## which looks fine in a viewer that composites on white and turns into a
  ## black rectangle in one that composites on black.  Not something to
  ## discover in a proof.
  ## cairo_pdf first; the plain pdf() device is only a fallback, and it cannot
  ## draw non-ASCII glyphs (mu, x, ...), so when both fail report BOTH errors --
  ## the first one is the real cause.
  cairo_err <- NULL
  ok <- tryCatch({
    ggsave(pdf_path, p, width = width, height = height, units = units,
           device = grDevices::cairo_pdf, bg = "white")
    TRUE
  }, error = function(e) { cairo_err <<- conditionMessage(e); FALSE })
  if (!ok) tryCatch(
    ggsave(pdf_path, p, width = width, height = height, units = units, bg = "white"),
    error = function(e) stop(sprintf(
      "save_fig(\"%s\") failed.\n  cairo_pdf: %s\n  pdf() fallback: %s",
      name, cairo_err, conditionMessage(e)), call. = FALSE))
  ggsave(png_path, p, width = width, height = height, units = units,
         dpi = FIG_DPI, bg = "white")
  meta_full <- c(
    list(figure = name,
         created = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
         created_by = "analysis_R (R)",
         sim_dir = SIM_DIR,
         width_mm = width, height_mm = height,
         pnas_size = if (is.null(size)) "custom" else size,
         font_family = FONT_FAMILY, base_pt = BASE_PT,
         greek_mode = GREEK_MODE_USED, greek_italic = GREEK_ITALIC_OK,
         png_dpi = FIG_DPI, raster_layer_dpi = RASTER_DPI,
         R_version = paste(R.version$major, R.version$minor, sep = ".")),
    meta
  )
  write_json(meta_full, file.path(dir, paste0(name, ".meta.json")),
             auto_unbox = TRUE, pretty = TRUE, digits = NA)
  invisible(c(pdf = pdf_path, png = png_path))
}

## Write a statistics table and say where it went (so notebooks stay readable).
save_tab <- function(df, name, dir = TAB_DIR) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  p <- file.path(dir, paste0(name, ".csv"))
  readr::write_csv(df, p)
  message(sprintf("wrote %s  (%d rows x %d cols)", p, nrow(df), ncol(df)))
  invisible(p)
}

## Display a plot at a sensible size inside Jupyter.
fig_size <- function(w = 6, h = 4) {
  options(repr.plot.width = w, repr.plot.height = h, repr.plot.res = 150)
}
