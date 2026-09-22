# A paired junction-stain and TEER dataset: what it takes, and why nobody has deposited one

Status 2026-09-22. PiMorph's barrier proxy is unvalidated because no public dataset pairs a junction
stain with a barrier measurement on the same monolayer. The one paired set that exists (NIST/NEI
iPSC-RPE) pairs TER with bright-field images, and on it the structural term of the proxy failed
(`runs/function_real/healthy2/REPORT.md`). The junction-coverage term, which the model was written for,
has never been tested. This note says exactly what a test needs, so that the gap is a decision rather
than an open question.

## Why this cannot be done with public data

A barrier measurement is an electrode or tracer reading on a live insert; a junction stain is a fixed,
antibody-labelled image of the same insert taken minutes to hours later. Both exist in hundreds of
papers, but they are reported as condition means (a TEER bar chart and a representative image), never as
per-insert pairs, and image deposits (BioImage Archive, IDR, Zenodo) hold images without the electrode
readings. Three systematic searches found none (`docs/DATASET_HUNT_2026-09-18.md`,
`docs/DATASET_HUNT_2026-09-19.md`, the barrier section of `runs/function_real/REPORT.md`). Figure-level
condition means (S-BIAD1169, Bromberger et al. 2024) gave no agreement, and 13 conditions is far below
the power needed. The data has to be produced with the pairing in mind.

## What the experiment is

Any laboratory with cell culture, a TEER meter (EVOM-type chopstick electrodes or an ECIS array) and a
fluorescence microscope can produce it in about three weeks. It is not something one person without a
bench can do, and it is not something a computational group needs a collaboration for beyond bench time.

1. Cells and inserts. HUVEC or HDMEC (pooled donor, passage 3 to 5) on 24-well Transwell inserts
   (0.4 um pore, 6.5 mm), fibronectin coated, seeded to confluence (about 1e5 cells per insert).
   Culture 5 to 7 days until TEER plateaus. 48 inserts (two 24-well plates).
2. Perturbations that move the barrier in both directions, so TEER spans a wide range on the same cell
   type: vehicle (n = 12); thrombin 1 U/mL for 30 min and histamine 100 uM for 15 min (rapid opening,
   n = 8 each); TNF-alpha 10 ng/mL for 18 h and VEGF 50 ng/mL for 4 h (slower remodelling, n = 6 each);
   forskolin 10 uM plus rolipram 10 uM for 30 min (tightening, n = 8). Expected TEER range on HUVEC
   about 10 to 40 Ohm cm2, roughly a threefold span, which is enough.
3. Readout. TEER read three times per insert immediately before fixation (blank-insert subtracted,
   area normalized), with the plate on a warming block so that reading and fixation are within 5
   minutes. Optional, on half the inserts: 70 kDa FITC-dextran flux for 30 min before fixation as a
   second readout, since TEER and tracer permeability measure different things (ions versus
   macromolecules) and the proxy claims the paracellular route.
4. Fixation and staining on the insert membrane: 4% PFA 10 min, VE-cadherin (adherens junction),
   ZO-1 or claudin-5 (tight junction), phalloidin, DAPI. Cut the membrane out and mount flat.
5. Imaging. Whole-insert tile scan at 20x (0.3 to 0.65 um per pixel), four channels, plus 40x fields
   for a subset. The insert has a 33 mm2 area with about 20,000 cells; four 2048 px fields per insert
   at 20x already give around 2,000 cells and 3,000 tricellular vertices, which is the field size the
   hCEC benchmark uses.
6. Blinding. The image analyst does not see the TEER table until the PiMorph statistics are frozen; the
   pre-registered outcome is the Spearman correlation between per-insert TEER and the permeability
   index computed with junction coverage, and the partial correlation given cell density and area CV.

## Power

From the correlations already seen: with n inserts and a two-sided 5% test at 80% power, detecting a
true Spearman of 0.5 needs 30 inserts, 0.4 needs 47, 0.3 needs 85. The AMD test had n = 10 at an observed
0.52, i.e. 33% power, which is why it could not decide anything. 48 inserts detect 0.4 and, with the
tightening arm, span enough TEER to separate "the proxy tracks junction state" from "the proxy tracks
cell density", which is the confound the Healthy-2 series exposed.

## Cost and time

Reagents (inserts, antibodies, agonists, fibronectin, PFA, mounting) about 1,500 to 2,500 USD; an EVOM2
meter is standard equipment in vascular biology labs and around 3,000 USD if it is not; imaging time
about two days on a core microscope. Three weeks from thaw to frozen images. The analysis is already
written: `pimorph reconstruct` with a junction-stain manifest, `pimorph.fields.multichannel` for the
VE-cadherin and ZO-1 coverage vectors, `pimorph.function.transport` for the index, and
`scripts/pimorph_function_healthy2.py stats` as the template for the pre-registered statistics.

## If bench time is not available

Two lower-value substitutes remain. First, ask the Bharti laboratory (NEI) for the ZO-1 images of the
ten AMD wells that have TER but no deposited images, which would take the RPE test from n = 10 to
n = 20 (power at r = 0.5 rises from 33% to 62%; still not decisive). Second, papers that publish
per-insert source data for both TEER and junction images are rare but not nonexistent; a targeted
request to the Polarity-JaM, Vestweber or Dejana groups for per-insert pairs from published experiments
would be the fastest route to a first test. Neither replaces the experiment above.
