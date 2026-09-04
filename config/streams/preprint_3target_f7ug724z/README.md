# Stream set for the PREPRINT 3-TARGET precipitation experiment on backbone `f7ug724z`.
#
# Forcing/input streams are copied VERBATIM from config/streams/imerg_diag_f7ug724z/ -- the
# directory the proven IMERG-only finetunes for this backbone run through -- so the encoder side
# of this experiment is bit-identical to those runs and the only change is on the output side.
#
# OUTPUT STREAMS (all three diagnostic, all three N320, all three single-channel `tp`):
#   IMERG_ANEMOI  id 40   satellite retrieval, 6 h accumulation   (imerg_anemoi.yml)
#   OPERAN_TP     id 41   ECMWF operational analysis, 6 h accum   (operan_tp.yml)
#   ERA5_TP       id 42   ERA5 reanalysis, 1 h accum -- SEE ITS HEADER, it is NOT the same
#                         quantity as the other two (era5_tp.yml)
#
# The physical loss is averaged over the streams that have targets (loss / ctr_streams), so each
# of the three carries weight 1/3. That is the intended equal-weight multi-task setup, but it
# does mean IMERG's gradient here is a THIRD of what it is in the single-target runs -- a real
# confound when comparing this experiment's IMERG scores against those runs.
#
# The inherited ERA5 output stream (id 1) stays neutralised exactly as in the source directory.
