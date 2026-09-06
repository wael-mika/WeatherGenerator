# Stream set for the PREPRINT operanonly experiment on backbone `dy0jlrmw`.
#
# Forcing/input streams copied VERBATIM from config/streams/imerg_diag_dy0jlrmw/, which is
# dy0jlrmw's own set. NOTE this backbone comes from a DIFFERENT family than nhv6tkln/aets3ku5:
# its saved streams_directory is `streams_preprint/era5_georing_avhrr_synop_lowres_linear/`,
# not `gjl6a0kb_reproduction_*`, so this directory could not be copied from the other arms.
# It has 5 forcing files (analysis, avhrr, geos, synop) and no era5.yml.
#
# Inherited stream ids are 0, 10-14, 20, 22, 30, so 41 (OPERAN_TP) and 42 (ERA5_TP) are free.
#
# The single output stream is OPERAN_TP. imerg_anemoi.yml is deliberately absent:
# omitting a file really removes the stream, because config.py sets `base_config.streams = None`
# whenever an overwrite supplies a streams_directory.
#
# dy0jlrmw's own freeze regex contains `.*ERA5.*`, which re.fullmatch would also match against
# `pred_heads.ERA5_TP`. The configs narrow it to `.*\.ERA5|.*\.ERA5\..*` -- in the era5only arm
# ERA5_TP is the ONLY trainable decoder, so the original pattern would train nothing at all.
