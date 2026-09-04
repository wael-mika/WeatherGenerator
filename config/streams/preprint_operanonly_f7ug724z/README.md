# Stream set for the PREPRINT OPERAN-ONLY precipitation experiment on backbone `f7ug724z`.
#
# Forcing/input streams are copied VERBATIM from config/streams/imerg_diag_f7ug724z/, so the encoder
# side is bit-identical to the proven IMERG-only finetunes for this backbone.
#
# THE SINGLE OUTPUT STREAM IS OPERAN_TP (id 41). imerg_anemoi.yml is deliberately ABSENT: this
# arm asks what the frozen backbone can do against the operational analysis alone, with no IMERG
# term in the loss at all. Omitting the file really does remove the stream -- config.py sets
# `base_config.streams = None` whenever an overwrite supplies a streams_directory, so nothing is
# resurrected from the checkpoint by the merge.
#
# Because OPERAN_TP is then the only stream with targets, ctr_streams is 1 and it carries the
# full physical loss -- the same gradient scale the single-target IMERG runs had, which makes
# this arm the clean control for the 3-target arm above.
#
# The inherited ERA5 output stream (id 1) stays neutralised exactly as in the source directory.
