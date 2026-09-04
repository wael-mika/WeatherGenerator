# Stream set for the PREPRINT ERA5-ONLY precipitation experiment on backbone `cw6a4szu`.
#
# Forcing/input streams are copied VERBATIM from the operan-only arm (and thus from that
# backbone's proven IMERG-only stream dir), so the encoder side is identical across all four arms
# and only the output side differs.
#
# THE SINGLE OUTPUT STREAM IS ERA5_TP (id 42). Neither imerg_anemoi.yml nor operan_tp.yml is
# present; omitting a file really does remove the stream, because config.py sets
# `base_config.streams = None` whenever an overwrite supplies a streams_directory, so nothing is
# resurrected from the checkpoint by the merge.
#
# ERA5_TP IS A 1-HOUR ACCUMULATION while IMERG and OPERAN tp are 6-hour. See era5_tp.yml for the
# full argument. This arm is comparable across backbones, not against the other arms' absolute
# scores.
#
# The inherited ERA5 diagnostic OUTPUT stream (id 1, `reconstruct: false`) is a different stream
# from ERA5_TP and stays neutralised. On f7ug724z the freeze regex is narrowed to
# `.*\.ERA5|.*\.ERA5\..*` so that it freezes that inherited stream WITHOUT also freezing
# ERA5_TP -- which here is the only trainable decoder, so the original `.*ERA5.*` would have
# trained nothing at all.
