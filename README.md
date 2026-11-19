# Passenger Counter

The video prediction script uses the trained checkpoint at
`runs/detect/train4/weights/best.pt`. Run `python predict_video.py` from the
repository root after installing Ultralytics, OpenCV, and yt-dlp.

The checkpoint is the only file tracked under `runs/`. The notebook's
resume-training and evaluation cells require local artifacts, such as `last.pt`,
`results.csv`, and validation plots, at the paths used in those cells. To use
those cells on a fresh clone, restore the matching artifacts or generate them
with a new training run and adjust the paths as needed.
