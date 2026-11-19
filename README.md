# Passenger Counter

From the repository root, install the script and notebook dependencies into
your virtual environment:

```shell
python -m pip install -r requirements.txt
```

For the NVIDIA CUDA 13.0 setup used in the notebook, install the matching
PyTorch wheels first, then run the requirements install above:

```shell
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
```

Run `python predict_video.py` from the repository root. The script uses the
trained checkpoint at `runs/detect/train4/weights/best.pt`.

The checkpoint is the only file tracked under `runs/`. The notebook's
resume-training and evaluation cells require local artifacts, such as `last.pt`,
`results.csv`, and validation plots, at the paths used in those cells. To use
those cells on a fresh clone, restore the matching artifacts or generate them
with a new training run and adjust the paths as needed.
