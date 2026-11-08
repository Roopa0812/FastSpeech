# English checkpoints

* `Svarah_step15000_model_fp16.pth.tar`: weights only (large tensors stored in fp16), enough for inference (`python inference.py ... --restore_step 15000` after renaming/copying it to `<ckpt_path>/15000.pth.tar`).
* Full training checkpoints (weights + optimizer, about 418 MB each, every 2,000-3,000 steps) are in the shared folder: https://drive.google.com/drive/folders/1a5j23s1Ur31dnGiNgBH_yArFlT_vwF1a?usp=drive_link
