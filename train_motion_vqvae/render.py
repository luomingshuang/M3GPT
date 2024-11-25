import os
import random
import shutil
import sys
import glob
import natsort
from pathlib import Path
from argparse import ArgumentParser

from options import render as opt

def render_cli() -> None:
    output_val_dir_pd = '/home/luomingshuang/codes/multi-modal-motion-generation/unihcp-for-unified-motion-tasks/train_motion-vqvae/visualization_samples/val_gt'
    output_val_dir_gt = '/home/luomingshuang/codes/multi-modal-motion-generation/unihcp-for-unified-motion-tasks/train_motion-vqvae/visualization_samples/val_pd'

    gt_joint_npy_files = glob.glob(os.path.join(output_val_dir_gt, "*.npy"))
    pd_joint_npy_files = glob.glob(os.path.join(output_val_dir_pd, "*.npy"))

    import numpy as np

    from render.blender import render
    from render.video import Video

    init = True

    output_dir = output_val_dir_gt
    for path in gt_joint_npy_files:

    # output_dir = output_val_dir_pd
    # for path in pd_joint_npy_files:
        # check existed mp4 or under rendering
        print(path)
        if opt.mode == "video":
            if os.path.exists(path.replace(".npy", ".mp4")) or os.path.exists(
                    path.replace(".npy", "_frames")):
                print(f"npy is rendered or under rendering {path}")
                continue
        else:
            # check existed png
            if os.path.exists(path.replace(".npy", ".png")):
                print(f"npy is rendered or under rendering {path}")
                continue

        if opt.mode == "video":
            frames_folder = os.path.join(
                output_dir,
                path.replace(".npy", "_frames").split('/')[-1])
            os.makedirs(frames_folder, exist_ok=True)
        else:
            frames_folder = os.path.join(
                output_dir,
                path.replace(".npy", ".png").split('/')[-1])

        try:
            data = np.load(path)
            if data.shape[0] == 1:
                data = data[0]
        except FileNotFoundError:
            print(f"{path} not found")
            continue

        if opt.mode == "video":
            frames_folder = os.path.join(
                output_dir,
                path.replace(".npy", "_frames").split("/")[-1])
        else:
            frames_folder = os.path.join(
                output_dir,
                path.replace(".npy", ".png").split("/")[-1])

        out = render(
            data,
            frames_folder,
            canonicalize=opt.canonicalize,
            exact_frame=opt.exact_frame,
            num=opt.num,
            mode=opt.mode,
            model_path=opt.model_path,
            faces_path=opt.faces_path,
            downsample=opt.downsample,
            always_on_floor=opt.always_on_floor,
            oldrender=opt.oldrender,
            res=opt.res,
            init=init,
            gt=opt.gt,
            accelerator=opt.accelerator,
            device=opt.device,
        )

        init = False

        if opt.mode == "video":
            shutil.copytree(frames_folder, frames_folder+'_img') 
            if opt.downsample:
                video = Video(frames_folder, fps=opt.fps)
            else:
                video = Video(frames_folder, fps=opt.fps)

            vid_path = frames_folder.replace("_frames", ".mp4")
            video.save(out_path=vid_path)
            shutil.rmtree(frames_folder)
            print(f"remove tmp fig folder and save video in {vid_path}")

        else:
            print(f"Frame generated at: {out}")

if __name__ == "__main__":
    render_cli()
