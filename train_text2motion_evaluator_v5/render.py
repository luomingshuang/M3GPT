import os
import random
import shutil
import sys
import glob
import natsort
from pathlib import Path
from argparse import ArgumentParser

import options_render as opt

def render_cli() -> None:
    # gt_joint_npy_files = glob.glob(os.path.join(opt_lm.joints_npy_save_path, "motion_ref/*.npy"))
    # pd_joint_npy_files = glob.glob(os.path.join(opt_lm.joints_npy_save_path, "rst/*.npy"))

    import numpy as np

    from render.blender import render
    from render.video import Video

    init = True

    # output_dir = os.path.join(opt_lm.joints_npy_save_path, 'motion_ref')
    # for path in gt_joint_npy_files:

    data_dir = '/home/luomingshuang/codes/multi-modal-motion-generation/unihcp-for-unified-motion-tasks-clean/train_text2motion_evaluator'
    task_dir = 'visualizations_tae/train/rst'

    joints_npy_dir = os.path.join(data_dir, task_dir)
    visualization_results_save_path = os.path.join(data_dir, task_dir)
    
    joints_npy_files = glob.glob(os.path.join(joints_npy_dir, "*.npy"))
    output_dir = visualization_results_save_path

    for path in joints_npy_files:
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
