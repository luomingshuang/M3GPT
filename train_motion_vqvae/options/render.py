input_mode='npy'
gt_joints_npy_dir='/home/luomingshuang/codes/multi-modal-motion-generation/unified-io-for-any-motion-tasks-lms/visualization_samples/vqvae_visulization/val_gt_v2'
pd_joints_npy_dir='/home/luomingshuang/codes/multi-modal-motion-generation/unified-io-for-any-motion-tasks-lms/visualization_samples/vqvae_visulization/val_pd_v2'

mode='video'

canonicalize=True
exact_frame=0.5
num=8

smpl_model_path='/home/luomingshuang/pretrained_weights/body_models/smpl'
model_path='/home/luomingshuang/pretrained_weights/body_models/smpl/'
faces_path='/data/luomingshuang/data/motiongpt_data/deps/smplh/smplh.faces'

downsample=False
always_on_floor=False
oldrender=True
res='med'

gt=False
accelerator='gpu'
device=[0]

fps=30