data_root = '/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints'

mean = '/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/Mean.npy'
std = '/data/luomingshuang/data/dataset_3dmotion/motionx_finedance_aistpp_30fps_22joints/Std.npy'

max_motion_length = int(196)
min_motion_length = int(40)
unit_length = int(4)

window_size = int(64)
fps = int(30)

vqvae_motion_save_path = '/data/luomingshuang/checkpoints/unified_io_motion_tasks/motion_vqvae_motiongpt_code_over_window_size'

motionx_motion_code_path = '/data/luomingshuang/checkpoints/unified_io_motion_tasks/motion_vqvae_motiongpt_code_over_window_size/motionx_vq_vae_codes'
finedance_motion_code_path = '/data/luomingshuang/checkpoints/unified_io_motion_tasks/motion_vqvae_motiongpt_code_over_window_size/finedance_vq_vae_codes'
aistpp_motion_code_path = '/data/luomingshuang/checkpoints/unified_io_motion_tasks/motion_vqvae_motiongpt_code_over_window_size/aistpp_vq_vae_codes'

finedance_music_code_path = '/data/luomingshuang/checkpoints/unified_io_motion_tasks/music_vqvae_jukebox_code/finedance_vqvae_code_jukebox'
aistpp_music_code_path = '/data/luomingshuang/checkpoints/unified_io_motion_tasks/music_vqvae_jukebox_code/aistpp_vqvae_code_jukebox'

motionx_text_path = '/data/luomingshuang/data/dataset_3dmotion/motion-x/texts/semantic_labels'

finedance_seq_len = int(120)
num_tokens_each_seconds = 7.5
windows = int(10)