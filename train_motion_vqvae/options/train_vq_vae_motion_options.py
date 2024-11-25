batch_size=int(1000)
num_workers=int(16)

## vqvae
quantizer='ema_reset'
code_num=int(512)
code_dim=int(512)
output_emb_width=int(512)
down_t=int(2)
stride_t=int(2)
width=int(512)
depth=int(3)
dilation_growth_rate=int(3)
norm=None
activation='relu'
nfeats=int(263)

## optimizer
lr=2e-4
betas=[0.9, 0.99]
weight_decay=0.0

## lr scheduler
T_max=10*100
eta_min=1e-6

max_epoch=999999
distributed=True

## loss
recons_loss='l1_smooth'
recons_feature=1.0
recons_velocity=0.5
vq_commit=0.02
gpt_loss=1.0

## log
log_interval=10
save_interval=100000

## save dir
vq_vae_motion_pretrained_dir='/data/luomingshuang/checkpoints/unified_io_motion_tasks/vq_vae_pretrained_motion'
vq_vae_motion_pretrained_dir_v2='/data/luomingshuang/checkpoints/unified_io_motion_tasks/vq_vae_pretrained_motion_v2'

## save vq vae codes dir
vqvae_motion_save_path = '/data/luomingshuang/checkpoints/unified_io_motion_tasks/motion_vqvae'