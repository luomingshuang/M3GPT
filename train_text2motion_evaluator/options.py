batch_size=int(50)
bs_stage2=int(100)

num_workers=int(16)

## transformer autoencoder
## transformer encoder
encoder_latent_dim=int(768)
encoder_ff_size=int(1024)
encoder_num_layers=int(6)
encoder_num_heads=int(8)
encoder_dropout=float(0.1)
encoder_activation='gelu'

## transformer decoder
decoder_latent_dim=int(768)
decoder_ff_size=int(1024)
decoder_num_layers=int(6)
decoder_num_heads=int(8)
decoder_dropout=float(0.1)
decoder_activation='gelu'

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
lr=1e-4
lr_small=1e-6
betas=[0.9, 0.99]
weight_decay=0.0001

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

load_pretrained_model=False
## save dir
motion_text_alignment_pretrained_dir='/data/luomingshuang/checkpoints/unified_io_motion_tasks/pretrained_text2motion_evaluator'

weight_file_name='pretrained_motion_tae_2_eos_768_8_new_joints_vecs.pt'

weight_file_name_stage2 = 'pretrained_motion_text_alignment.pt'