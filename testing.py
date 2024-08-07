import train_wavegan as tw
import argparse
import glob
import sys
import os

args = tw.argProcess(
    'train_wavegan.py train ./train \
	--data_dir ../sc09/test3 \
	--data_first_slice \
	--data_pad_end \
	--data_fast_wav'
)

#args.data_slice_len //= 2
args.wavegan_latent_dim //= 2
args.wavegan_kernel_len //= 2
args.wavegan_dim //= 2
args.train_batch_size //= 4

# Make train dir
if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

# Save args
with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

# Make model kwarg dicts
setattr(args, 'wavegan_g_kwargs', {
    'slice_len': args.data_slice_len,
    'nch': args.data_num_channels,
    'kernel_len': args.wavegan_kernel_len,
    'dim': args.wavegan_dim,
    'use_batchnorm': args.wavegan_batchnorm,
    'upsample': args.wavegan_genr_upsample
})
setattr(args, 'wavegan_d_kwargs', {
    'kernel_len': args.wavegan_kernel_len,
    'dim': args.wavegan_dim,
    'use_batchnorm': args.wavegan_batchnorm,
    'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
})

if args.mode == 'train':
    fps = glob.glob(os.path.join(args.data_dir, '*'))
    if len(fps) == 0:
        raise Exception('Did not find any audio files in specified directory')
    print('Found {} audio files in specified directory'.format(len(fps)))
    tw.infer(args)
    tw.train(fps, args)
elif args.mode == 'preview':
    tw.preview(args)
elif args.mode == 'incept':
    tw.incept(args)
elif args.mode == 'infer':
    tw.infer(args)
else:
    raise NotImplementedError()