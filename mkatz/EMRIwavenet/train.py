"""Trainining script for WaveNet vocoder

usage: train.py [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    -h, --help                   Show this help message and exit
"""
from docopt import docopt

import sys

import os
from os.path import dirname, join, expanduser
from tqdm import tqdm  # , trange
from datetime import datetime
import random

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavenet_vocoder import builder
import lrschedule

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler

from nnmnkwii import preprocessing as P
from nnmnkwii.datasets import FileSourceDataset, FileDataSource

import librosa.display

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tensorboardX import SummaryWriter
from matplotlib import cm
from warnings import warn

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
from wavenet_vocoder.mixture import discretized_mix_logistic_loss
from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic

import audio
from hparams import hparams, hparams_debug_string

from sklearn.preprocessing import MinMaxScaler

import h5py

global_step = 0
global_test_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


def sanity_check(model, c, g):
    if model.has_speaker_embedding():
        if g is None:
            raise RuntimeError(
                "WaveNet expects speaker embedding, but speaker-id is not provided")
    #else:
    #    if g is not None:
    #        raise RuntimeError(
    #            "WaveNet expects no speaker embedding, but speaker-id is provided")

    if model.local_conditioning_enabled():
        if c is None:
            raise RuntimeError("WaveNet expects conditional features, but not given")
    else:
        if c is not None:
            raise RuntimeError("WaveNet expects no conditional features, but given")


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0, constant_values=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=constant_values)
    return x


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


def clone_as_averaged_model(device, model, ema):
    assert ema is not None
    averaged_model = build_model().to(device)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        mask_ = mask.expand_as(target)
        losses = self.criterion(input, target)
        return ((losses * mask_).sum()) / mask_.sum()


class DiscretizedMixturelogisticLoss(nn.Module):
    def __init__(self):
        super(DiscretizedMixturelogisticLoss, self).__init__()

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, 1)
        mask_ = mask.expand_as(target)

        losses = discretized_mix_logistic_loss(
            input, target, num_classes=hparams.quantize_channels,
            log_scale_min=hparams.log_scale_min, reduce=False)
        assert losses.size() == target.size()
        return ((losses * mask_).sum()) / mask_.sum()


def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


def assert_ready_for_upsampling(x, c):
    assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size()


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def save_waveplot(path, y_hat, y_target):
    sr = hparams.sample_rate

    plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(y_target)
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(y_hat)
    ax2.set_xlabel('time (s)')
    xticks = np.arange(5)*1000
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks*10)

    ax2.set_ylabel('Normalized Strain')
    ax1.set_ylabel('Normalized Strain')
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()


def eval_model(global_step, writer, device, model, y, c, g, input_lengths, eval_dir, ema=None):
    if ema is not None:
        print("Using averaged model for evaluation")
        model = clone_as_averaged_model(device, model, ema)
        model.make_generation_fast_()

    model.eval()
    #pick one of the available waves to try to emulate
    idx = np.random.randint(0, len(y))
    length = input_lengths[idx].data.cpu().item()
    
    # (T,)
    y_target = y[idx].view(-1).data.cpu().numpy()[:length]

    if c is not None:
        if hparams.upsample_conditional_features:
            c = c[idx, :, :length // audio.get_hop_size()].unsqueeze(0)
        else:
            c = c[idx, :, :length].unsqueeze(0)
        assert c.dim() == 3
        print("Shape of local conditioning features: {}".format(c.size()))
    if g is not None:
        # TODO: test
        g = g[idx]
        print("Shape of global conditioning features: {}".format(g.size()))

    # Dummy silence
    if is_mulaw_quantize(hparams.input_type):
        initial_value = P.mulaw_quantize(0, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        initial_value = P.mulaw(0.0, hparams.quantize_channels)
    else:
        #initial_value = 0.0
        initial_value = float(y_target[0])
    #TODO change initial value to first value of actual waveform instead of zero?? <MLK, 10/19>
    print("Intial value:", initial_value)

    # (C,)
    if is_mulaw_quantize(hparams.input_type):
        initial_input = np_utils.to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = torch.from_numpy(initial_input).view(
            1, 1, hparams.quantize_channels)
    else:
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)
    initial_input = initial_input.to(device)

    # Run the model in fast eval mode
    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=g, T=length, softmax=True, quantize=True, tqdm=tqdm,
            log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
        y_target = P.inv_mulaw_quantize(y_target, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
        y_target = P.inv_mulaw(y_target, hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    # Save audio
    os.makedirs(eval_dir, exist_ok=True)
    path = join(eval_dir, "step_noncausal_{:09d}_predicted.npy".format(global_step))
    np.save(path, y_hat)
    path = join(eval_dir, "step_noncausal_{:09d}_target.npy".format(global_step))
    np.save(path, y_target)

    # save figure
    path = join(eval_dir, "step_noncausal_{:09d}_waveplots.png".format(global_step))
    save_waveplot(path, y_hat, y_target)


def save_states(global_step, writer, y_hat, y, input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))
    idx = np.random.randint(0, len(y_hat))
    length = input_lengths[idx].data.cpu().item()

    # (B, C, T)
    if y_hat.dim() == 4:
        y_hat = y_hat.squeeze(-1)

    if is_mulaw_quantize(hparams.input_type):
        # (B, T)
        y_hat = F.softmax(y_hat, dim=1).max(1)[1]

        # (T,)
        y_hat = y_hat[idx].data.cpu().long().numpy()
        y = y[idx].view(-1).data.cpu().long().numpy()

        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
        y = P.inv_mulaw_quantize(y, hparams.quantize_channels)
    else:
        # (B, T)
        y_hat = sample_from_discretized_mix_logistic(
            y_hat, log_scale_min=hparams.log_scale_min)
        # (T,)
        y_hat = y_hat[idx].view(-1).data.cpu().numpy()
        y = y[idx].view(-1).data.cpu().numpy()

        if is_mulaw(hparams.input_type):
            y_hat = P.inv_mulaw(y_hat, hparams.quantize_channels)
            y = P.inv_mulaw(y, hparams.quantize_channels)

    # Mask by length
    y_hat[length:] = 0
    y[length:] = 0

    # Save audio
    audio_dir = join(checkpoint_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    path = join(audio_dir, "step{:09d}_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(audio_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y, sr=hparams.sample_rate)


def __train_step(device, phase, epoch, global_step, global_test_step,
                 model, optimizer, writer, criterion,
                 x, y, c, g, input_lengths,
                 checkpoint_dir, eval_dir=None, do_eval=False, ema=None):
    sanity_check(model, c, g)

    # x : (B, C, T)
    # y : (B, T, 1)
    # c : (B, C, T)
    # g : (B,)
    train = (phase == "train")
    clip_thresh = hparams.clip_thresh
    if train:
        model.train()
        step = global_step
    else:
        model.eval()
        step = global_test_step

    # Learning rate schedule
    current_lr = hparams.initial_learning_rate
    if train and hparams.lr_schedule is not None:
        lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
        current_lr = lr_schedule_f(
            hparams.initial_learning_rate, step, **hparams.lr_schedule_kwargs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    optimizer.zero_grad()

    # Prepare data
    x, y = x.to(device), y.to(device)
    input_lengths = input_lengths.to(device)
    
    c = c.to(device) if c is not None else None
    g = g.to(device) if g is not None else None
    
    # (B, T, 1)
    mask = sequence_mask(input_lengths, max_len=x.size(-1)).unsqueeze(-1)
    mask = mask[:, 1:, :]

    # Apply model: Run the model in regular eval mode
    # NOTE: softmax is handled in F.cross_entrypy_loss
    # y_hat: (B x C x T)

    if use_cuda:
        # multi gpu support
        # you must make sure that batch size % num gpu == 0
        y_hat = torch.nn.parallel.data_parallel(model, (x, c, g, False))
    else:
        y_hat = model(x, c, g, False)

    if is_mulaw_quantize(hparams.input_type):
        # wee need 4d inputs for spatial cross entropy loss
        # (B, C, T, 1)
        y_hat = y_hat.unsqueeze(-1)
        loss = criterion(y_hat[:, :, :-1, :], y[:, 1:, :], mask=mask)
    else:
        loss = criterion(y_hat[:, :, :-1], y[:, 1:, :], mask=mask)

    if train and step > 0 and step % hparams.checkpoint_interval == 0:
        save_states(step, writer, y_hat, y, input_lengths, checkpoint_dir)
        save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch, ema)

    if do_eval:
        # NOTE: use train step (i.e., global_step) for filename
        eval_model(global_step, writer, device, model, y, c, g, input_lengths, eval_dir, ema)

    # Update
    if train:
        loss.backward()
        if clip_thresh > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh)
        optimizer.step()
        # update moving average
        if ema is not None:
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)

    # Logs
    writer.add_scalar("{} loss".format(phase), float(loss.item()), step)
    if train:
        if clip_thresh > 0:
            writer.add_scalar("gradient norm", grad_norm, step)
        writer.add_scalar("learning rate", current_lr, step)

    return loss.item()


def train_loop(device, model, data_loaders, optimizer, writer, checkpoint_dir=None):
    if is_mulaw_quantize(hparams.input_type):
        criterion = MaskedCrossEntropyLoss()
    else:
        criterion = DiscretizedMixturelogisticLoss()

    if hparams.exponential_moving_average:
        ema = ExponentialMovingAverage(hparams.ema_decay)
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    else:
        ema = None

    global global_step, global_epoch, global_test_step
    while global_epoch < hparams.nepochs:
        for phase in ['train', 'test']:
            data_loader = data_loaders[phase]
            train = (phase == "train")
            running_loss = 0.
            test_evaluated = False
            
            for step, (x, y, c, g, input_lengths) in enumerate(data_loader):
                 # Whether to save eval (i.e., online decoding) result
                if bool(c[0] == 0.0) and bool(c[1] == 0.0):
                    c = None
                do_eval = False
                eval_dir = join(checkpoint_dir, "{}_eval".format(phase))
                # Do eval per eval_interval for train
                if train and global_step > 0 \
                        and global_step % hparams.train_eval_interval == 0:
                    do_eval = True
                # Do eval for test
                # NOTE: Decoding WaveNet is quite time consuming, so
                # do only once in a single epoch for testset
                if not train and not test_evaluated \
                        and global_epoch % hparams.test_eval_epoch_interval == 0:
                    do_eval = True
                    test_evaluated = True
                if do_eval:
                    print("[{}] Eval at train step {}".format(phase, global_step))

                # Do step
                running_loss += __train_step(device,
                                             phase, global_epoch, global_step, global_test_step, model,
                                             optimizer, writer, criterion, x, y, c, g, input_lengths,
                                             checkpoint_dir, eval_dir, do_eval, ema)

                # update global state
                if train:
                    global_step += 1
                else:
                    global_test_step += 1
                print(step)
            # log per epoch
            averaged_loss = running_loss / len(data_loader)
            writer.add_scalar("{} loss (per epoch)".format(phase),
                              averaged_loss, global_epoch)
            print("Step {} [{}] Loss: {}".format(
                global_step, phase, running_loss / len(data_loader)))

        global_epoch += 1


def save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch, ema=None):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    global global_test_step
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

    if ema is not None:
        averaged_model = clone_as_averaged_model(device, model, ema)
        checkpoint_path = join(
            checkpoint_dir, "checkpoint_non_causal_step{:09d}_ema.pth".format(global_step))
        torch.save({
            "state_dict": averaged_model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
            "global_test_step": global_test_step,
        }, checkpoint_path)
        print("Saved averaged checkpoint:", checkpoint_path)


def build_model():
    if is_mulaw_quantize(hparams.input_type):
        if hparams.out_channels != hparams.quantize_channels:
            raise RuntimeError(
                "out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")
    if hparams.upsample_conditional_features and hparams.cin_channels < 0:
        s = "Upsample conv layers were specified while local conditioning disabled. "
        s += "Notice that upsample conv layers will never be used."
        warn(s)

    model = getattr(builder, hparams.builder)(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        weight_normalization=hparams.weight_normalization,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_scales=hparams.upsample_scales,
        freq_axis_kernel_size=hparams.freq_axis_kernel_size,
        scalar_input=is_scalar_input(hparams.input_type),
        legacy=hparams.legacy,
    )
    return model


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_test_step = checkpoint.get("global_test_step", 0)

    return model


# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
def restore_parts(path, model):
    print("Restore part of the model from: {}".format(path))
    state = _load(path)["state_dict"]
    model_dict = model.state_dict()
    valid_state_dict = {k: v for k, v in state.items() if k in model_dict}

    try:
        model_dict.update(valid_state_dict)
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        # there should be invalid size of weight(s), so load them per parameter
        print(str(e))
        model_dict = model.state_dict()
        for k, v in valid_state_dict.items():
            model_dict[k] = v
            try:
                model.load_state_dict(model_dict)
            except RuntimeError as e:
                print(str(e))
                warn("{}: may contain invalid size of weight. skipping...".format(k))

class Dataset(data_utils.Dataset):
    def __init__(self, data_root, filename, features_name, labels_name, labels_scaler):
        self.filename = filename
        self.features_name, self.labels_name = features_name, labels_name
        self.data_root = data_root
        self.labels_scaler = labels_scaler

        with h5py.File(data_root + filename, 'r') as f:
            self.data_length = len(f[features_name])

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample
        with h5py.File(self.data_root + self.filename, 'r') as f:
            #print('start data')
            X = torch.from_numpy(np.asarray(f[self.features_name][index]/np.abs(f[self.features_name][index]).max())[np.newaxis,:]*0.999)
            y = torch.from_numpy(self.labels_scaler.transform(f[self.labels_name][index][np.newaxis,:]).squeeze())

            #print('end data')
        # Load data and get label
        return X, X.transpose(0,1), torch.zeros(()), y, X.shape[-1] 

def get_data_loaders(data_root, speaker_id, batch_size=hparams.batch_size, test_shuffle=True, lims = {'ecc':[0.01, 0.7], 'inc':[0.01, 1.3962634016]}):
    data_loaders = {}
    scalerY = MinMaxScaler()
    scaler = scalerY.fit(np.array([np.linspace(lims['ecc'][0], lims['ecc'][1]), np.linspace(lims['inc'][0], lims['inc'][1])]).T)
    local_conditioning = hparams.cin_channels > 0
    for phase in ["train", "test"]:
        dataset = Dataset(data_root, 'dilated_%s.hdf5'%phase, 'features', 'labels', scalerY)

        data_loaders[phase] = data_utils.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loaders

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    checkpoint_restore_parts = args["--restore-parts"]
    speaker_id = args["--speaker-id"]
    speaker_id = int(speaker_id) if speaker_id is not None else None
    preset = args["--preset"]

    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(dirname(__file__), "data", "ljspeech")

    log_event_path = args["--log-event-path"]
    reset_optimizer = args["--reset-optimizer"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"
    print(hparams_debug_string())

    fs = hparams.sample_rate

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataloader setup
    data_loaders = get_data_loaders(data_root, speaker_id, test_shuffle=True)
    
    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = build_model().to(device)

    receptive_field = model.receptive_field
    print("Receptive field (samples / ms): {} / {}".format(
        receptive_field, receptive_field / fs * 1000))

    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
        hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
        amsgrad=hparams.amsgrad)

    if checkpoint_restore_parts is not None:
        restore_parts(checkpoint_restore_parts, model)

    # Load checkpoints
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer)

    # Setup summary writer for tensorboard
    if log_event_path is None:
        log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("TensorBoard event log path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    # Train!
    try:
        train_loop(device, model, data_loaders, optimizer, writer,
                   checkpoint_dir=checkpoint_dir)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        save_checkpoint(
            device, model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")

    sys.exit(0)
