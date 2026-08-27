# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

import torch
from Dataset import ShapeDataset
from ShapeNet import PeriodicMaxwellNet
from maxwell_losses.helmholtz_checker import helmholtz_residual_loss_periodic_pml
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
import wandb
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import random
import logging
import argparse
import os
import json
from datetime import datetime


def main(load_ckpt, reset_lr=False, epochs_override=None, n_samples=None, skip_valid=False, sample_id=None, theta=None):
    directory = "test_run1"
    os.makedirs(directory, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(
                            os.getcwd(), directory, f"maxwellnet_{datetime.now():%Y-%m-%d %H-%M-%S}.log"),
                        filemode='w')

    logging.info("training " + directory)

    specs_filename = os.path.join('specs_maxwell.json')

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs_maxwell.json"'
        )

    specs = json.load(open(specs_filename))

    seed_number = get_spec_with_default(specs, "Seed", None)
    if seed_number != None:
        fix_seed(seed_number, torch.cuda.is_available())

    rank = 0
    device = torch.device('cuda' if torch.cuda.is_available()
                           else 'mps' if torch.backends.mps.is_available()
                           else 'cpu')

    logging.info("Experiment description: \n" +
                 ' '.join([str(elem) for elem in specs["Description"]]))
    logging.info("Training with " + str(device))

    physical_specs = specs["PhysicalSpecs"]
    mode = physical_specs['mode']
    pml_thickness = physical_specs['pml_thickness']

    model = PeriodicMaxwellNet(**specs["NetworkSpecs"], mode=mode)
    if torch.cuda.device_count() > 1:
        logging.info("Multiple GPUs: " + str(torch.cuda.device_count()))
    if load_ckpt is not None:
        load_path = os.path.join(os.getcwd(), directory, 'model', load_ckpt)
        ckpt_dict = torch.load(load_path + '.pt')
        ckpt_epoch = ckpt_dict['epoch']
        logging.info("Checkpoint loaded from {}-epoch".format(ckpt_epoch))
        model.load_state_dict(ckpt_dict['state_dict'])
        wandb_run_id = ckpt_dict.get('wandb_run_id')
    else:
        wandb_run_id = None

    model = torch.nn.DataParallel(model)
    model.train()
    model = model.to(device)

    logging.info("Number of network parameters: {}".format(
        sum(p.data.nelement() for p in model.parameters())))
    logging.debug(specs["NetworkSpecs"])
    logging.debug(specs["PhysicalSpecs"])

    optimizer = torch.optim.Adam(model.parameters(), lr=get_spec_with_default(
        specs, "LearningRate", 0.0001), weight_decay=0)
    scheduler = StepLR(optimizer, step_size=get_spec_with_default(
        specs, "LearningRateDecayStep", 10000), gamma=get_spec_with_default(specs, "LearningRateDecay", 1.0))

    batch_size = get_spec_with_default(specs, "BatchSize", 1)
    epochs = epochs_override if epochs_override is not None else get_spec_with_default(specs, "Epochs", 1)
    snapshot_freq = specs["SnapshotFrequency"]
    save_e_field = get_spec_with_default(specs, "SaveEField", True)

    checkpoints = list(range(snapshot_freq, epochs + 1, snapshot_freq))

    filename = 'maxwellnet_' + mode
    wandb_specs = get_spec_with_default(specs, "WandB", {})
    log_freq = wandb_specs.get("LogFrequency", 1)
    wandb.init(
        project=wandb_specs.get("Project", "maxwellnet"),
        entity=wandb_specs.get("Entity"),
        mode=wandb_specs.get("Mode", "online"),
        dir=directory,
        name=filename,
        config=specs,
        id=wandb_run_id,
        resume="allow" if wandb_run_id is not None else None,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    wandb.define_metric("train_step")
    wandb.define_metric("train/loss_step", step_metric="train_step")

    hf_config = get_spec_with_default(specs, "HFConfig", None)
    valid_fraction = get_spec_with_default(specs, "ValidFraction", 0.1)
    n_samples = n_samples if n_samples is not None else get_spec_with_default(specs, "NumTrainSamples", None)
    sample_id = sample_id if sample_id is not None else get_spec_with_default(specs, "SampleId", None)
    theta = theta if theta is not None else get_spec_with_default(specs, "SampleTheta", None)
    force_valid_ids = get_spec_with_default(specs, "ForceValidIds", None)
    skip_valid = skip_valid or get_spec_with_default(specs, "SkipValid", False)
    train_dataset, valid_dataset = ShapeDataset.load_train_valid(
        hf_config, mode, valid_fraction, seed_number if seed_number is not None else 0,
        n_samples=n_samples, sample_id=sample_id, theta=theta, force_valid_ids=force_valid_ids)

    # Fetching a sample means decoding an HF Arrow row (numpy conversion of
    # the optical_constant array), which is CPU-bound and otherwise
    # serializes with GPU/MPS compute; worker processes overlap it instead.
    # os.cpu_count() reports the whole machine's core count, not what's
    # actually allocated to this process -- on a shared cluster node under
    # SLURM/cgroups (e.g. a job given 2 of a node's many cores), that
    # overshoots badly and triggers DataLoader's own "excessive worker"
    # warning. os.sched_getaffinity(0) (Linux-only, hence the guard)
    # reports the CPU set this process can actually run on.
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cpus = os.cpu_count() or 0
    num_workers = get_spec_with_default(specs, "NumWorkers", min(8, available_cpus))
    # Cap below the dataset size so an overfitting run (n_samples small)
    # doesn't spin up worker processes it has nothing to hand them.
    train_num_workers = min(num_workers, len(train_dataset))
    train_loader_kwargs = dict(num_workers=train_num_workers, pin_memory=(device.type == "cuda"))
    if train_num_workers > 0:
        train_loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"))
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    # Don't shuffle when deliberately targeting a small, fixed set of
    # samples (a single row via n_samples, or every row for one sample_id)
    # -- shuffling a handful of items adds nothing but non-determinism.
    overfit_single = (n_samples == 1) or (sample_id is not None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=(not overfit_single), sampler=None, **train_loader_kwargs)
    logging.info("Train Dataset length: {}".format(len(train_dataset)))
    loss_train = torch.zeros(
        (int(epochs),), dtype=torch.float32, requires_grad=False)

    perform_valid = len(valid_dataset) > 0 and not skip_valid

    if perform_valid == True:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                   shuffle=False, sampler=None, **loader_kwargs)
        logging.info("Valid Dataset length: {}".format(len(valid_dataset)))
        loss_valid = torch.zeros(
            (int(epochs),), dtype=torch.float32, requires_grad=False)

    if load_ckpt is not None:
        if reset_lr:
            new_lr = get_spec_with_default(specs, "LearningRate", 0.0001)
            for group in optimizer.param_groups:
                group['lr'] = new_lr
            logging.info("Resetting LR to {} and restarting decay schedule from {}-epoch".format(
                new_lr, ckpt_epoch))
        else:
            optimizer.load_state_dict(ckpt_dict['optimizer'])
            scheduler.load_state_dict(ckpt_dict['scheduler'])
        loss_train[:ckpt_epoch:] = ckpt_dict['loss_train'][:ckpt_epoch:]
        logging.info("Check point loaded from {}-epoch".format(ckpt_epoch))

        start_epoch = ckpt_epoch
    else:
        start_epoch = 0

    logging.info("Training start")

    for epoch in range(start_epoch + 1, epochs + 1):
        train(train_loader, model, optimizer, epoch, loss_train,
              device, mode, pml_thickness, log_freq, directory, save_e_field)
        logging.info("[Train] {} epoch. Loss: {:.5f}".format(
            epoch, loss_train[epoch-1].item())) if rank == 0 else None
        if perform_valid and log_freq and epoch % log_freq == 0:
            valid(valid_loader, model, epoch, loss_valid,
                  device, mode, pml_thickness, directory, save_e_field)
            logging.info("[Valid] {} epoch. Loss: {:.5f}".format(
                epoch, loss_valid[epoch-1].item())) if rank == 0 else None

        if epoch in checkpoints:
            logging.info("Checkpoint saved at {} epoch.".format(
                epoch)) if rank == 0 else None
            if rank == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_train': loss_train,
                    'scheduler': scheduler.state_dict(),
                    'wandb_run_id': wandb.run.id,
                }, directory, str(epoch) + '_' + mode)

        if epoch % 200 == 0:
            logging.info("'latest' checkpoint saved at {} epoch.".format(
                epoch)) if rank == 0 else None
            if rank == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_train': loss_train,
                    'scheduler': scheduler.state_dict(),
                    'wandb_run_id': wandb.run.id,
                }, directory, 'latest')

        wandb.log({'train/lr': optimizer.param_groups[0]['lr'], 'epoch': epoch}) if rank == 0 else None

        scheduler.step()

    wandb.finish() if rank == 0 else None


def _compute_loss(data, model, device, mode, pml_thickness):
    optical_constant = data['optical_constant'].to(torch.complex64).to(device)
    pol = data['pol'][0]
    # wavelength/grid spacing are the same across a batch (ShapeDataset
    # samples are drawn at fixed dimensions/resolution), so the first
    # sample's values apply to the whole batch.
    wavelength_a_real = data['wavelength_nm'][0].item() * 10.0
    delta_x_a_real = data['delta_x_a'][0].item()
    delta_z_a_real = data['delta_z_a'][0].item()

    envelope, epsilon_map, incident, kz, kx = model(
        optical_constant,
        data['theta'].to(torch.float32).to(device),
        data['wavelength_nm'].to(torch.float32).to(device),
        data['delta_x_a'].to(torch.float32).to(device),
        data['delta_z_a'].to(torch.float32).to(device),
    )

    # Re-express in units of the sample's own wavelength (wavelength=1,
    # matching fl/replicate's/master's convention) instead of absolute
    # Angstroms, so the residual/loss magnitude no longer depends on this
    # dataset's arbitrary choice of Angstrom as the length unit -- k=2*pi
    # is fixed regardless of the sample's real k0, and grid spacing becomes
    # a fraction of one wavelength. This is a pure unit-system change, not
    # a physics change: `incident` (built from the real kz/kx/delta) is
    # left as-is, and kz/kx are rescaled by wavelength_a_real so that
    # kz*dz and kx*Lx (the only ways they enter the residual, via the
    # z-truncation phase-advance and the Bloch phase) come out identical
    # to their real-unit values -- verified: kz_norm*dz_norm ==
    # kz_real*wavelength_a_real * (dz_real/wavelength_a_real) ==
    # kz_real*dz_real exactly, and likewise for kx*Lx.
    wavelength_a = 1.0
    delta_x_a = delta_x_a_real / wavelength_a_real
    delta_z_a = delta_z_a_real / wavelength_a_real
    kz = kz * wavelength_a_real
    kx = kx * wavelength_a_real

    residual = helmholtz_residual_loss_periodic_pml(
        envelope, epsilon_map, incident, kz, kx, pol, wavelength_a, delta_x_a, delta_z_a, pml_thickness)

    loss = torch.mean(residual.abs().pow(2))
    return loss, envelope, residual, incident, epsilon_map, kz, kx, wavelength_a, delta_x_a, delta_z_a


# Matches ShapeNet.PeriodicMaxwellNet._EPS_INDEX exactly, so the FEM
# ground-truth field is selected into the same channel convention/shape
# that `envelope`/`epsilon_map` already use for the physics residual.
_E_FIELD_INDEX = {'te': [2], 'tm': [0, 1]}


def _fem_residual(e_field, epsilon_map, incident, kz, kx, mode, wavelength_a, delta_x_a, delta_z_a, pml_thickness):
    """Helmholtz residual of the FEM ground-truth field itself, plugged
    into the same discrete residual the model is trained against -- an
    empirical floor on how small |residual|^2 can get on this fixed
    finite-difference grid, since even the FEM solver's field doesn't
    exactly satisfy *this* grid's discrete equation (different mesh /
    discretization from FEM's own)."""
    pol = 's' if mode == 'te' else 'p'
    e_sel = e_field[..., _E_FIELD_INDEX[mode]].movedim(-1, 1).to(torch.complex64)  # (B, k, Nz, Nx)
    if mode == 'te':
        e_sel = e_sel[:, 0]
        envelope_gt = e_sel / incident - 1.0
    else:
        envelope_gt = e_sel / incident.unsqueeze(1) - 1.0
    return helmholtz_residual_loss_periodic_pml(
        envelope_gt, epsilon_map, incident, kz, kx, pol, wavelength_a, delta_x_a, delta_z_a, pml_thickness)


def train(train_loader, model, optimizer, epoch, loss_train, device, mode, pml_thickness, log_freq, directory, save_e_field):
    model.train()
    n_batches = len(train_loader)
    log_envelope = log_residual = log_incident = log_e_field = log_sample_ids = log_theta = None
    log_epsilon_map = log_kz = log_kx = log_wavelength_a = log_delta_x_a = log_delta_z_a = None
    with torch.set_grad_enabled(True):
        count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for batch_idx, data in enumerate(pbar):
            loss, envelope, residual, incident, epsilon_map, kz, kx, wavelength_a, delta_x_a, delta_z_a = _compute_loss(
                data, model, device, mode, pml_thickness)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            optimizer.step()

            loss_train[epoch-1] += loss.item()
            count += 1
            pbar.set_postfix(loss=loss.item(), grad_norm=grad_norm.item())

            train_step = (epoch - 1) * n_batches + batch_idx
            wandb.log({'train/loss_step': loss.item(), 'train/grad_norm': grad_norm.item(), 'train_step': train_step})

            if batch_idx == 0:
                log_envelope = envelope.detach().cpu()
                log_residual = residual.detach().cpu()
                log_incident = incident.detach().cpu()
                log_e_field = data['e_field'].detach().cpu()
                log_sample_ids = list(data['sample_id'])
                log_theta = data['theta'].detach().cpu().tolist()
                log_epsilon_map = epsilon_map.detach().cpu()
                log_kz = kz.detach().cpu()
                log_kx = kx.detach().cpu()
                log_wavelength_a, log_delta_x_a, log_delta_z_a = wavelength_a, delta_x_a, delta_z_a

        loss_train[epoch-1] = loss_train[epoch-1] / count

    wandb.log({'train/loss': loss_train[epoch-1].item(), 'epoch': epoch})
    if log_freq and epoch % log_freq == 0:
        log_residual_gt = _fem_residual(log_e_field, log_epsilon_map, log_incident, log_kz, log_kx, mode,
                                        log_wavelength_a, log_delta_x_a, log_delta_z_a, pml_thickness)
        log_fields_to_wandb(log_envelope, log_residual, log_residual_gt, log_incident, log_e_field,
                            log_sample_ids, log_theta, mode, 'train', epoch)
        if save_e_field:
            save_e_field_npz(log_envelope, log_incident, log_sample_ids, log_theta, mode, 'train', epoch, directory)


def valid(valid_loader, model, epoch, loss_valid, device, mode, pml_thickness, directory, save_e_field):
    model.eval()
    log_envelope = log_residual = log_incident = log_e_field = log_sample_ids = log_theta = None
    log_epsilon_map = log_kz = log_kx = log_wavelength_a = log_delta_x_a = log_delta_z_a = None
    with torch.set_grad_enabled(False):
        count = 0

        pbar = tqdm(valid_loader, desc=f"Epoch {epoch} [valid]", leave=False)
        for batch_idx, data in enumerate(pbar):
            loss, envelope, residual, incident, epsilon_map, kz, kx, wavelength_a, delta_x_a, delta_z_a = _compute_loss(
                data, model, device, mode, pml_thickness)

            loss_valid[epoch-1] += loss.item()
            count += 1
            pbar.set_postfix(loss=loss.item())

            if batch_idx == 0:
                log_envelope = envelope.detach().cpu()
                log_residual = residual.detach().cpu()
                log_incident = incident.detach().cpu()
                log_e_field = data['e_field'].detach().cpu()
                log_sample_ids = list(data['sample_id'])
                log_theta = data['theta'].detach().cpu().tolist()
                log_epsilon_map = epsilon_map.detach().cpu()
                log_kz = kz.detach().cpu()
                log_kx = kx.detach().cpu()
                log_wavelength_a, log_delta_x_a, log_delta_z_a = wavelength_a, delta_x_a, delta_z_a

        loss_valid[epoch-1] = loss_valid[epoch-1] / count

    wandb.log({'valid/loss': loss_valid[epoch-1].item(), 'epoch': epoch})
    log_residual_gt = _fem_residual(log_e_field, log_epsilon_map, log_incident, log_kz, log_kx, mode,
                                    log_wavelength_a, log_delta_x_a, log_delta_z_a, pml_thickness)
    log_fields_to_wandb(log_envelope, log_residual, log_residual_gt, log_incident, log_e_field,
                        log_sample_ids, log_theta, mode, 'valid', epoch)
    if save_e_field:
        save_e_field_npz(log_envelope, log_incident, log_sample_ids, log_theta, mode, 'valid', epoch, directory)


# Cap how many samples in the logged batch get their own figure, so a
# large (e.g. full-dataset) batch doesn't generate dozens of images per
# log point -- deliberately-small batches (like "both angles of one
# sample_id") are well under this and get one figure each.
MAX_LOGGED_SAMPLES = 8


# e_field's last dim is 3 components ordered [Ez, Ex, Ey] (matching
# ShapeNet._EPS_INDEX); 'te' (s-pol) only has a nonzero Ey (index 2), 'tm'
# ground-truth plotting here only shows Ez (index 0), matching the 'x'
# component already being omitted from the envelope/residual panels below.
_E_FIELD_CHANNEL = {'te': 2, 'tm': 0}


def _fft_log_intensity(field_abs):
    """log10(|FFT2(field)|^2 + eps), DC-centered -- field_abs is a real 2D
    tensor (an amplitude map), so the spectrum's symmetry doesn't matter
    for visualization; this is purely a diagnostic for how much
    high-spatial-frequency content a field amplitude map carries."""
    spectrum = torch.fft.fftshift(torch.fft.fft2(field_abs))
    return torch.log10(spectrum.abs().pow(2) + 1e-12)


def _log_intensity(x):
    """log10(x + eps), for plotting an always-nonnegative quantity (e.g.
    |residual|^2) whose dynamic range spans many orders of magnitude."""
    return torch.log10(x + 1e-12)


def _sample_panels(a, r, inc, e_gt, r_gt):
    """The 9 (part, title) panels shown per sample: real/amplitude of the
    envelope, real/amplitude of the reconstructed total field E_total =
    incident * (1 + envelope) and its 2D-FFT log intensity, log|residual|^2,
    the FEM ground-truth field's amplitude and its 2D-FFT log intensity,
    and the FEM field's own log|residual|^2 (an empirical floor -- how far
    even the "true" field is from exactly satisfying this discrete
    residual). imaginary(envelope) is deliberately omitted to keep each
    sample's row from growing further."""
    e_total = (1.0 + a) * inc
    e_total_amp = e_total.abs()
    e_gt_amp = e_gt.abs()
    return [
        (a.real, 'real(envelope)'),
        (a.abs(), 'amplitude(envelope)'),
        (e_total.real, 'real(E_total)'),
        (e_total_amp, 'amplitude(E_total)'),
        (_fft_log_intensity(e_total_amp), 'log|FFT(amp(E_total))|^2'),
        (_log_intensity(r.abs().pow(2)), 'log|residual|^2'),
        (e_gt_amp, 'amplitude(E_field) [FEM]'),
        (_fft_log_intensity(e_gt_amp), 'log|FFT(amp(E_field))|^2 [FEM]'),
        (_log_intensity(r_gt.abs().pow(2)), 'log|residual|^2 [FEM]'),
    ]


_PANELS_PER_SAMPLE = 9


def _plot_angle_group(indices, envelope, residual, residual_gt, incident, e_field, sample_ids, mode):
    """One figure per incidence angle: one row per sample, each row the 9
    panels from _sample_panels."""
    n_samples = len(indices)
    ncols = _PANELS_PER_SAMPLE
    fig, axes = plt.subplots(n_samples, ncols, figsize=(ncols * 3.2, n_samples * 3.2), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, i in enumerate(indices):
        e_gt = e_field[i, ..., _E_FIELD_CHANNEL[mode]]
        if mode == 'te':
            a, r, r_gt, inc = envelope[i], residual[i], residual_gt[i], incident[i]
        else:
            # tm: this grid shows the 'z' component only; not the primary
            # use case for this layout, so the 'x' component isn't tiled.
            a, r, r_gt, inc = envelope[i, 0], residual[i, 0], residual_gt[i, 0], incident[i]

        for k, (part, title) in enumerate(_sample_panels(a, r, inc, e_gt, r_gt)):
            ax = axes[row, k]
            im = ax.imshow(part.numpy(), origin='lower', aspect='equal', cmap='magma')
            ax.set_title(f'{sample_ids[i]}\n{title}', fontsize=8)
            fig.colorbar(im, ax=ax)

    return fig


def log_fields_to_wandb(envelope, residual, residual_gt, incident, e_field, sample_ids, theta, mode, train_valid, epoch):
    """Group the logged batch (up to MAX_LOGGED_SAMPLES) by incidence
    angle, and produce one combined figure per angle -- rather than one
    figure per sample -- tiling every sample sharing that angle as one row
    each (see _plot_angle_group)."""
    n = min(len(sample_ids), MAX_LOGGED_SAMPLES)
    angle_groups = {}
    for i in range(n):
        angle_groups.setdefault(round(theta[i], 3), []).append(i)

    images = {}
    for angle, indices in angle_groups.items():
        fig = _plot_angle_group(indices, envelope, residual, residual_gt, incident, e_field, sample_ids, mode)
        images[f'{train_valid}/{mode}/envelope_residual_theta{angle:.0f}'] = wandb.Image(fig)
        plt.close(fig)

    images['epoch'] = epoch
    wandb.log(images)


def save_e_field_npz(envelope, incident, sample_ids, theta, mode, train_valid, epoch, directory):
    """Save the predicted total complex field E_total = incident * (1 +
    envelope) for every currently-logged sample (same MAX_LOGGED_SAMPLES cap
    and cadence as log_fields_to_wandb) to a single .npz per log point,
    under `directory`/e_field/."""
    if mode == 'te':
        e_total = (1.0 + envelope) * incident
    else:
        e_total = (1.0 + envelope) * incident.unsqueeze(1)

    n = min(len(sample_ids), MAX_LOGGED_SAMPLES)
    out_dir = os.path.join(directory, 'e_field')
    os.makedirs(out_dir, exist_ok=True)

    fields = {f"{sample_ids[i]}_theta{theta[i]:.0f}": e_total[i].numpy() for i in range(n)}
    path = os.path.join(out_dir, f'{train_valid}_{mode}_epoch{epoch}.npz')
    np.savez(path, sample_ids=np.array(sample_ids[:n]), theta=np.array(theta[:n]), **fields)


def save_checkpoint(state, directory, filename):
    model_directory = os.path.join(directory, 'model')
    if os.path.exists(model_directory) == False:
        os.makedirs(model_directory)
    torch.save(state, os.path.join(model_directory, filename + '.pt'))


def fix_seed(seed, is_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train a MaxwellNet")
    arg_parser.add_argument(
        "--load_ckpt",
        "-l",
        default=None,
        help="This should specify a filename of your checkpoint within 'directory'/model if you want to continue your training from the checkpoint.",
    )
    arg_parser.add_argument(
        "--reset_lr",
        action="store_true",
        help="When resuming from --load_ckpt, ignore the checkpoint's saved optimizer/scheduler state and restart the optimizer at the LearningRate in specs_maxwell.json, with the decay schedule restarting from the checkpoint's epoch.",
    )
    arg_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override Epochs from specs_maxwell.json.",
    )
    arg_parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Limit training to this many unique samples (e.g. 1, for an overfitting sanity check). Overrides NumTrainSamples in specs_maxwell.json.",
    )
    arg_parser.add_argument(
        "--skip_valid",
        action="store_true",
        help="Skip validation entirely, regardless of ValidFraction, for faster overfitting runs. Also honored via SkipValid in specs_maxwell.json.",
    )
    arg_parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="Restrict training to every row matching this sample_id (e.g. 'sample_0000', all its incidence angles), bypassing the shuffle/split entirely with an empty valid split. Overrides SampleId in specs_maxwell.json.",
    )
    arg_parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Used together with --sample_id: further restrict to just the row(s) at this incidence angle (degrees), e.g. 0.0. Overrides SampleTheta in specs_maxwell.json.",
    )

    args = arg_parser.parse_args()
    main(args.load_ckpt, args.reset_lr, args.epochs, args.n_samples, args.skip_valid, args.sample_id, args.theta)
