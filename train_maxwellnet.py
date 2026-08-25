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
    skip_valid = skip_valid or get_spec_with_default(specs, "SkipValid", False)
    train_dataset, valid_dataset = ShapeDataset.load_train_valid(
        hf_config, mode, valid_fraction, seed_number if seed_number is not None else 0,
        n_samples=n_samples, sample_id=sample_id, theta=theta)

    # Fetching a sample means decoding an HF Arrow row (numpy conversion of
    # the optical_constant array), which is CPU-bound and otherwise
    # serializes with GPU/MPS compute; worker processes overlap it instead.
    num_workers = get_spec_with_default(specs, "NumWorkers", min(8, os.cpu_count() or 0))
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
              device, mode, pml_thickness, log_freq)
        logging.info("[Train] {} epoch. Loss: {:.5f}".format(
            epoch, loss_train[epoch-1].item())) if rank == 0 else None
        if perform_valid:
            valid(valid_loader, model, epoch, loss_valid,
                  device, mode, pml_thickness, log_freq)
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
    return loss, envelope, residual, incident


def train(train_loader, model, optimizer, epoch, loss_train, device, mode, pml_thickness, log_freq):
    model.train()
    n_batches = len(train_loader)
    log_envelope = log_residual = log_incident = log_sample_ids = log_theta = None
    with torch.set_grad_enabled(True):
        count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for batch_idx, data in enumerate(pbar):
            loss, envelope, residual, incident = _compute_loss(
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
                log_sample_ids = list(data['sample_id'])
                log_theta = data['theta'].detach().cpu().tolist()

        loss_train[epoch-1] = loss_train[epoch-1] / count

    wandb.log({'train/loss': loss_train[epoch-1].item(), 'epoch': epoch})
    if log_freq and epoch % log_freq == 0:
        log_fields_to_wandb(log_envelope, log_residual, log_incident, log_sample_ids, log_theta, mode, 'train', epoch)


def valid(valid_loader, model, epoch, loss_valid, device, mode, pml_thickness, log_freq):
    model.eval()
    log_envelope = log_residual = log_incident = log_sample_ids = log_theta = None
    with torch.set_grad_enabled(False):
        count = 0

        pbar = tqdm(valid_loader, desc=f"Epoch {epoch} [valid]", leave=False)
        for batch_idx, data in enumerate(pbar):
            loss, envelope, residual, incident = _compute_loss(
                data, model, device, mode, pml_thickness)

            loss_valid[epoch-1] += loss.item()
            count += 1
            pbar.set_postfix(loss=loss.item())

            if batch_idx == 0:
                log_envelope = envelope.detach().cpu()
                log_residual = residual.detach().cpu()
                log_incident = incident.detach().cpu()
                log_sample_ids = list(data['sample_id'])
                log_theta = data['theta'].detach().cpu().tolist()

        loss_valid[epoch-1] = loss_valid[epoch-1] / count

    wandb.log({'valid/loss': loss_valid[epoch-1].item(), 'epoch': epoch})
    if log_freq and epoch % log_freq == 0:
        log_fields_to_wandb(log_envelope, log_residual, log_incident, log_sample_ids, log_theta, mode, 'valid', epoch)


# Cap how many samples in the logged batch get their own figure, so a
# large (e.g. full-dataset) batch doesn't generate dozens of images per
# log point -- deliberately-small batches (like "both angles of one
# sample_id") are well under this and get one figure each.
MAX_LOGGED_SAMPLES = 8


def log_fields_to_wandb(envelope, residual, incident, sample_ids, theta, mode, train_valid, epoch):
    """Plot real, imaginary, and amplitude of the predicted envelope; real
    and amplitude of the reconstructed total field E_total = incident *
    (1 + envelope); and the Helmholtz residual (|residual|^2) -- one
    figure per sample in the batch (up to MAX_LOGGED_SAMPLES), labeled by
    sample_id and incidence angle so e.g. the same sample_id at different
    angles is distinguishable rather than only the first batch entry."""
    n = min(len(sample_ids), MAX_LOGGED_SAMPLES)
    images = {}
    for i in range(n):
        label = f'{sample_ids[i]}_theta{theta[i]:.0f}'
        if mode == 'te':
            components = [('', envelope[i], residual[i], incident[i])]
        else:
            components = [('_z', envelope[i, 0], residual[i, 0], incident[i]),
                          ('_x', envelope[i, 1], residual[i, 1], incident[i])]

        for suffix, a, r, inc in components:
            e_total = (1.0 + a) * inc

            fig, axes = plt.subplots(1, 6, figsize=(30, 4), constrained_layout=True)
            for ax, part, title in zip(axes[:3], (a.real, a.imag, a.abs()), ('real', 'imaginary', 'amplitude')):
                im = ax.imshow(part.numpy(), origin='lower', aspect='equal', cmap='magma')
                ax.set_title(f'{label}\n{title}(envelope){suffix}', fontsize=9)
                fig.colorbar(im, ax=ax)

            for ax, part, title in zip(axes[3:5], (e_total.real, e_total.abs()), ('real', 'amplitude')):
                im = ax.imshow(part.numpy(), origin='lower', aspect='equal', cmap='magma')
                ax.set_title(f'{label}\n{title}(E_total){suffix}', fontsize=9)
                fig.colorbar(im, ax=ax)

            im = axes[5].imshow(r.abs().pow(2).numpy(), origin='lower', aspect='equal', cmap='magma')
            axes[5].set_title(f'{label}\n|residual|^2{suffix}', fontsize=9)
            fig.colorbar(im, ax=axes[5])

            images[f'{train_valid}/{mode}/envelope_residual/{label}{suffix}'] = wandb.Image(fig)
            plt.close(fig)

    images['epoch'] = epoch
    wandb.log(images)


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
