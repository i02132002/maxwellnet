# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

import torch
from Dataset import ShapeDataset
from ShapeNet import PeriodicMaxwellNet
from losses.helmholtz_checker import helmholtz_residual_loss_periodic_pml
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


def main(load_ckpt):
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
    epochs = get_spec_with_default(specs, "Epochs", 1)
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
    train_dataset, valid_dataset = ShapeDataset.load_train_valid(
        hf_config, mode, valid_fraction, seed_number if seed_number is not None else 0)

    # Fetching a sample means decoding an HF Arrow row (numpy conversion of
    # the optical_constant array), which is CPU-bound and otherwise
    # serializes with GPU/MPS compute; worker processes overlap it instead.
    num_workers = get_spec_with_default(specs, "NumWorkers", min(8, os.cpu_count() or 0))
    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"))
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, sampler=None, **loader_kwargs)
    logging.info("Train Dataset length: {}".format(len(train_dataset)))
    loss_train = torch.zeros(
        (int(epochs),), dtype=torch.float32, requires_grad=False)

    perform_valid = len(valid_dataset) > 0

    if perform_valid == True:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                   shuffle=False, sampler=None, **loader_kwargs)
        logging.info("Valid Dataset length: {}".format(len(valid_dataset)))
        loss_valid = torch.zeros(
            (int(epochs),), dtype=torch.float32, requires_grad=False)

    if load_ckpt is not None:
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
    wavelength_a = data['wavelength_nm'][0].item() * 10.0
    delta_x_a = data['delta_x_a'][0].item()
    delta_z_a = data['delta_z_a'][0].item()

    field_pred, epsilon_map, incident, kz = model(
        optical_constant,
        data['theta'].to(torch.float32).to(device),
        data['wavelength_nm'].to(torch.float32).to(device),
        data['delta_x_a'].to(torch.float32).to(device),
        data['delta_z_a'].to(torch.float32).to(device),
    )

    residual = helmholtz_residual_loss_periodic_pml(
        field_pred, epsilon_map, incident, kz, pol, wavelength_a, delta_x_a, delta_z_a, pml_thickness)

    loss = torch.mean(residual.abs().pow(2))
    return loss, field_pred, residual, incident


def train(train_loader, model, optimizer, epoch, loss_train, device, mode, pml_thickness, log_freq):
    model.train()
    n_batches = len(train_loader)
    log_field_pred = log_residual = log_incident = log_sample_ids = None
    with torch.set_grad_enabled(True):
        count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for batch_idx, data in enumerate(pbar):
            loss, field_pred, residual, incident = _compute_loss(data, model, device, mode, pml_thickness)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
            optimizer.step()

            loss_train[epoch-1] += loss.item()
            count += 1
            pbar.set_postfix(loss=loss.item())

            train_step = (epoch - 1) * n_batches + batch_idx
            wandb.log({'train/loss_step': loss.item(), 'train_step': train_step})

            if batch_idx == 0:
                log_field_pred = field_pred.detach().cpu()
                log_residual = residual.detach().cpu()
                log_incident = incident.detach().cpu()
                log_sample_ids = list(data['sample_id'])

        loss_train[epoch-1] = loss_train[epoch-1] / count

    wandb.log({'train/loss': loss_train[epoch-1].item(), 'epoch': epoch})
    if log_freq and epoch % log_freq == 0:
        log_fields_to_wandb(log_field_pred, log_residual, log_incident, log_sample_ids, mode, 'train', epoch)


def valid(valid_loader, model, epoch, loss_valid, device, mode, pml_thickness, log_freq):
    model.eval()
    log_field_pred = log_residual = log_incident = log_sample_ids = None
    with torch.set_grad_enabled(False):
        count = 0

        pbar = tqdm(valid_loader, desc=f"Epoch {epoch} [valid]", leave=False)
        for batch_idx, data in enumerate(pbar):
            loss, field_pred, residual, incident = _compute_loss(data, model, device, mode, pml_thickness)

            loss_valid[epoch-1] += loss.item()
            count += 1
            pbar.set_postfix(loss=loss.item())

            if batch_idx == 0:
                log_field_pred = field_pred.detach().cpu()
                log_residual = residual.detach().cpu()
                log_incident = incident.detach().cpu()
                log_sample_ids = list(data['sample_id'])

        loss_valid[epoch-1] = loss_valid[epoch-1] / count

    wandb.log({'valid/loss': loss_valid[epoch-1].item(), 'epoch': epoch})
    if log_freq and epoch % log_freq == 0:
        log_fields_to_wandb(log_field_pred, log_residual, log_incident, log_sample_ids, mode, 'valid', epoch)


# Fixed color scale for E-field plots, so brightness is comparable across
# samples/epochs instead of each panel autoscaling to its own min/max. The
# incident plane wave has unit amplitude (see ShapeNet._incident_wave), so
# the total field's real/imaginary parts and amplitude are expected to stay
# within a small multiple of that.
FIELD_VMIN, FIELD_VMAX = -2.0, 2.0
AMP_VMIN, AMP_VMAX = 0.0, 2.0


def log_fields_to_wandb(field, residual, incident, sample_ids, mode, train_valid, epoch):
    images = {}

    if mode == 'te':
        images.update(field_grid_to_wandb(field, residual, incident, sample_ids, mode, train_valid))
    else:
        single_field = field[0]
        for label, f in zip(('z', 'x'), (single_field[0], single_field[1])):
            for part_name, part, vmin, vmax in (
                ('amplitude', f.abs(), AMP_VMIN, AMP_VMAX),
                ('real', f.real, FIELD_VMIN, FIELD_VMAX),
                ('imaginary', f.imag, FIELD_VMIN, FIELD_VMAX),
            ):
                normalized = (part - vmin) / (vmax - vmin)
                normalized = torch.clamp(normalized, 0.0, 1.0)
                normalized = torch.flip(normalized, dims=[0])  # wandb.Image draws row 0 at the top; flip to match the origin='lower' residual plots
                images[f'{train_valid}/{mode}/{part_name}_{label}'] = wandb.Image(normalized.unsqueeze(0))

    images.update(plot_helmholtz_residual(residual[0], mode, train_valid))

    images['epoch'] = epoch
    wandb.log(images)


def field_grid_to_wandb(field, residual, incident, sample_ids, mode, train_valid, nrows=3, ncols=3):
    """Plot an nrows x ncols grid of samples' Ey field, each cell showing
    real(E_y), imaginary(E_y), amplitude(E_y), the Helmholtz residual
    (|residual|^2), and amplitude(E_y - incident plane wave) — i.e. the
    scattered field — side by side, titled with sample_id."""
    n = min(nrows * ncols, field.shape[0], residual.shape[0], incident.shape[0], len(sample_ids))
    fig, axes = plt.subplots(nrows, ncols * 5, figsize=(ncols * 11.0, nrows * 2.4),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax_real, ax_imag, ax_amp, ax_resid, ax_scat = (
            axes[row, col * 5], axes[row, col * 5 + 1], axes[row, col * 5 + 2],
            axes[row, col * 5 + 3], axes[row, col * 5 + 4])
        ax_real.axis('off')
        ax_imag.axis('off')
        ax_amp.axis('off')
        ax_resid.axis('off')
        ax_scat.axis('off')
        if i >= n:
            continue

        Ey = field[i]
        ax_real.imshow(Ey.real.numpy(), origin='lower', aspect='equal', vmin=FIELD_VMIN, vmax=FIELD_VMAX)
        ax_real.set_title(f'{sample_ids[i]}\nreal(E_y)', fontsize=8)
        ax_imag.imshow(Ey.imag.numpy(), origin='lower', aspect='equal', vmin=FIELD_VMIN, vmax=FIELD_VMAX)
        ax_imag.set_title(f'{sample_ids[i]}\nimaginary(E_y)', fontsize=8)
        ax_amp.imshow(Ey.abs().numpy(), origin='lower', aspect='equal', vmin=AMP_VMIN, vmax=AMP_VMAX)
        ax_amp.set_title(f'{sample_ids[i]}\namplitude(E_y)', fontsize=8)
        ax_resid.imshow(residual[i].abs().pow(2).numpy(), origin='lower', aspect='equal')
        ax_resid.set_title(f'{sample_ids[i]}\n|residual|^2', fontsize=8)
        ax_scat.imshow((Ey - incident[i]).abs().numpy(), origin='lower', aspect='equal', vmin=AMP_VMIN, vmax=AMP_VMAX)
        ax_scat.set_title(f'{sample_ids[i]}\namplitude(E_y - incident)', fontsize=8)

    image = wandb.Image(fig)
    plt.close(fig)
    return {f'{train_valid}/{mode}/field_grid': image}


def plot_helmholtz_residual(residual, mode, train_valid):
    """Render |Helmholtz residual|^2 as a labeled heatmap (with colorbar) per field component."""
    if mode == 'te':
        residuals = [('', residual)]
    else:
        residuals = [('z', residual[0]), ('x', residual[1])]

    images = {}
    for label, r in residuals:
        loss_map = r.abs().pow(2).numpy()
        suffix = f'_{label}' if label else ''

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(loss_map, origin='lower', aspect='equal')
        fig.colorbar(im, ax=ax, label='|residual|^2')
        ax.set_title(f'Helmholtz residual{suffix}')
        images[f'{train_valid}/{mode}/helmholtz_loss{suffix}'] = wandb.Image(fig)
        plt.close(fig)

    return images


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

    args = arg_parser.parse_args()
    main(args.load_ckpt)
