# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

import torch
from Dataset import ShapeDataset
from ShapeNet import PeriodicMaxwellNet
from losses.helmholtz_checker import helmholtz_residual_loss_periodic_pml
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    assert batch_size == 1, (
        "ShapeDataset samples have varying grid sizes, so only BatchSize=1 is supported.")
    epochs = get_spec_with_default(specs, "Epochs", 1)
    snapshot_freq = specs["SnapshotFrequency"]

    checkpoints = list(range(snapshot_freq, epochs + 1, snapshot_freq))

    filename = 'maxwellnet_' + mode
    writer = SummaryWriter(os.path.join(directory, 'tensorboard_' + filename))
    writer_freq = get_spec_with_default(specs, "TensorboardFrequency", None)

    hf_config = get_spec_with_default(specs, "HFConfig", "validation")
    valid_fraction = get_spec_with_default(specs, "ValidFraction", 0.1)
    train_dataset, valid_dataset = ShapeDataset.load_train_valid(
        hf_config, mode, valid_fraction, seed_number if seed_number is not None else 0)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, sampler=None)
    logging.info("Train Dataset length: {}".format(len(train_dataset)))
    loss_train = torch.zeros(
        (int(epochs),), dtype=torch.float32, requires_grad=False)

    perform_valid = len(valid_dataset) > 0

    if perform_valid == True:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                   shuffle=True, pin_memory=True, sampler=None)
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
              device, mode, pml_thickness, writer, writer_freq)
        logging.info("[Train] {} epoch. Loss: {:.5f}".format(
            epoch, loss_train[epoch-1].item())) if rank == 0 else None
        if perform_valid:
            valid(valid_loader, model, epoch, loss_valid,
                  device, mode, pml_thickness, writer, writer_freq)
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
                }, directory, 'latest')

        scheduler.step()

    writer.close() if rank == 0 else None


def _compute_loss(data, model, device, mode, pml_thickness):
    optical_constant = data['optical_constant'].to(device)
    pol = data['pol'][0]
    wavelength_a = data['wavelength_nm'].item() * 10.0
    delta_x_a = data['delta_x_a'].item()
    delta_z_a = data['delta_z_a'].item()

    field_pred, epsilon_map = model(optical_constant)

    residual = helmholtz_residual_loss_periodic_pml(
        field_pred[0], epsilon_map[0], pol, wavelength_a, delta_x_a, delta_z_a, pml_thickness)

    loss = torch.mean(residual.abs().pow(2))
    return loss, field_pred[0]


def train(train_loader, model, optimizer, epoch, loss_train, device, mode, pml_thickness, writer, writer_freq):
    model.train()
    with torch.set_grad_enabled(True):
        count = 0

        for data in train_loader:
            loss, field_pred = _compute_loss(data, model, device, mode, pml_thickness)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
            optimizer.step()

            loss_train[epoch-1] += loss.item()
            count += 1

        loss_train[epoch-1] = loss_train[epoch-1] / count

    if epoch % writer_freq == 0 and writer != None:
        to_tensorboard(field_pred.clone().detach().cpu(), loss_train[epoch-1].numpy(), epoch,
                       mode, writer, 'train')


def valid(valid_loader, model, epoch, loss_valid, device, mode, pml_thickness, writer, writer_freq):
    model.eval()
    with torch.set_grad_enabled(False):
        count = 0

        for data in valid_loader:
            loss, field_pred = _compute_loss(data, model, device, mode, pml_thickness)

            loss_valid[epoch-1] += loss.item()
            count += 1

        loss_valid[epoch-1] = loss_valid[epoch-1] / count

    if epoch % writer_freq == 0 and writer != None:
        to_tensorboard(field_pred.clone().detach().cpu(), loss_valid[epoch-1].numpy(), epoch,
                       mode, writer, 'valid')


def to_tensorboard(field, losses, epoch, mode, writer, train_valid):
    if mode == 'te':
        fields = [field]
        labels = ['y']
    else:
        fields = [field[0], field[1]]
        labels = ['z', 'x']

    for label, f in zip(labels, fields):
        amplitude = f.abs()
        amplitude = amplitude - torch.min(amplitude)
        amplitude = amplitude / torch.max(amplitude)
        writer.add_image(train_valid + '/' + mode + '/amplitude_' +
                         label, amplitude.unsqueeze(0), epoch)

        real = f.real
        real = real - torch.min(real)
        real = real / torch.max(real)
        writer.add_image(train_valid + '/' + mode + '/real_' +
                         label, real.unsqueeze(0), epoch)

        imaginary = f.imag
        imaginary = imaginary - torch.min(imaginary)
        imaginary = imaginary / torch.max(imaginary)
        writer.add_image(train_valid + '/' + mode + '/imaginary_' +
                         label, imaginary.unsqueeze(0), epoch)

    writer.add_scalar(train_valid + '/' + mode, losses, epoch)


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
        help="This should specify a filename of your checkpoint within 'directory'\model if you want to continue your training from the checkpoint.",
    )

    args = arg_parser.parse_args()
    main(args.load_ckpt)
