import pickle
import hydra
import numpy as np
import torch
import datasets

from pathlib import Path
from itertools import chain
from omegaconf import DictConfig
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
from gon_pytorch import modules, utils


def train(model, batches, input, opt, device, epoch, latent_reg, log_every=100):
    model.train()

    seen_samples = 0
    inner_loss_sum = 0
    outer_loss_sum = 0
    latent_l2_loss_sum = 0

    latent_buffer = torch.zeros(len(batches.dataset), model.decoder.latent_dim)
    label_buffer = torch.zeros(len(batches.dataset), dtype=torch.long)

    for step, (images, labels) in enumerate(batches):
        images = images.to(device)

        batch_input = input.repeat(len(images), 1, 1, 1)

        # Obtain latent with respect to origin
        latents, inner_loss = model.infer_latents(batch_input, images)

        # Optimize model with obtained latent
        out = model(batch_input, latents)
        outer_loss = model.loss_outer(out, images)

        latent_reg_loss = latent_reg * (latents ** 2).sum()
        loss = outer_loss + latent_reg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        latent_buffer[seen_samples:seen_samples + len(images)] = latents.detach().cpu()
        label_buffer[seen_samples:seen_samples + len(images)] = labels

        seen_samples += len(images)
        inner_loss_sum += inner_loss.item()
        outer_loss_sum += outer_loss.item()
        latent_l2_loss_sum += latent_reg_loss

        step += 1

        if step % log_every == 0:
            stats = {
                'avg inner loss': inner_loss_sum,
                'avg outer loss': outer_loss_sum
            }
            if latent_reg:
                stats['latent l2 loss'] = latent_l2_loss_sum

            print(f'[EPOCH {epoch:03d}][{seen_samples:05d}/{len(batches.dataset):05d}] ' +
                  ', '.join(f'{k}: {v / step:.4f}' for k, v in stats.items()))

    return latent_buffer, label_buffer


def eval(model, batches, input, device):
    model.eval()

    seen_samples = 0
    inner_loss_sum = 0
    outer_loss_sum = 0

    latent_buffer = torch.zeros(len(batches.dataset), model.decoder.latent_dim)
    label_buffer = torch.zeros(len(batches.dataset), dtype=torch.long)

    for step, (images, labels) in enumerate(batches):
        images = images.to(device)
        batch_input = input.repeat(len(images), 1, 1, 1)

        # Obtain latent with respect to origin
        latents, inner_loss = model.infer_latents(batch_input, images)

        # Calculate loss for obtained latent
        out = model(batch_input, latents)
        outer_loss = model.loss_outer(out, images)

        inner_loss_sum += inner_loss.item()
        outer_loss_sum += outer_loss.item()

        latent_buffer[seen_samples:seen_samples + len(images)] = latents.detach().cpu()
        label_buffer[seen_samples:seen_samples + len(images)] = labels

        seen_samples += len(images)

    print(f'inner loss: {inner_loss_sum / len(batches):.4f}, outer loss: {outer_loss_sum / len(batches):.4f}')

    return latent_buffer, label_buffer


def sample(model, input, mean, cov, n_samples):
    model.eval()

    latents = torch.tensor(
        np.random.multivariate_normal(mean, cov, size=n_samples), dtype=torch.float32
    ).to(input.device)

    model_input = input.repeat(n_samples, 1, 1, 1)
    samples = model(model_input, latents)
    return samples


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    device = cfg.training.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = Path.cwd()
    log_dir.mkdir(parents=True, exist_ok=True)

    recon_dir = log_dir / 'reconstructions'
    recon_dir.mkdir(exist_ok=True)

    sample_dir = log_dir / 'samples'
    sample_dir.mkdir(exist_ok=True)

    print(f'Logging to {str(log_dir)}')

    dataset_cls = getattr(datasets, cfg.dataset.name, None)
    if dataset_cls is None:
        raise ValueError(f'Unknown dataset {cfg.dataset.name}')

    dataset = dataset_cls(cfg.dataset.root, transforms.Compose([
        transforms.Resize(cfg.dataset.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.permute(1, 2, 0))
    ]))

    train_batches = DataLoader(
        dataset.train, cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers
    )
    test_batches = DataLoader(
        dataset.test, cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers
    )

    fixed_batch = next(iter(train_batches))[0][:cfg.logging.n_recons_per_epoch].to(device)

    pos_encoder_kwargs = {'in_dim': 2, **cfg.model.pos_encoder.get('args', {})}
    pos_encoder_cls = {
        'none': modules.IdentityPositionalEncoding,
        'gaussian': modules.GaussianFourierFeatureTransform,
        'nerf': modules.NeRFPositionalEncoding
    }.get(cfg.model.pos_encoder.name)
    if pos_encoder_cls is None:
        raise ValueError(f'Unknown positional encoder \'{cfg.model.pos_encoder.name}\'')

    pos_encoder = pos_encoder_cls(**pos_encoder_kwargs)
    grid = utils.get_xy_grid(cfg.dataset.image_size, cfg.dataset.image_size)
    model_input = grid.to(device)

    decoder = modules.ImplicitDecoder(
        latent_dim=cfg.model.latent_dim,
        out_dim=dataset.num_channels,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        block_factory=utils.get_block_factory(cfg.model.activation, cfg.model.bias),
        pos_encoder=pos_encoder,
        modulation=cfg.model.latent_modulation,
        dropout=cfg.model.dropout,
        final_activation=torch.sigmoid
    )
    model = modules.GON(decoder, cfg.model.latent_updates, cfg.model.learn_origin).to(device)

    print(model)
    print(f'# of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=1e-3)

    try:
        print('TRAINING')
        for epoch in range(cfg.training.epochs):
            train_latents, train_labels = train(
                model, train_batches, model_input, opt, device, epoch, cfg.model.latent_reg, cfg.logging.log_every
            )

            model.eval()

            train_latents = train_latents.numpy()
            train_labels = train_labels.numpy()

            recon_input = model_input.repeat(len(fixed_batch), 1, 1, 1)
            latent = model.infer_latents(recon_input, fixed_batch)[0]
            recon = model.forward(recon_input, latent)

            gt_recon_pairs = torch.stack(list(chain.from_iterable(zip(fixed_batch, recon))))
            save_image(make_grid(gt_recon_pairs.permute(0, 3, 1, 2), normalize=True), recon_dir / f'{epoch:03d}.png')

            if cfg.logging.n_samples_per_epoch:
                cov = np.cov(train_latents.T)
                mean = np.mean(train_latents, 0)

                samples = sample(model, model_input, mean, cov, cfg.logging.n_samples_per_epoch)
                save_image(samples.permute(0, 3, 1, 2), sample_dir / f'{epoch:03d}.png', normalize=True)

                stats = {'cov': cov, 'mean': mean}
                (log_dir / 'stats.p').write_bytes(pickle.dumps(stats))

            (log_dir / 'train_data.p').write_bytes(pickle.dumps({'latents': train_latents, 'labels': train_labels}))
            torch.save(model, log_dir / 'model.p')
    except KeyboardInterrupt:
        print('Interrupting training')

    print('EVALUATION')
    test_latents, test_labels = eval(model, test_batches, model_input, device)

    test_latents = test_latents.numpy()
    test_labels = test_labels.numpy()

    (log_dir / 'test_data.p').write_bytes(pickle.dumps({'latents': test_latents, 'labels': test_labels}))


if __name__ == '__main__':
    main()
