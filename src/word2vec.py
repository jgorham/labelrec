import json
import os
from random import shuffle

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


# Define the Word2Vec model
class Word2Vec(nn.Module):
    def __init__(self, n_artists, n_labels, embedding_dim):
        super(Word2Vec, self).__init__()
        self.n_artists = n_artists
        self.n_labels = n_labels
        self.embedding_dim = embedding_dim
        self.art_embed = nn.Embedding(self.n_artists, self.embedding_dim, dtype=torch.float32)
        self.lab_embed = nn.Embedding(self.n_labels, self.embedding_dim, dtype=torch.float32)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5 / self.embedding_dim
        self.art_embed.weight.data.uniform_(-init_range, init_range)
        self.lab_embed.weight.data.uniform_(-init_range, init_range)

    def forward(self, artist_idx, label_idx, noise_idxs):
        art_embeds = self.art_embed(artist_idx)
        lab_embeds = self.lab_embed(label_idx)
        noise_embeds = self.lab_embed(noise_idxs)
        embed_norm = (
            torch.mean(torch.pow(art_embeds, 2.0))
            + torch.mean(torch.pow(lab_embeds, 2.0))
        )

        scores = (art_embeds * lab_embeds).sum(axis=1, keepdims=True)
        noise_scores = (art_embeds[:, None] * noise_embeds).sum(axis=2)
        return scores, noise_scores, embed_norm


# Define the Noise Contrastive Estimation (NCE) loss function
class NoiseContrastiveLoss(nn.Module):

    def __init__(self, embedding_reg = 0.0):
        super(NoiseContrastiveLoss, self).__init__()
        self.embedding_reg = embedding_reg

    def forward(self, scores, noise_scores, embed_norm):
        batch_size = scores.size(0)
        true_noise_logits = torch.cat([scores, noise_scores], dim=1)
        true_noise_labels = torch.cat([
            torch.ones(scores.shape),
            torch.zeros(noise_scores.shape),
        ], dim=1).to(scores.device)
        loss = nn.CrossEntropyLoss()(true_noise_logits, true_noise_labels) + self.embedding_reg * embed_norm
        return loss


# Custom dataset class for Word2Vec
class Word2VecDataset(IterableDataset):
    def __init__(self, root_dir, art_emb_idx, lab_emb_idx):
        self.root_dir = root_dir
        self.art_emb_idx = art_emb_idx
        self.lab_emb_idx = lab_emb_idx
        self.file_paths = self._get_file_paths()

    def __iter__(self):
        file_paths = [f for f in self.file_paths]
        shuffle(file_paths)
        for file_path in tqdm(file_paths):
            data = pd.read_csv(file_path).sample(frac=1, replace=False)
            data['artist_id'] = data['artist_id'].apply(lambda xx: self.art_emb_idx[xx])
            data['label_id'] = data['label_id'].apply(lambda xx: self.lab_emb_idx[xx])
            values = data[['release_id', 'artist_id', 'label_id']].values.tolist()
            for row in values:
                yield row

    def _get_file_paths(self):
        file_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths


@click.command()
@click.option(
    '--training_data_dir',
    type=str,
    help='Directory that contains features generated for training.',
)
@click.option(
    '--device',
    type=str,
    default='cpu',
    help='Device to train on.'
)
@click.option(
    '--from-model',
    type=str,
    help='Previous model to use for a warm start..'
)
@click.option(
    '--embedding_dim',
    type=int,
    default=128,
    help='The dimension of the embedding vectors.',
)
@click.option(
    '--n_epochs',
    type=int,
    default=10,
    help='The number of training epochs.',
)
@click.option(
    '--n_negative_samples',
    type=int,
    default=10,
    help='The number of negative samples per positive sample in NCE loss.',
)
@click.option(
    '--batch_size',
    type=int,
    default=256,
    help='The batch size of a gradient step.',
)
@click.option(
    '--learning_rate',
    type=float,
    default=1e-3,
    help='The learning rate for SGD.',
)
@click.option(
    '--embedding_reg',
    type=float,
    default=5e-7,
    help='The regularization applied to all embedding vectors.',
)
@click.argument('output_dir', type=str, nargs=1)
def cli(
    training_data_dir,
    device,
    from_model,
    embedding_dim,
    n_epochs,
    n_negative_samples,
    batch_size,
    learning_rate,
    embedding_reg,
    output_dir,
):
    """Apply word2vec on the corpus of release data."""

    # preprocess data
    with open(os.path.join(training_data_dir, 'art_emb_idx.json'), 'r') as fh:
        art_emb_idx = json.load(fh)
        art_emb_idx = {int(k): v for k, v in art_emb_idx.items()}

    with open(os.path.join(training_data_dir, 'lab_emb_idx.json'), 'r') as fh:
        lab_emb_idx = json.load(fh)
        lab_emb_idx = {int(k): v for k, v in lab_emb_idx.items()}

    n_artists = len(art_emb_idx)
    n_labels = len(lab_emb_idx)

    # get training data
    dataset = Word2VecDataset(
        training_data_dir,
        art_emb_idx,
        lab_emb_idx,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    # construct model
    model = Word2Vec(n_artists, n_labels, embedding_dim)
    if from_model is not None:
        from_state_dict = torch.load(from_model)
        model.load_state_dict(from_state_dict)

    # Define the loss function and the optimizer
    criterion = NoiseContrastiveLoss(embedding_reg=embedding_reg)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        for release_idx, artist_idx, label_idx in dataloader:
            noise_idxs = torch.multinomial(
                input=torch.ones(n_labels),
                num_samples=batch_size * n_negative_samples,
                replacement=True,
            ).view((batch_size, n_negative_samples))

            artist_idx = artist_idx.to(device)
            label_idx = label_idx.to(device)
            noise_idxs = noise_idxs.to(device)

            optimizer.zero_grad()
            scores, noise_scores, embed_norm = model(artist_idx, label_idx, noise_idxs)
            loss = criterion(scores, noise_scores, embed_norm)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print("Epoch {}, Loss: {:.4f}".format(epoch + 1, total_loss / n_batches))

    # Save the model and the embedding mappings
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'embedding.model'))

    with open(os.path.join(output_dir, 'art_emb_idx_inv.json'), 'w') as fh:
        json.dump({v: k for k, v in art_emb_idx.items()}, fh)

    with open(os.path.join(output_dir, 'lab_emb_idx_inv.json'), 'w') as fh:
        json.dump({v: k for k, v in lab_emb_idx.items()}, fh)


if __name__ == "__main__":
    cli()
