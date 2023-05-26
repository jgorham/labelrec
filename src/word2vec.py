import json
import os

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset


def _join_features(rel_art_df, rel_lab_df, lab_df):
    # get release_id => label_id
    res_df = pd.merge(
        rel_lab_df[['release_id', 'label_name']].drop_duplicates(),
        lab_df[['id', 'name']].rename(columns=lambda col: f'label_{col}'),
        on=['label_name'],
    )
    res_df = res_df[['release_id', 'label_id']]
    # join in artist_id
    res_df = pd.merge(
        res_df,
        rel_art_df[['release_id', 'artist_id']],
        on=['release_id'],
    )
    # get artist idx mapping
    art_emb_idx = {int(v): i for i,v in enumerate(np.sort(res_df['artist_id'].unique()))}
    lab_emb_idx = {int(v): i for i,v in enumerate(np.sort(res_df['label_id'].unique()))}
    return res_df, art_emb_idx, lab_emb_idx


# Define the Word2Vec model
class Word2Vec(nn.Module):
    def __init__(self, n_artists, n_labels, embedding_dim):
        super(Word2Vec, self).__init__()
        self.n_artists = n_artists
        self.n_labels = n_labels
        self.embedding_dim = embedding_dim
        self.art_embed = nn.Embedding(self.n_artists, self.embedding_dim)
        self.lab_embed = nn.Embedding(self.n_labels, self.embedding_dim)
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
class Word2VecDataset(Dataset):
    def __init__(self, data, artist_emb_idx, label_emb_idx):
        self.data = data
        self.artist_emb_idx = artist_emb_idx
        self.label_emb_idx = label_emb_idx

    def __getitem__(self, index):
        artist_id = self.artist_emb_idx[self.data[index][0]]
        label_id = self.label_emb_idx[self.data[index][1]]
        return artist_id, label_id

    def __len__(self):
        return len(self.data)


@click.command()
@click.option(
    '--release_file',
    type=str,
    help='Csv file that contains all the releases in the corpus.',
)
@click.option(
    '--release_artist_file',
    type=str,
    help='Csv file that contains all the artists with a release in the corpus.',
)
@click.option(
    '--release_label_file',
    type=str,
    help='Csv file that contains all the labels with a release in the corpus.',
)
@click.option(
    '--label_file',
    type=str,
    help='Csv file that contains all the labels with a release in the corpus.',
)
@click.option(
    '--device',
    type=str,
    default='cpu',
    help='Device to train on.'
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
    default=32,
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
    default=1e-6,
    help='The regularization applied to all embedding vectors.',
)
@click.argument('output_dir', type=str, nargs=1)
def cli(
    release_file,
    release_artist_file,
    release_label_file,
    label_file,
    device,
    embedding_dim,
    n_epochs,
    n_negative_samples,
    batch_size,
    learning_rate,
    embedding_reg,
    output_dir,
):
    """Apply word2vec on the corpus of release data."""

    # rel_df = pd.read_csv(release_file)
    rel_art_df = pd.read_csv(release_artist_file)
    rel_lab_df = pd.read_csv(release_label_file)
    lab_df = pd.read_csv(label_file)

    # preprocess data
    res_df, art_emb_idx, lab_emb_idx = _join_features(rel_art_df, rel_lab_df, lab_df)
    n_artists = len(art_emb_idx)
    n_labels = len(lab_emb_idx)

    # get training data
    dataset = Word2VecDataset(
        res_df[['artist_id', 'label_id']].values,
        art_emb_idx,
        lab_emb_idx,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # construct model
    model = Word2Vec(n_artists, n_labels, embedding_dim)

    # Define the loss function and the optimizer
    criterion = NoiseContrastiveLoss(embedding_reg=embedding_reg)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(n_epochs):
        total_loss = 0.0
        for artist_idx, label_idx in dataloader:
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

        print("Epoch {}, Loss: {:.4f}".format(epoch+1, total_loss / len(dataloader)))

    # Save the model and the embedding mappings
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'embedding.model'))

    with open(os.path.join(output_dir, 'art_emb_idx.json'), 'w') as fh:
        json.dump(art_emb_idx, fh)

    with open(os.path.join(output_dir, 'lab_emb_idx.json'), 'w') as fh:
        json.dump(lab_emb_idx, fh)


if __name__ == "__main__":
    cli()
