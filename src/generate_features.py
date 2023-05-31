import json
import os
from math import ceil

import click
import numpy as np
import pandas as pd


def _drop_outliers(df, colname, min_val=None, max_val=None):
    counts = df[colname].value_counts()

    if min_val is not None:
        counts = counts[counts >= min_val]

    if max_val is not None:
        counts = counts[counts <= max_val]

    counts = counts.index.to_frame(name=colname).reset_index(drop=True)
    return pd.merge(df, counts, on=[colname])


def _join_features(rel_art_df, rel_lab_df, lab_df, drop_outliers=True):
    # get release_id => label_id
    res_df = pd.merge(
        rel_lab_df[['release_id', 'label_name']],
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
    if drop_outliers:
        res_df = _drop_outliers(res_df, 'label_id', min_val=2, max_val=1000)
        res_df = _drop_outliers(res_df, 'artist_id', min_val=2)

    # get artist idx mapping
    art_emb_idx = {int(v): i for i,v in enumerate(np.sort(res_df['artist_id'].unique()))}
    lab_emb_idx = {int(v): i for i,v in enumerate(np.sort(res_df['label_id'].unique()))}
    return res_df, art_emb_idx, lab_emb_idx


@click.command()
@click.option(
    '--release_genre_file',
    type=str,
    required=True,
    help='Csv file containing the release id of all tracks in a genre.',
)
@click.option(
    '--release_artist_file',
    type=str,
    required=True,
    help='Csv file that contains all the artists with a release in the corpus.',
)
@click.option(
    '--release_label_file',
    type=str,
    required=True,
    help='Csv file that contains all the labels with a release in the corpus.',
)
@click.option(
    '--label_file',
    type=str,
    required=True,
    help='Csv file that contains all the labels with a release in the corpus.',
)
@click.option(
    '--rows_per_chunk',
    type=int,
    default=128000,
    help='Number of rows in each file chunk of the final data.',
)
@click.argument('output_dir', type=str, nargs=1)
def cli(
    release_genre_file,
    release_artist_file,
    release_label_file,
    label_file,
    rows_per_chunk,
    output_dir,
):
    """Apply word2vec on the corpus of release data."""

    # read in genre data
    rel_genre_df = pd.read_csv(release_genre_file)
    # read in artist data, subset to genre data immediately
    rel_art_df = pd.read_csv(release_artist_file, usecols=['release_id', 'artist_id']).drop_duplicates()
    rel_art_df = pd.merge(rel_genre_df, rel_art_df, on=['release_id'])
    # read in label data, subset to genre data immediately
    rel_lab_df = pd.read_csv(release_label_file, usecols=['release_id', 'label_name']).drop_duplicates()
    rel_lab_df = pd.merge(rel_genre_df, rel_lab_df, on=['release_id'])
    # finally read in label data and delete the genre data
    lab_df = pd.read_csv(label_file, usecols=['id', 'name'])

    # preprocess data
    res_df, art_emb_idx, lab_emb_idx = _join_features(rel_art_df, rel_lab_df, lab_df)

    # shuffle data
    res_df = res_df.sample(frac=1.0, replace=False)

    # Save the embedding mappings and final csv
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'art_emb_idx.json'), 'w') as fh:
        json.dump(art_emb_idx, fh)

    with open(os.path.join(output_dir, 'lab_emb_idx.json'), 'w') as fh:
        json.dump(lab_emb_idx, fh)

    for i in range(ceil(len(res_df) / rows_per_chunk)):
        res_df.iloc[i * rows_per_chunk : (i + 1) * rows_per_chunk].to_csv(
            os.path.join(output_dir, f'training_dat_{i}.csv'), sep=',', index=False,
        )


if __name__ == "__main__":
    cli()
