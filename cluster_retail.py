"""Reproduce the clustering workflow from Appendix 2 of the thesis."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

COLORS = ("red", "blue", "green", "cyan", "magenta", "gray", "orange", "brown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input workbook (.xlsx)")
    parser.add_argument("--sheet", default="Лист6", help="Excel sheet name")
    parser.add_argument("--output-dir", type=Path, default=Path("cluster-output"))
    args = parser.parse_args()

    dataset = pd.read_excel(args.input, sheet_name=args.sheet)
    if len(dataset) < 10 or dataset.shape[1] < 6:
        parser.error("the input sheet needs at least 10 rows and 6 columns")

    try:
        features = dataset.iloc[:, [4, 5]].to_numpy(dtype=float)
    except ValueError:
        parser.error("columns 5 and 6 must contain numeric values")
    if not np.isfinite(features).all():
        parser.error("columns 5 and 6 must contain finite values for every row")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_counts = range(1, 11)
    inertia = []
    for count in cluster_counts:
        model = KMeans(n_clusters=count, init="k-means++", n_init=10, random_state=42)
        model.fit(features)
        inertia.append(model.inertia_)

    fig, ax = plt.subplots()
    ax.plot(cluster_counts, inertia)
    ax.set(title="The Elbow Method", xlabel="Number of clusters", ylabel="SSE")
    fig.savefig(args.output_dir / "elbow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    model = KMeans(n_clusters=8, init="k-means++", n_init=10, random_state=100)
    clusters = model.fit_predict(features)

    fig, ax = plt.subplots()
    for cluster, color in enumerate(COLORS):
        points = features[clusters == cluster]
        ax.scatter(points[:, 0], points[:, 1], s=100, c=color, label=f"Cluster {cluster + 1}")
    centers = model.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], s=200, c="yellow", label="Centroids")
    ax.set(
        title="Кластеризация предприятий ритейла",
        xlabel="Сред. коэф. автономии, %",
        ylabel="ROE, %",
    )
    ax.legend(bbox_to_anchor=(1, 0.8))
    fig.savefig(args.output_dir / "retail-clusters.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
