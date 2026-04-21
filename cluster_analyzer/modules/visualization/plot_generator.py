from __future__ import annotations

from pathlib import Path
from typing import List, Counter, Dict

import matplotlib.pyplot as plt
import pandas as pd

from cluster_analyzer.modules.datamodel.cluster_info import ClusterInfo
from cluster_analyzer.modules.utility import normalize_relation

try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


class PlotGenerator:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_top_metric(self, df: pd.DataFrame, all_assignment,
                        temporal_assignment, n_participants, metric: str,
                        title: str, ylabel: str, filename: str, top_k: int = 12) -> Path:
        plot_df = df.sort_values(metric, ascending=False).head(top_k)
        #plt.figure(figsize=(10, 6))
        plt.figure(figsize=(20, 10))
        #plt.figure(figsize=(10, 20))

        plt.bar(plot_df["cluster_name"].astype(str), plot_df[metric])
        # plt.bar(plot_df["representatives"].astype(str), plot_df[metric])
        plt.xlabel("Cluster Name")
        plt.ylabel(ylabel)
        if metric == "p_saccade_given_cluster":
            baseline = (temporal_assignment * 1000) / (all_assignment * n_participants)
            plt.axhline(y=baseline, color="r", linestyle="--")
            plt.text(
                x=0,
                y=baseline,
                s=f"{baseline:.4f} %",
                va="bottom",
                ha="right"
            )
        plt.title(title)
        plt.tight_layout()
        out = self.output_dir / filename
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        return out

    def plot_scene_vs_temporal_counts(self, df: pd.DataFrame, filename: str = "scene_vs_temporal_counts.png",
                                      top_k: int = 12) -> Path:
        plot_df = df.sort_values("n_temporal", ascending=False).head(top_k)
        x = range(len(plot_df))
        width = 0.4
        plt.figure(figsize=(20, 10))
        plt.bar([i - width / 2 for i in x], plot_df["n_all_scene"], width=width, label="All scene-graph")
        plt.bar([i + width / 2 for i in x], plot_df["n_temporal"], width=width, label="Temporal gaze")
        plt.xticks(list(x), plot_df["cluster_name"].astype(str))
        plt.xlabel("Cluster Name")
        plt.ylabel("Assignment count")
        plt.title("Scene-graph vs Temporal Gaze Assignments per Cluster")
        plt.legend()
        plt.tight_layout()
        out = self.output_dir / filename
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        return out

    def plot_availability_vs_selection(self, df: pd.DataFrame,
                                       filename: str = "availability_vs_selection_scatter.png") -> Path:
        plt.figure(figsize=(8, 6))
        plt.scatter(df["p_all_scene"], df["p_temporal"])
        max_val = max(df["p_all_scene"].max(), df["p_temporal"].max(), 1e-6)
        plt.plot([0, max_val], [0, max_val], linestyle="--")
        for _, row in df.iterrows():
            plt.annotate(str(row["representatives"]), (row["p_all_scene"], row["p_temporal"]))
        plt.xlabel("Scene availability proportion")
        plt.ylabel("Temporal gaze proportion")
        plt.title("Availability vs Selection by Cluster")
        plt.tight_layout()
        out = self.output_dir / filename
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        return out

    def generate_cluster_wordclouds(self, relation_df: pd.DataFrame, metric: str = "n_temporal",
                                    top_k_clusters: int = 12, min_value: float = 1.0) -> List[Path]:
        out_paths: List[Path] = []
        cluster_order = relation_df.groupby("cluster_id")["n_temporal"].sum().sort_values(ascending=False).head(
            top_k_clusters).index.tolist()
        wc_dir = self.output_dir / f"wordclouds_{metric}"
        wc_dir.mkdir(parents=True, exist_ok=True)
        for cluster_id in cluster_order:
            sub = relation_df[relation_df["cluster_id"] == cluster_id].copy()
            cluster_name = sub["cluster_name"].iloc[0]
            sub = sub[sub[metric] >= min_value]
            freqs = {row["relation"]: float(row[metric]) for _, row in sub.iterrows() if float(row[metric]) > 0}
            if not freqs:
                continue
            out = wc_dir / f"cluster_{cluster_id}_wordcloud.png"
            if WORDCLOUD_AVAILABLE:
                wc = WordCloud(width=900, height=500, background_color="white",
                               collocations=False).generate_from_frequencies(freqs)
                plt.figure(figsize=(10, 5.5))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                # plt.title(f"Cluster {relation_df[relation_df["cluster_name"]]} word cloud ({metric})")
                plt.title(f"Cluster: \" {cluster_name}\"({metric})")
                plt.tight_layout()
                plt.savefig(out, dpi=200, bbox_inches="tight")
                plt.close()
            else:
                bar_df = pd.DataFrame({"relation": list(freqs.keys()), "value": list(freqs.values())}).sort_values(
                    "value", ascending=True).tail(12)
                plt.figure(figsize=(9, 6))
                plt.barh(bar_df["relation"], bar_df["value"])
                plt.xlabel(metric)
                plt.title(f"Cluster {cluster_id} relation sizes ({metric})")
                plt.tight_layout()
                plt.savefig(out, dpi=200, bbox_inches="tight")
                plt.close()
            out_paths.append(out)
        return out_paths

    def plot_cluster_relation_count_heatmap(
            self,
            relation_df: pd.DataFrame,
            value_col: str = "n_temporal",
            top_relations: int = 40,
            filename: str = "cluster_relation_count_heatmap.png",
    ) -> Path:
        heatmap_df = (
            relation_df.groupby(["relation", "cluster_id"])[value_col]
            .sum()
            .unstack(fill_value=0)
        )

        top_idx = heatmap_df.sum(axis=1).sort_values(ascending=False).head(top_relations).index
        heatmap_df = heatmap_df.loc[top_idx]

        try:
            heatmap_df = heatmap_df[sorted(heatmap_df.columns, key=lambda x: int(x))]
        except Exception:
            heatmap_df = heatmap_df[sorted(heatmap_df.columns)]

        plt.figure(figsize=(12, max(6, len(heatmap_df) * 0.35)))
        plt.imshow(heatmap_df.values, aspect="auto")
        plt.colorbar(label=value_col)

        plt.xticks(
            ticks=range(len(heatmap_df.columns)),
            labels=[str(c) for c in heatmap_df.columns],
            rotation=90
        )
        plt.yticks(
            ticks=range(len(heatmap_df.index)),
            labels=list(heatmap_df.index)
        )

        plt.xlabel("Cluster ID")
        plt.ylabel("Relation")
        plt.title(f"Cluster–Relation Count Heatmap ({value_col})")
        plt.tight_layout()

        out = self.output_dir / filename
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        return out

    def generate_cluster_report_wordclouds(
            self,
            clusters: Dict[str, ClusterInfo],
            max_clusters: int | None = 10,
    ) -> List[Path]:
        """
        Word clouds from the original cluster_report members.
        This visualizes the semantic content of each cluster.
        """

        out_paths: List[Path] = []
        wc_dir = self.output_dir / "wordclouds_cluster_report_members"
        wc_dir.mkdir(parents=True, exist_ok=True)

        cluster_items = sorted(
            clusters.items(),
            key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])
        )
        if max_clusters is not None:
            cluster_items = cluster_items[:max_clusters]

        for cluster_id, cluster in cluster_items:
            members = [normalize_relation(x) for x in cluster.members if str(x).strip()]
            if not members:
                continue

            # frequency inside cluster report
            freqs = dict(Counter(members))

            out = wc_dir / f"cluster_{cluster_id}_report_wordcloud.png"

            if WORDCLOUD_AVAILABLE:
                wc = WordCloud(
                    width=900,
                    height=500,
                    background_color="white",
                    collocations=False,
                ).generate_from_frequencies(freqs)

                plt.figure(figsize=(10, 5.5))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"Cluster:\"{cluster.name}\"")
                plt.tight_layout()
                plt.savefig(out, dpi=200, bbox_inches="tight")
                plt.close()
            else:
                # fallback
                bar_df = (
                    pd.DataFrame({
                        "member": list(freqs.keys()),
                        "value": list(freqs.values())
                    })
                    .sort_values("value", ascending=True)
                    .tail(15)
                )

                plt.figure(figsize=(9, 6))
                plt.barh(bar_df["member"], bar_df["value"])
                plt.xlabel("count in cluster report")
                plt.title(f"Cluster {cluster_id} semantic members")
                plt.tight_layout()
                plt.savefig(out, dpi=200, bbox_inches="tight")
                plt.close()

            out_paths.append(out)

        return out_paths
