import json

from clustering.pipeline.cluster_pipeline import PredicateClusteringPipeline

if __name__ == "__main__":
    pipeline = PredicateClusteringPipeline(
        embedding_json_path=r"C:\SCIoI\Labrotation_SceneGraph\embedding\results\predicate_to_glove_300d.json",
        output_dir=r"C:\SCIoI\Labrotation_SceneGraph\clustering\results\kMedoids_results",


        mode="fixed", # fixed or search
        fixed_k=10,
        distance_metric="cosine",
        #fixed_k=None #this works in search mode
        #k_min=10,#this works in search mode
        #k_max=10,#this works in search mode
        l2_normalize=True,
        random_state=42,
        max_iter=300,
        tsne_perplexity=30.0,
        tsne_max_iter=1000,
    )

    summary = pipeline.run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))