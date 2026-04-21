from __future__ import annotations

import json

from cluster_analyzer.pipeline.relation_preference_pipeline import RelationPreferencePipeline

if __name__ == "__main__":
    pipeline = RelationPreferencePipeline(
        cluster_report_path=r"C:\SCIoI\Labrotation_SceneGraph\clustering\results\kMedoids_results\cluster_report.json",
        scene_relations_path=r"C:\SCIoI\Labrotation_SceneGraph\embedding\results\all_predicates_dict.json",
        temporal_relations_path=r"C:\SCIoI\Labrotation_SceneGraph\embedding\results\temporal_relation_dict.json",
        output_dir=r"/cluster_analyzer\results\cluster_preference_outputs",
    )
    print(json.dumps(pipeline.run(), indent=2))



