import json
import os

import torch


def glove_embedding(path, predicate_set):
    wv_dict, wv_arr, wv_size = torch.load(path, weights_only=False)

    vectors = torch.Tensor(len(predicate_set), wv_size)
    vectors.normal_(0, 1)

    for i, token in enumerate(predicate_set):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print(f"{token} -> {lw_token}")
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print(f"fail on {token}")
    return vectors



#with open(r"/home/shokoofeh/Labrotation_SceneGraph/embedding/demo.json", "r", encoding="utf-8") as f:
with open(r"C:\SCIoI\Labrotation_SceneGraph\embedding\results\demo.json", "r", encoding="utf-8") as f:
    data = json.load(f)
all_predicates=[value for value in data.values() if value not in ("False", "None", False, None)]
# all predicates, including duplicates
#all_predicates = data["all_predicate"]
print("Len of all Predicators", len(all_predicates))
# unique predicates only
unique_predicates = sorted(set(all_predicates))
print("Len unique Predicators:", len(unique_predicates))
print(all_predicates)



glove_path_file = r"/home/shokoofeh/Labrotation_SceneGraph/embedding/pretrained/glove_6B.pt"

vectors = glove_embedding(glove_path_file, unique_predicates)
print(unique_predicates)
print(vectors.shape)

# build output json structure
output = {
    "num_predicates": len(unique_predicates),
    "embedding_dim": vectors.shape[1],
    "predicates": unique_predicates,
    "vectors": vectors.tolist()
}
embedding_map = {
    pred: vec for pred, vec in zip(unique_predicates, vectors.tolist())
}

output_path = "/home/shokoofeh/Labrotation_SceneGraph/embedding/predicate_to_glove_300d.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(embedding_map, f, ensure_ascii=False, indent=2)

print(f"Saved predicate-to-vector mapping to: {output_path}")
