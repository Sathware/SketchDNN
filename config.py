NODE_FEATURE_DIMENSION = 20
EDGE_FEATURE_DIMENSION = 17
MAX_NUM_PRIMITIVES = 24
MAX_NUM_CONSTRAINTS = 208
NUM_PRIMITIVE_TYPES = 5
NUM_CONSTRAINT_TYPES = 9

GRAPH_EMBEDDING_SIZE = 512

HYPER_PARAMETERS = {
    "graph embedding dimension": GRAPH_EMBEDDING_SIZE,
    "node feature dimension": NODE_FEATURE_DIMENSION,
    "edge feature dimension": EDGE_FEATURE_DIMENSION,
    # Encoder Hyper Parameters
    "encoder node mlp_in dims": [NODE_FEATURE_DIMENSION, 64, NODE_FEATURE_DIMENSION],
    "encoder edge mlp_in dims": [EDGE_FEATURE_DIMENSION, 32, EDGE_FEATURE_DIMENSION],
    "encoder tf params": [
        {"heads":8,"node mlp_attn dims":[NODE_FEATURE_DIMENSION, 32, 12],"node mlp_out dims":[12, 32, NODE_FEATURE_DIMENSION], "edge mlp_attn dims":[EDGE_FEATURE_DIMENSION, 24, 8],"edge mlp_out dims":[8, 24, EDGE_FEATURE_DIMENSION]},
        {"heads":8,"node mlp_attn dims":[NODE_FEATURE_DIMENSION, 32, 12],"node mlp_out dims":[12, 32, NODE_FEATURE_DIMENSION], "edge mlp_attn dims":[EDGE_FEATURE_DIMENSION, 24, 8],"edge mlp_out dims":[8, 24, EDGE_FEATURE_DIMENSION]},
        {"heads":8,"node mlp_attn dims":[NODE_FEATURE_DIMENSION, 32, 12],"node mlp_out dims":[12, 32, NODE_FEATURE_DIMENSION], "edge mlp_attn dims":[EDGE_FEATURE_DIMENSION, 24, 8],"edge mlp_out dims":[8, 24, EDGE_FEATURE_DIMENSION]},
        {"heads":8,"node mlp_attn dims":[NODE_FEATURE_DIMENSION, 32, 12],"node mlp_out dims":[12, 32, NODE_FEATURE_DIMENSION], "edge mlp_attn dims":[EDGE_FEATURE_DIMENSION, 24, 8],"edge mlp_out dims":[8, 24, EDGE_FEATURE_DIMENSION]},
        ],
    "encoder node mlp_out dims": [NODE_FEATURE_DIMENSION * MAX_NUM_PRIMITIVES, 512, 32, 64],
    "encoder edge mlp_out dims": [EDGE_FEATURE_DIMENSION * MAX_NUM_PRIMITIVES * MAX_NUM_PRIMITIVES, 4096, 2048, 64],
    "encoder embedding mlp dims": [128, 512, 1024, GRAPH_EMBEDDING_SIZE],
    # Decoder Hyper Parameters
    "decoder embedding mlp dims": [GRAPH_EMBEDDING_SIZE, 1024, 2048],
    "decoder node mlp_in dims": [2048, NODE_FEATURE_DIMENSION * MAX_NUM_PRIMITIVES],
    "decoder edge mlp_in dims": [2048, 4096, EDGE_FEATURE_DIMENSION * MAX_NUM_PRIMITIVES * MAX_NUM_PRIMITIVES],
    "decoder tf heads": [
        {"heads":8,"node mlp_attn dims":[NODE_FEATURE_DIMENSION, 32, 12],"node mlp_out dims":[NODE_FEATURE_DIMENSION, 32, 12], "edge mlp_in dims":[EDGE_FEATURE_DIMENSION, 24, 8],"edge mlp_out dims":[EDGE_FEATURE_DIMENSION, 24, 8]},
        {"heads":8,"node mlp_attn dims":[NODE_FEATURE_DIMENSION, 32, 12],"node mlp_out dims":[NODE_FEATURE_DIMENSION, 32, 12], "edge mlp_in dims":[EDGE_FEATURE_DIMENSION, 24, 8],"edge mlp_out dims":[EDGE_FEATURE_DIMENSION, 24, 8]},
        {"heads":8,"node mlp_attn dims":[NODE_FEATURE_DIMENSION, 32, 12],"node mlp_out dims":[NODE_FEATURE_DIMENSION, 32, 12], "edge mlp_in dims":[EDGE_FEATURE_DIMENSION, 24, 8],"edge mlp_out dims":[EDGE_FEATURE_DIMENSION, 24, 8]},
        {"heads":8,"node mlp_attn dims":[NODE_FEATURE_DIMENSION, 32, 12],"node mlp_out dims":[NODE_FEATURE_DIMENSION, 32, 12], "edge mlp_in dims":[EDGE_FEATURE_DIMENSION, 24, 8],"edge mlp_out dims":[EDGE_FEATURE_DIMENSION, 24, 8]},
        ],
    "decoder node mlp_out dims": [NODE_FEATURE_DIMENSION, 64, NODE_FEATURE_DIMENSION],
    "decoder edge mlp_out dims": [EDGE_FEATURE_DIMENSION, 64, EDGE_FEATURE_DIMENSION],
}