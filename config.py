NODE_FEATURE_DIMENSION = 20
EDGE_FEATURE_DIMENSION = 17
MAX_NUM_PRIMITIVES = 24
MAX_NUM_CONSTRAINTS = 208
NUM_PRIMITIVE_TYPES = 5
NUM_CONSTRAINT_TYPES = 9

# NodeBool   = slice(start = 0, stop = 1, step = 1)
# NodeType   = slice(start = 1, stop = 7, step = 1)
# NodeParams = slice(start = 7, stop = 20, step = 1)
# EdgeSubA   = slice(start = 0, stop = 4, step = 1)
# EdgeSubB   = slice(start = 4, stop = 8, step = 1)
# EdgeType   = slice(start = 8, stop = 17, step = 1)

log_clip = -10

node_bce_weight = 1.0
node_cross_weight = 1.0
node_mse_weight = 16.0

edge_suba_weight = 0.1
edge_subb_weight = 0.1
edge_constraint_weight = 0.1

kld_weight = .001
reg_weight = .01

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