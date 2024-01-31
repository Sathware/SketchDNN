NODE_FEATURE_DIMENSION = 20
EDGE_FEATURE_DIMENSION = 17
MAX_NUM_PRIMITIVES = 24
MAX_NUM_CONSTRAINTS = 208
NUM_PRIMITIVE_TYPES = 5
NUM_CONSTRAINT_TYPES = 9

HYPER_PARAMETERS = {
    "graph embedding dimension": 512,
    "device" : 'cuda',
    "node feature dimension": NODE_FEATURE_DIMENSION,
    "edge feature dimension": EDGE_FEATURE_DIMENSION,
    "encoder node mlp intermediate dims": [32, 24],
    "encoder node mlp out dim": 16,
    "encoder edge mlp intermediate dims": [16, 8],
    "encoder edge mlp out dim": 4,
    "encoder num attention heads": 4,
    "encoder num transformer layers": 4,
    "encoder mean mlp intermediate dims": [64, 256],
    "encoder logvar mlp intermediate dims": [64, 256],
    "decoder node create mlp intermediate dims": [32, 24],
    "decoder node create mlp out dim": 16,
    "decoder edge create mlp intermediate dims": [16, 8],
    "decoder edge create mlp out dim": 4,
    "decoder num attention heads": 4,
    "decoder num transformer layers": 4,
    "decoder node out mlp intermediate dims": [32, 24],
    "decoder edge out mlp intermediate dims": [16, 8],
}