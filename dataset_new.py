# %%
import os.path
import torch
from matplotlib import pyplot as plt
import numpy as np
from torch_geometric.data import Dataset, download_url, Data
import torch_geometric.utils as pygutils
from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MAX_NUM_PRIMITIVES, MAX_NUM_CONSTRAINTS
# You have to import sketchgraphs this way otherwise you get type errors
os.chdir('SketchGraphs/')
import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data._entity import Point, Line, Circle, Arc, EntityType
from sketchgraphs.data.sketch import Sketch
from sketchgraphs.data._constraint import *
os.chdir('../')

import math

# %%
class SketchDataset(Dataset):
    def __init__(self, root, transform = None, pre_transform = None, pre_filter = None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['sg_all.npy']
    
    @property 
    def processed_file_names(self):
        # Make processed fir if not already exist
        if os.path.exists(self.processed_dir):
            return os.listdir(self.processed_dir)
        else:
            return []
    
    @property
    def num_node_features(self):
        return 20
    
    @property
    def num_edge_features(self):
        return 17
    
    def download(self):
        path = download_url(url = "https://sketchgraphs.cs.princeton.edu/sequence/sg_all.npy", 
                            folder = self.raw_dir
                           );
        # print("Downloaded SketchGraphs dataset to " + path)

    def process(self):
        # Change dir to SketchGraphs so module not found error doesn't popup
        os.chdir('SketchGraphs/')
        # Load SketchGraphs sequence data dictionary
        seq_data = flat_array.load_dictionary_flat(os.path.join("../", self.raw_paths[0]))
        sequences = seq_data["sequences"]
        idx = 0
        for i in range(len(sequences)):
            if idx % 10000 == 0:
                print("Saved Graphs: ", idx, "\t", "Processed Sketches: ", (100*i+100)/len(sequences), "%")
            
            seq = sequences[i]
            sketch = datalib.sketch_from_sequence(seq)
            # Filter out sketches with less than 7 primitives or more than 24 primitives or more than 208 constraints
            if len(sketch.entities) < 7 or len(sketch.entities) > 24 or len(sketch.constraints) > 208:
                continue
            
            node_features, adjacency_list, edge_features = SketchDataset.sketch_to_graph(sketch)
            # node_features, adjacency_list = SketchDataset.sort_graph(node_features, adjacency_list)
            data = Data(x = node_features, edge_index = adjacency_list, edge_attr = edge_features)
            torch.save(data, os.path.join("../", self.processed_dir, f'data_{idx}.pt'))
            idx += 1
        # Change dir back
        os.chdir('../')
        
    @staticmethod
    def sketch_to_graph(sketch):
        # Setup output data structures
        num_nodes = len(sketch.entities)
        num_edges = len(sketch.constraints)
        # Node feature matrix
        node_matrix = torch.zeros(size=(num_nodes if num_nodes < MAX_NUM_PRIMITIVES else MAX_NUM_PRIMITIVES, NODE_FEATURE_DIMENSION));
        # Adjacency matrix
        edge_index = torch.zeros(size=(num_edges if num_edges < MAX_NUM_CONSTRAINTS else MAX_NUM_CONSTRAINTS, 2));
        # Edge feature matrix
        edge_attr = torch.zeros(size=(num_edges if num_edges < MAX_NUM_CONSTRAINTS else MAX_NUM_CONSTRAINTS, EDGE_FEATURE_DIMENSION));
        # Build node feature matrix according to the schema outlined in paper
        idx = 0;
        node_ref_to_idx = {};
        for key, value in sketch.entities.items():
            # Enforce maximum 24 primitives limit
            if (idx == MAX_NUM_PRIMITIVES):
                break
            
            node_feature = torch.zeros(20);
            node_ref_to_idx[key] = idx;
            node_feature[0] = int(value.isConstruction);
            match value.type:
                case EntityType.Line:
                    node_feature[1] = 1;
                    node_feature[6:8] = torch.from_numpy(value.start_point);
                    node_feature[8:10] = torch.from_numpy(value.end_point);
                case EntityType.Circle:
                    node_feature[2] = 1;
                    node_feature[10:12] = torch.from_numpy(value.center_point);
                    node_feature[12] = value.radius;
                case EntityType.Arc:
                    node_feature[3] = 1;
                    node_feature[13:15] = torch.from_numpy(value.center_point);
                    node_feature[15] = value.radius;
                    angle_start_offset = value.endParam if value.clockwise else value.startParam;
                    angle_end_offset = value.startParam if value.clockwise else value.endParam;
                    angle = math.atan2(value.yDir, value.xDir);
                    node_feature[16] = ((angle + angle_start_offset) % (2*math.pi)) / (2*math.pi)
                    node_feature[17] = ((angle + angle_end_offset) % (2*math.pi)) / (2*math.pi)
                case EntityType.Point:
                    node_feature[4] = 1;
                    node_feature[18] = value.x;
                    node_feature[19] = value.y;
                case _:
                    continue
                
            node_matrix[idx] = node_feature
            idx += 1
        # Remove all unused entries before returning, since there are a variable number of relevant primitives per sketch
        node_matrix = node_matrix[:idx]
        # Build adjacency list and edge feature matrix
        idx = 0;
        edge_exists = {}
        for value in sketch.constraints.values():
            # Enforce maximum 288 constraints limit
            if (idx == MAX_NUM_CONSTRAINTS):
                break
            edge_feature = torch.zeros(17);
            # Set one hot encoding for constraint type
            match value.type:
                case ConstraintType.Coincident:
                    edge_feature[8] = 1;
                case ConstraintType.Horizontal:
                    edge_feature[9] = 1;
                case ConstraintType.Vertical:
                    edge_feature[10] = 1;
                case ConstraintType.Parallel:
                    edge_feature[11] = 1;
                case ConstraintType.Perpendicular:
                    edge_feature[12] = 1;
                case ConstraintType.Tangent:
                    edge_feature[13] = 1;
                case ConstraintType.Midpoint:
                    edge_feature[14] = 1;
                case ConstraintType.Equal:
                    edge_feature[15] = 1;
                case _:
                    continue;
            connection = value.get_references()
            node_a_ref = connection[0].split('.')
            # Constraint references irrelevant primitive -----
            if node_a_ref[0] not in node_ref_to_idx:
                continue
            node_a_idx = node_ref_to_idx[node_a_ref[0]]
            # Add one hot encoding for where the constraint is applied for primitive
            if len(node_a_ref) == 2:
                    match node_a_ref[1]:
                        case "start":
                            edge_feature[0] = 1
                        case "center":
                            edge_feature[1] = 1
                        case "end":
                            edge_feature[2] = 1
            else:
                edge_feature[3] = 1
            # If constraint only applies to 1 primitive
            if len(connection) == 1:
                # Multi graphs are not supported -----
                if (node_a_idx, node_a_idx) in edge_exists:
                    continue
                # Add a self loop on node
                edge_index[idx] = torch.tensor([node_a_idx, node_a_idx])
                edge_exists[(node_a_idx, node_a_idx)] = True
                # Save edge feature vector
                edge_attr[idx] = edge_feature
                idx += 1
                continue
            # If constraint applies to 2 primitives
            node_b_ref = connection[1].split('.')
            # Constraint references irrelevant primitive -----
            if node_b_ref[0] not in node_ref_to_idx:
                continue
            node_b_idx = node_ref_to_idx[node_b_ref[0]]
            # Add one hot encoding for where the constraint is applied for second primitive
            if len(node_b_ref) == 2:
                match node_b_ref[1]:
                    case "start":
                        edge_feature[4] = 1
                    case "center":
                        edge_feature[5] = 1
                    case "end":
                        edge_feature[6] = 1
            else:
                edge_feature[7] = 1
            # Multi graphs are not supported -----
            if (node_a_idx, node_b_idx) in edge_exists:
                continue
            # Add an edge between the 2 nodes
            edge_index[idx] = torch.tensor([node_a_idx, node_b_idx])
            edge_exists[(node_a_idx, node_b_idx)] = True
            # Save edge feature vector
            edge_attr[idx] = edge_feature
            idx += 1
        # Remove all unused adjacency info since there are a variable number of relevant constraints per sketch
        edge_index = edge_index[:idx]
        edge_attr = edge_attr[:idx]
        return node_matrix, edge_index.T.contiguous(), edge_attr

    @staticmethod
    def graph_to_sketch(node_matrix, edge_index, edge_attr):
        sketch = Sketch()
        # Add entities
        for idx in range(len(node_matrix)):
            entity = node_matrix[idx]
            match torch.argmax(entity[1:5]):
                case 0:
                    # Create Line
                    id = str(idx + 1)
                    isConstructible = bool(entity[0])
                    pnt = entity[6:8]
                    startParam = 0
                    dir = (entity[8:10] - entity[6:8]) / torch.linalg.vector_norm(entity[8:10] - entity[6:8])
                    endParam = torch.linalg.vector_norm(entity[8:10] - entity[6:8])
                    line = Line(entityId = id,
                                isConstruction = isConstructible, 
                                pntX = pnt[0], 
                                pntY = pnt[1], 
                                dirX = dir[0], 
                                dirY = dir[1], 
                                startParam = startParam, 
                                endParam = endParam
                               );
                    sketch.entities[id] = line
                case 1:
                    # Create Circle
                    id = str(idx + 1)
                    isConstructible = bool(entity[0])
                    center = entity[10:12]
                    radius = entity[12]
                    circle = Circle(entityId = id, 
                                  isConstruction = isConstructible, 
                                  xCenter = center[0], 
                                  yCenter = center[1], 
                                  xDir = 1, 
                                  yDir = 0, 
                                  radius = radius, 
                                  clockwise = False
                                 );
                    sketch.entities[id] = circle
                case 2: 
                    # Create Arc
                    id = str(idx + 1)
                    isConstructible = bool(entity[0])
                    center = entity[13:15]
                    radius = entity[15]
                    startParam = entity[16] * (2*math.pi)
                    endParam = entity[17] * (2*math.pi)
                    arc = Arc(entityId = id, 
                              isConstruction = isConstructible, 
                              xCenter = center[0], 
                              yCenter = center[1], 
                              xDir = 1, 
                              yDir = 0,
                              radius = radius, 
                              startParam = startParam,
                              endParam = endParam, 
                              clockwise = False
                             );
                    sketch.entities[id] = arc
                case 3:
                    # Create Point
                    id = str(idx + 1)
                    isConstructible = bool(entity[0])
                    x = entity[18]
                    y = entity[19]
                    point = Point(entityId = id, 
                                  isConstruction = isConstructible,
                                  x = x,
                                  y = y
                                 );
                    sketch.entities[id] = point
        # Add constraints
        for idx in range(len(edge_attr)):
            constraint = edge_attr[idx]
            identifier = "c_" + str(idx)
            constraintType = ConstraintType.Coincident # Initial Value
            param_ids = None
            params = []
            # Convert one hot encoding to constraint label
            match torch.argmax(constraint[8:17]):
                case 0:
                    # Coincident
                    constraintType = ConstraintType.Coincident
                case 1:
                    # Horizontal
                    constraintType = ConstraintType.Horizontal
                case 2:
                    # Vertical
                    constraintType = ConstraintType.Vertical
                case 3:
                    # Parallel
                    constraintType = ConstraintType.Parallel
                case 4:
                    # Perpendicular
                    constraintType = ConstraintType.Perpendicular
                case 5:
                    # Tangent
                    constraintType = ConstraintType.Tangent
                case 6:
                    # Midpoint
                    constraintType = ConstraintType.Midpoint
                case 7:
                    # Equal
                    constraintType = ConstraintType.Equal
            # Adjust reference parameter ids if necessary
            if constraintType == ConstraintType.Midpoint:
                param_ids = ['local0', 'local1']
            else:
                param_ids = ['localFirst', 'localSecond']
            edge = edge_index.T[idx]
            if torch.equal(edge[0], edge[1]):
                # Constraint only applies to single entity
                node_ref = str(edge[0])
                match torch.argmax(constraint[0:4]):
                    case 0:
                        node_ref = node_ref + ".start"
                    case 1:
                        node_ref = node_ref + ".center"
                    case 2:
                        node_ref = node_ref + ".end"
                param1 = LocalReferenceParameter(param_ids[0], node_ref)
                params.append(param1)
            else:
                # Constraint applies to 2 primitives
                node_a_ref = str(edge[0])
                match torch.argmax(constraint[0:4]):
                    case 0:
                        node_ref = node_a_ref + ".start"
                    case 1:
                        node_ref = node_a_ref + ".center"
                    case 2:
                        node_ref = node_a_ref + ".end"
                node_b_ref = str(edge[1])
                match torch.argmax(constraint[4:8]):
                    case 0:
                        node_ref = node_b_ref + ".start"
                    case 1:
                        node_ref = node_b_ref + ".center"
                    case 2:
                        node_ref = node_b_ref + ".end"
                param1 = LocalReferenceParameter(param_ids[0], node_a_ref)
                params.append(param1)
                param2 = LocalReferenceParameter(param_ids[1], node_b_ref)
                params.append(param2)
            sketch.constraints[identifier] = Constraint(identifier, constraintType, params)
        return sketch

    @staticmethod
    def sort_graph(node_matrix, adjacency_list):
        # Helper variables
        num_nodes = len(node_matrix)
        node_out_degrees = pygutils.degree(adjacency_list.long()[0], num_nodes)
        # Generate permutation indices to sort graph
        indices = range(num_nodes)
        node_permutation_indices = sorted(  indices, key = lambda idx: (  node_matrix[idx].tolist() + [node_out_degrees[idx]]  )  )
        # return sorted node feature matrix and adjacency list
        sorted_node_matrix = node_matrix[node_permutation_indices]
        node_oldidx_to_newidx = [node_permutation_indices.index(value) for value in range(len(node_permutation_indices))]
        updated_adjacency_list = torch.Tensor(node_oldidx_to_newidx)[adjacency_list.long()]
        return sorted_node_matrix, updated_adjacency_list
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        if self.transform != None:
            return self.transform(data)
        return data


# %%
dataset = SketchDataset(root="data/")

# %%
dataset.len()


