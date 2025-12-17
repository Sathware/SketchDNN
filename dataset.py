import os
import torch
import numpy
from torch.utils.data import TensorDataset
from tqdm import tqdm
import requests
import math
import itertools
import config
import utils
# You have to import sketchgraphs this way otherwise you get type errors
os.chdir('SketchGraphs/')
import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data._entity import Point, Line, Circle, Arc, EntityType
from sketchgraphs.data.sketch import Sketch
from sketchgraphs.data._constraint import *
os.chdir('../')

class SketchDataset(TensorDataset):
    """
    Args:
        root : str           - specifies the root directory to store raw and processed data files
        include_edges : bool - if enabled then the dataset will also process and use edge data in which __getitem__ will return tensor data in the order: (nodes, node_params_mask, edges)
        
    """
    def __init__(self, root : str, include_edges = False):
        self.include_edges = include_edges
        self.raw_dir = os.path.join(root, "raw/")
        self.processed_dir = os.path.join(root, "processed/")
        
        # Download raw data if not already present
        raw_file_paths = [os.path.join(self.raw_dir, file_name) for file_name in self.raw_file_names]
        if not all([os.path.exists(path) for path in raw_file_paths]):
            self.download()

        # Process raw sketch data into a tensor format if not already done so
        processed_file_paths = [os.path.join(self.processed_dir, file_name) for file_name in self.processed_file_names]
        if not all([os.path.exists(path) for path in processed_file_paths]):
            self.process()
        
        self.tensors = [torch.load(path) for path in processed_file_paths]
        
    @property
    def raw_file_names(self):
        return ['sg_t16_train.npy', 'sg_t16_validation.npy', 'sg_t16_test.npy']
    
    @property 
    def processed_file_names(self):
        return ["nodes.pt","node_params_mask.pt","edges.pt"] if self.include_edges else ["nodes.pt","node_params_mask.pt"]
    
    @property
    def num_node_features(self):
        return config.NODE_FEATURE_DIMENSION
    
    @property
    def num_edge_features(self):
        return config.EDGE_FEATURE_DIMENSION if self.include_edges else 0
    
    @staticmethod
    def render_graph(nodes, edges = None, ax = None):
        return datalib.render_sketch(SketchDataset.graph_to_sketch(nodes, edges), ax)
    
    def __len__(self):
        return self.tensors[0].size(0)
    
    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)
    
    def download(self):
        os.makedirs(name = self.raw_dir, exist_ok = True)

        for file_name in self.raw_file_names:
            print(f"Downloading File: {file_name}")
            file_path = os.path.join(self.raw_dir, file_name)
            
            response = requests.get(url = f"https://sketchgraphs.cs.princeton.edu/sequence/{file_name}", stream = True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name, ncols=80)
            with open(file_path, mode = "wb") as file:
                # Write to file in 1 kilo Byte chunks
                for chunk in response.iter_content(chunk_size = 2 ** 10):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))


        print(f"All downloads finished!\nLPIPS SketchGraphs dataset in {self.raw_dir}\n")

    def process(self):
        os.makedirs(name = self.processed_dir, exist_ok = True)
        print(f"Processing raw sketches into tensor format")
        # Pre allocate memory
        save_nodes = torch.zeros(2127996, config.MAX_NUM_PRIMITIVES, config.NODE_FEATURE_DIMENSION)
        save_edges = torch.zeros(2127996, config.MAX_NUM_PRIMITIVES, config.MAX_NUM_PRIMITIVES, config.EDGE_FEATURE_DIMENSION) if self.include_edges else None
        
        # Change dir to SketchGraphs so module not found error doesn't popup
        os.chdir('SketchGraphs/')
        # Load SketchGraphs sequence data dictionary
        partitions = [flat_array.load_dictionary_flat(os.path.join("../", self.raw_dir, file_name))["sequences"] for file_name in self.raw_file_names]
        sequences = itertools.chain(*partitions)
        total_num_sequences = sum([len(part) for part in partitions])
        
        idx = 0
        for _ in tqdm(range(total_num_sequences), unit = "Sketches"):
            seq = next(sequences)
            sketch = datalib.sketch_from_sequence(seq)
            # Passover invalid sketches
            if sketch is None:
                continue
            # Filter out sketches with less than 7 or more than max_num primitives
            # or less constraints than primitives or more than 208 constraints
            if len(sketch.entities) < config.MIN_NUM_PRIMITIVES or len(sketch.entities) > config.MAX_NUM_PRIMITIVES or len(sketch.constraints) < len(sketch.entities) or len(sketch.constraints) > 208:
                continue
            # Construct Data Object containing graph
            nodes, edges = SketchDataset.sketch_to_graph(sketch)

            # Sanity checks
            num_none_types = nodes[:,6].sum()
            if num_none_types > (config.MAX_NUM_PRIMITIVES - config.MIN_NUM_PRIMITIVES):
                continue
            if (not nodes.isfinite().all()):
                continue
                
            # Append graph
            save_nodes[idx] = nodes

            if self.include_edges:
                save_edges[idx] = edges
            
            idx += 1
        
        save_nodes = save_nodes[:idx] # Discard unused space

        if self.include_edges: save_edges = save_edges[:idx] # Discard unused space

        # Normalize Sketches
        save_nodes = utils.BoundingBoxShiftScale(save_nodes)

        # Remove duplicate sketches
        unique_indices = utils.GetUniqueIndices(save_nodes, 2 ** 8)
        save_nodes = save_nodes[unique_indices]
        if self.include_edges: save_edges = save_edges[unique_indices]

        # Save file flag and Change dir back
        os.chdir('../')
        torch.save(save_nodes, os.path.join(self.processed_dir, self.processed_file_names[0]))
        torch.save(self.batched_params_mask(save_nodes), os.path.join(self.processed_dir, self.processed_file_names[1]))
        if self.include_edges: torch.save(save_edges, os.path.join(self.processed_dir, self.processed_file_names[2]))
        print(f"Data processing finished!\nProcessed datafiles are in {self.processed_dir}")

    '''Convert an onshape sketch to a graph representation where the output is two tensors: nodes (num_primitives x primitive_features) and edges (num_primitives x num_primitives x constraint_features)'''
    @staticmethod
    def sketch_to_graph(sketch : Sketch, include_edges : bool = False):
        '''Extract Node Information into feature matrix'''
        node_matrix = torch.zeros(size=(config.MAX_NUM_PRIMITIVES, config.NODE_FEATURE_DIMENSION))
        node_ref_to_idx = {} # Constraints in onshape sketches point to nodes by a unique string id "reference", needed to make the correct adjacency matrix

        idx = 0
        for key, primitive in sketch.entities.items():
            if (idx >= config.MAX_NUM_PRIMITIVES):
                break
            
            match primitive.type:
                case EntityType.Line:
                    if numpy.allclose(primitive.start_point, primitive.end_point, rtol = 0): continue
                    node_matrix[idx, 2] = 1
                    node_matrix[idx, 7:9] = torch.from_numpy(primitive.start_point)
                    node_matrix[idx, 9:11] = torch.from_numpy(primitive.end_point)
                case EntityType.Circle:
                    if numpy.isclose(primitive.radius, 0, rtol = 0): continue
                    node_matrix[idx, 3] = 1
                    node_matrix[idx, 11:13] = torch.from_numpy(primitive.center_point)
                    node_matrix[idx, 13] = primitive.radius
                case EntityType.Arc:
                    if numpy.isclose(primitive.radius, 0, rtol = 0): continue
                    if numpy.allclose(primitive.start_point, primitive.end_point, rtol = 0): continue
                    node_matrix[idx, 4] = 1
                    node_matrix[idx, 14:16] = torch.from_numpy(primitive.center_point)
                    node_matrix[idx, 16] = primitive.radius
                    angle_start_offset = -primitive.endParam if primitive.clockwise else primitive.startParam
                    angle_end_offset = -primitive.startParam if primitive.clockwise else primitive.endParam
                    angle = math.atan2(primitive.yDir, primitive.xDir)
                    node_matrix[idx, 17] = ((angle + angle_start_offset) % (2*math.pi)) / (2*math.pi)
                    node_matrix[idx, 18] = ((angle + angle_end_offset) % (2*math.pi)) / (2*math.pi)
                case EntityType.Point:
                    node_matrix[idx, 5] = 1
                    node_matrix[idx, 19] = primitive.x
                    node_matrix[idx, 20] = primitive.y
                case _:
                    continue
                
            node_ref_to_idx[key] = idx
            node_matrix[idx, int(primitive.isConstruction)] = 1
            idx += 1
        # Change unused rows to none type primitives
        node_matrix[idx:] = torch.tensor([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        node_matrix = utils.ToIscosceles(node_matrix) # Change arc parameterization to terminal points and curvature instead of center, radius, and terminal angles
        if not include_edges: return node_matrix, None

        '''Extract Edge Information'''
        edge_index = torch.zeros(size=(config.MAX_NUM_CONSTRAINTS, 2)) # Adjacency matrix
        edge_attr = torch.zeros(size=(config.MAX_NUM_CONSTRAINTS, config.EDGE_FEATURE_DIMENSION)) # Edge feature matrix
        # Build adjacency list and edge feature matrix
        idx = 0
        edge_exists = {}
        for primitive in sketch.constraints.values():
            # Enforce maximum 208 constraints limit
            if (idx >= config.MAX_NUM_CONSTRAINTS or idx >= config.MAX_NUM_PRIMITIVES ** 2):
                break
            edge_feature = torch.zeros(17)
            # Set one hot encoding for constraint type
            match primitive.type:
                case ConstraintType.Coincident:
                    edge_feature[8] = 1
                case ConstraintType.Horizontal:
                    edge_feature[9] = 1
                case ConstraintType.Vertical:
                    edge_feature[10] = 1
                case ConstraintType.Parallel:
                    edge_feature[11] = 1
                case ConstraintType.Perpendicular:
                    edge_feature[12] = 1
                case ConstraintType.Tangent:
                    edge_feature[13] = 1
                case ConstraintType.Midpoint:
                    edge_feature[14] = 1
                case ConstraintType.Equal:
                    edge_feature[15] = 1
                case _:
                    continue
            connection = primitive.get_references()
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
                edge_feature[7] = 1 # constraint is not applied for another primitive
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
        edge_index = edge_index[:idx].T.contiguous()
        edge_attr = edge_attr[:idx]
        edge_tensor = torch.sparse_coo_tensor(edge_index, edge_attr, (config.MAX_NUM_PRIMITIVES, config.MAX_NUM_PRIMITIVES, config.EDGE_FEATURE_DIMENSION)).to_dense()
        edge_tensor[torch.abs(edge_tensor).sum(dim = 2) == 0] = torch.Tensor([0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1]) # change empty constraints to none type
        return node_matrix, edge_tensor

    @staticmethod
    def graph_to_sketch(nodes, edges = None):
        nodes = utils.ToNaive(nodes) # Convert to center, radius, and terminal angles parameterization for arcs
        sketch = Sketch()
        '''Add primitives'''
        for idx in range(len(nodes)):
            primitive = nodes[idx]
            isConstructible = bool(primitive[0].item() < 0.5)
            match torch.argmax(primitive[config.NODE_TYPE_SLICE]):
                case 0:
                    # Create Line
                    id = str(idx + 1)
                    start = primitive[7:9]
                    end = primitive[9:11]
                    pnt = start
                    dir = (end - start)
                    startParam = 0
                    endParam = torch.linalg.vector_norm(dir)
                    dir = dir / endParam
                    line = Line(entityId = id,
                                isConstruction = isConstructible, 
                                pntX = pnt[0].item(), 
                                pntY = pnt[1].item(), 
                                dirX = dir[0].item(), 
                                dirY = dir[1].item(), 
                                startParam = startParam, 
                                endParam = endParam.item()
                               );
                    sketch.entities[id] = line
                case 1:
                    # Create Circle
                    id = str(idx + 1)
                    center = primitive[11:13]
                    radius = primitive[13]
                    circle = Circle(entityId = id, 
                                  isConstruction = isConstructible, 
                                  xCenter = center[0].item(), 
                                  yCenter = center[1].item(), 
                                  xDir = 1, 
                                  yDir = 0, 
                                  radius = radius.item(), 
                                  clockwise = False
                                 );
                    sketch.entities[id] = circle
                case 2: 
                    # Create Arc
                    id = str(idx + 1)
                    center = primitive[14:16]
                    radius = primitive[16]
                    startParam = primitive[17] * (2*math.pi)
                    endParam = primitive[18] * (2*math.pi)
                    arc = Arc(entityId = id, 
                              isConstruction = isConstructible, 
                              xCenter = center[0].item(), 
                              yCenter = center[1].item(), 
                              xDir = 1, 
                              yDir = 0,
                              radius = radius.item(), 
                              startParam = startParam.item(),
                              endParam = endParam.item(), 
                              clockwise = False
                             );
                    sketch.entities[id] = arc
                case 3:
                    # Create Point
                    id = str(idx + 1)
                    x = primitive[19]
                    y = primitive[20]
                    point = Point(entityId = id, 
                                  isConstruction = isConstructible,
                                  x = x.item(),
                                  y = y.item()
                                 );
                    sketch.entities[id] = point
                case _:
                    continue

        if edges is None: return sketch

        '''Add constraints'''
        idx = 0
        for i in range(edges.size(0)):
            for j in range(edges.size(1)):
                constraint = edges[i][j]
                identifier = "c_" + str(idx)
                constraintType = ConstraintType.Coincident # Initial Value
                param_ids = None
                params = []
                # Convert one hot encoding to constraint label
                match torch.argmax(constraint[config.EDGE_TYPE_SLICE]):
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
                    case _:
                        # None
                        continue
                # Adjust reference parameter ids if necessary
                if constraintType == ConstraintType.Midpoint:
                    param_ids = ['local0', 'local1']
                else:
                    param_ids = ['localFirst', 'localSecond']
                edge = torch.Tensor([i, j])
                if torch.equal(edge[0], edge[1]):
                    # Constraint only applies to single entity
                    node_ref = str(int(edge[0].item()))
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
                    node_a_ref = str(int(edge[0].item()))
                    match torch.argmax(constraint[0:4]):
                        case 0:
                            node_ref = node_a_ref + ".start"
                        case 1:
                            node_ref = node_a_ref + ".center"
                        case 2:
                            node_ref = node_a_ref + ".end"
                    node_b_ref = str(int(edge[1].item()))
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
                idx = idx + 1
        return sketch
    
    @staticmethod
    def params_mask(nodes):
        mask = torch.zeros(size = (config.MAX_NUM_PRIMITIVES, config.NODE_FEATURE_DIMENSION - config.NUM_PRIMITIVE_TYPES - 2)) # subtracting by two for isconstructible binary one-hot vector
        i = 0
        for node in nodes:
            match torch.argmax(node[config.NODE_TYPE_SLICE]):
                case 0:
                    # Line
                    mask[i][0:4] = 1
                case 1:
                    # Circle
                    mask[i][4:7] = 1
                case 2:
                    # Arc
                    mask[i][7:12] = 1
                case 3:
                    # Point
                    mask[i][12:] = 1
            i = i + 1
        return mask
    
    @staticmethod
    def batched_params_mask(nodes):
        # batch_size x num_nodes x num_parameters
        batch_size = nodes.size(0)
        mask = torch.zeros(size = (batch_size, config.MAX_NUM_PRIMITIVES, config.NODE_FEATURE_DIMENSION - config.NUM_PRIMITIVE_TYPES - 2)) # subtracting by two for isconstructible binary one-hot vector
        for b in range(batch_size):
            for i, node in enumerate(nodes[b]):
                match torch.argmax(node[config.NODE_TYPE_SLICE]):
                    case 0:
                        # Line
                        mask[b,i,0:4] = 1
                    case 1:
                        # Circle
                        mask[b,i,4:7] = 1
                    case 2:
                        # Arc
                        mask[b,i,7:12] = 1
                    case 3:
                        # Point
                        mask[b,i,12:] = 1
        return mask
    
    @staticmethod
    def get_start_point(primitive):
        match primitive.type:
            case EntityType.Line:
                return torch.from_numpy(primitive.start_point)
            case EntityType.Circle:
                center = torch.from_numpy(primitive.center_point)
                center[0] += primitive.radius # add radius to x coord
                return center
            case EntityType.Arc:
                return torch.from_numpy(primitive.start_point)
            case EntityType.Point:
                return torch.Tensor([primitive.x, primitive.y])
            case _:
                return None
            
    @staticmethod
    def get_end_point(primitive):
        match primitive.type:
            case EntityType.Line:
                return torch.from_numpy(primitive.end_point)
            case EntityType.Circle:
                center = torch.from_numpy(primitive.center_point)
                center[0] += primitive.radius # add radius to x coord
                return center
            case EntityType.Arc:
                return torch.from_numpy(primitive.end_point)
            case EntityType.Point:
                return torch.Tensor([primitive.x, primitive.y])
            case _:
                return None
            
    @staticmethod
    def get_mid_point(primitive):
        match primitive.type:
            case EntityType.Line:
                return (torch.from_numpy(primitive.end_point) + torch.from_numpy(primitive.end_point)) / 2
            case EntityType.Circle:
                center = torch.from_numpy(primitive.center_point)
                center[0] -= primitive.radius # add radius to x coord
                return center
            case EntityType.Arc:
                return torch.from_numpy(primitive.mid_point)
            case EntityType.Point:
                return torch.Tensor([primitive.x, primitive.y])
            case _:
                return None
            
    @staticmethod
    def superimpose_constraints(sketch, ax):
        for constraint in sketch.constraints.values():
            point1 = None
            point2 = None
            color = None
            #primitive1_idx = int(float(constraint.parameters[0].referenceMain)) + 1
            #print(sketch.entities[str(primitive1_idx)])
            prim1 = constraint.parameters[0].value.split('.')
            prim1_idx = prim1[0]
            prim1_sub = prim1[1] if len(prim1) > 1 else ''
            prim1_idx = str(int(prim1_idx) + 1)
            match prim1_sub:
                case "start":
                    point1 = SketchDataset.get_start_point(sketch.entities[prim1_idx])
                case "center":
                    point1 = SketchDataset.get_mid_point(sketch.entities[prim1_idx])
                case "end":
                    point1 = SketchDataset.get_end_point(sketch.entities[prim1_idx])
                case _:
                    point1 = SketchDataset.get_start_point(sketch.entities[prim1_idx])
            # print(constraint.parameters[0].value)
            if len(constraint.parameters) > 1:
                #primitive2_idx = int(float(constraint.parameters[1].referenceMain)) + 1
                #print(sketch.entities[str(primitive2_idx)])
                prim2 = constraint.parameters[1].value.split('.')
                prim2_idx = prim2[0]
                prim2_sub = prim2[1] if len(prim1) > 1 else ''
                prim2_idx = str(int(prim2_idx) + 1)
                match prim2_sub:
                    case "start":
                        point2 = SketchDataset.get_start_point(sketch.entities[prim2_idx])
                    case "center":
                        point2 = SketchDataset.get_mid_point(sketch.entities[prim2_idx])
                    case "end":
                        point2 = SketchDataset.get_end_point(sketch.entities[prim2_idx])
                    case _:
                        point2 = SketchDataset.get_start_point(sketch.entities[prim2_idx])
                # print(constraint.parameters[1].value)

            match constraint.type:
                case ConstraintType.Coincident:
                    color = 'green'
                case ConstraintType.Horizontal:
                    color = 'red'
                case ConstraintType.Vertical:
                    color = 'brown'
                case ConstraintType.Parallel:
                    color = 'pink'
                case ConstraintType.Perpendicular:
                    color = 'purple'
                case ConstraintType.Tangent:
                    color = 'orange'
                case ConstraintType.Midpoint:
                    color = 'blue'
                case ConstraintType.Equal:
                    color = 'slategrey'
                case _:
                    continue
            ax.plot((point1[0], point2[0]), (point1[1], point2[1]), color, linestyle='--', linewidth=1, marker=None)
        return