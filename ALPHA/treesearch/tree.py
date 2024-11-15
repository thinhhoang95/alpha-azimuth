from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from .eval import eval_plan
from ALPHA.obstacle.obstacle import Obstacle
import ALPHA.obstacle.reception as obrece
@dataclass
class Edge:
    """Represents an action taken between nodes"""
    action: Any
    policy_category: str # 'obstacle', 'flow', 'hole', 'vector'...
    prior_probability: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0
    
    @property
    def mean_value(self) -> float:
        return self.total_value / max(1, self.visit_count)

@dataclass
class Node:
    """Represents a state in the search tree"""
    state: Any # represents a full plan
    parent: Optional['Node'] = None
    parent_edge: Optional[Edge] = None
    children: Dict[Any, 'Node'] = field(default_factory=dict)
    edges: Dict[Any, Edge] = field(default_factory=dict)
    
    # Statistics for the node
    visit_count: int = 0
    value_sum: float = 0.0
    best_value: float = float('-inf')
    # is_terminal: bool = False
    inherent_cost: float = 0.0 # the cost associated with the state of the node itself
                               # or, direct evaluation of the flight plan of this node
    # Inherent values: associated with the state of the node itself (not the children)
    is_inherently_evaluated: bool = False
    time_cost: float = 0.0
    obstacle_hits: int = 0
    
    @property
    def mean_value(self) -> float:
        return self.value_sum / max(1, self.visit_count)
    
    def add_child(self, action: Any, state: Any, policy_category: str) -> 'Node':
        """Creates a new child node connected by the given action"""
        edge = Edge(action=action, policy_category=policy_category)
        child = Node(state=state, parent=self, parent_edge=edge)
        self.children[action] = child
        self.edges[action] = edge
        return child
    
    def is_expanded(self) -> bool:
        """Returns whether this node has any children"""
        return len(self.children) > 0
    
    def evaluate_inherent(self, obstacles: Obstacle, airspace: Any, wind_field: Any = None) -> float:
        """Evaluates the node using the evaluation function"""
        time_cost, obstacle_hits = eval_plan(self.state, obstacles, airspace, wind_field)
        self.time_cost = time_cost
        self.obstacle_hits = obstacle_hits
        self.inherent_cost = time_cost + obstacle_hits * 1000
        self.is_inherently_evaluated = True

    def get_actions(self, obstacles: Obstacle):
        # For each segment in the flight path
        for i in range(len(self.state.waypoints) - 1):
            start_point = self.state.waypoints[i]
            end_point = self.state.waypoints[i + 1]
            # ----------------------------------------------------------------------------------
            # OBSTACLE AVOIDANCE POLICY
            # ----------------------------------------------------------------------------------
            # Get the attention weight and the proposed action
            attention_weight_obstacle, proposed_action_obstacle = obrece.propose_action(start_point, end_point, obstacles)
            # TODO: resample the action based on the attention weight
            if proposed_action_obstacle is not None: # if there is an obstacle in the way
                return i, proposed_action_obstacle # index of segment and proposed action
            # otherwise, there is no obstacle in the way, the obstacle avoidance policy returns None
        return None, None # No obstacle in the way

class Tree:
    """Search tree implementation"""
    def __init__(self, root_state: Any):
        self.root = Node(state=root_state)

    def backup_values(self):
        """Backs up best values from leaf nodes to root"""
        def backup_node(node: Node) -> float:
            # If leaf node, best value is its inherent cost
            if not node.is_expanded():
                node.best_value = node.inherent_cost
                return node.best_value
            
            # For internal nodes, recursively get best values of children
            children_values = [backup_node(child) for child in node.children.values()]
            # Set best value as minimum of children's best values
            node.best_value = min(children_values)
            return node.best_value
        
        # Start backup from root
        backup_node(self.root)
    
    def evaluate_all_nodes(self, obstacles: Obstacle, airspace: Any, wind_field: Any = None):
        """Evaluates the inherent cost for all nodes in the tree"""
        def evaluate_node(node: Node):
            # Evaluate inherent cost (best_value will automatically be updated with backup_values())
            node.evaluate_inherent(obstacles, airspace, wind_field)
            for child in node.children.values():
                # if not child.is_inherently_evaluated:
                evaluate_node(child)
                
        evaluate_node(self.root)
        self.backup_values() # Update best values for all nodes

    
    def get_node(self, state: Any) -> Optional[Node]:
        """Find a node in the tree with the given state (basic implementation)"""
        def search(node: Node) -> Optional[Node]:
            if node.state == state:
                return node
            for child in node.children.values():
                result = search(child)
                if result is not None:
                    return result
            return None
        
        return search(self.root)
    
    def size(self) -> int:
        """Returns the total number of nodes in the tree"""
        def count_nodes(node: Node) -> int:
            return 1 + sum(count_nodes(child) for child in node.children.values())
        return count_nodes(self.root)
    
    def find_best_leaf(self) -> Optional[Node]:
        """Returns the leaf node that determines the root's best value"""
        def find_best_leaf_recursive(node: Node) -> Optional[Node]:
            # If leaf node and its value matches root's best value, we found it
            if not node.is_expanded() and abs(node.best_value - self.root.best_value) < 1e-10:
                return node
            
            # For internal nodes, recursively search children
            for child in node.children.values():
                # Only search child if its best value matches root's best value
                if abs(child.best_value - self.root.best_value) < 1e-10:
                    result = find_best_leaf_recursive(child)
                    if result is not None:
                        return result
            return None
        
        return find_best_leaf_recursive(self.root)
    
    def plot(self, ax=None, show_values=True, value_type='inherent'):
        """Plots the tree structure visually using networkx and matplotlib
        
        Args:
            ax: Optional matplotlib axis to plot on
            show_values: Whether to show node values in the plot
            value_type: Type of value to show in the plot ('inherent', 'best', 'inherent+obstaclehits', 'num_wp')
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Create directed graph
        G = nx.DiGraph()
        
        def add_nodes_edges(node: Node, parent_id=None):
            node_id = str(node.state)
            if show_values:
                if value_type == 'inherent':
                    label = f'{node.inherent_cost:.2f}'
                elif value_type == 'best':
                    label = f'{node.best_value:.2f}'
                elif value_type == 'inherent+obstaclehits':
                    label = f'{node.inherent_cost:.2f} + {node.obstacle_hits} H'
                elif value_type == 'num_wp':
                    label = f'{len(node.state)} WPs'
            else:
                label = 's'
            G.add_node(node_id, label=label)
            
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
                
            for child in node.children.values():
                add_nodes_edges(child, node_id)
        
        add_nodes_edges(self.root)
        
        # Create plot
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
            
        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax, 
                with_labels=True,
                node_color='lightblue',
                node_size=2000,
                arrowsize=20,
                labels=nx.get_node_attributes(G, 'label'),
                font_size=8)
        
        plt.tight_layout()
        plt.title(f'{value_type} values')
        plt.show()
        return ax