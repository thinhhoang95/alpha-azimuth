from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from .eval import eval_plan
from ALPHA.obstacle.obstacle import Obstacle

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
    is_terminal: bool = False
    inherent_cost: float = 0.0 # the cost associated with the state of the node itself
                               # or, direct evaluation of the flight plan of this node
    
    @property
    def mean_value(self) -> float:
        return self.value_sum / max(1, self.visit_count)
    
    def add_child(self, action: Any, state: Any) -> 'Node':
        """Creates a new child node connected by the given action"""
        edge = Edge(action=action)
        child = Node(state=state, parent=self, parent_edge=edge)
        self.children[action] = child
        self.edges[action] = edge
        return child
    
    def is_expanded(self) -> bool:
        """Returns whether this node has any children"""
        return len(self.children) > 0
    
    def evaluate_inherent(self, obstacles: Obstacle, airspace: Any, wind_field: Any = None) -> float:
        """Evaluates the node using the evaluation function"""
        self.inherent_cost = eval_plan(self.state, obstacles, airspace, wind_field)

class Tree:
    """Search tree implementation"""
    def __init__(self, root_state: Any):
        self.root = Node(state=root_state)
    
    def evaluate_inherent_cost_for_all_nodes(self, obstacles: Obstacle, airspace: Any, wind_field: Any = None):
        """Evaluates the inherent cost for all nodes in the tree"""
        def evaluate_node(node: Node):
            node.evaluate_inherent(obstacles, airspace, wind_field)
            for child in node.children.values():
                evaluate_node(child)
                
        evaluate_node(self.root)

    
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
