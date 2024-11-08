import numpy as np
from scipy import linalg

class MarkovArrivalProcess:
    def __init__(self, D0, D1):
        """
        Initialize MAP with transition matrices D0 and D1
        
        Parameters:
        D0 (numpy.ndarray): Non-arrival transition matrix
        D1 (numpy.ndarray): Arrival transition matrix
        """
        self.D0 = np.array(D0)
        self.D1 = np.array(D1)
        self.n_states = self.D0.shape[0]
        
        # Validate input matrices
        self._validate_matrices()
        
        # Calculate fundamental matrices
        self.Q = self.D0 + self.D1  # Infinitesimal generator
        self.pi = self._compute_stationary_distribution()
        self.lambda_ = self._compute_arrival_rate()
    
    def _validate_matrices(self):
        """Validate the MAP matrices properties"""
        if self.D0.shape != self.D1.shape:
            raise ValueError("D0 and D1 must have the same shape")
        
        if not np.allclose(self.D0.shape[0], self.D0.shape[1]):
            raise ValueError("Matrices must be square")
        
        # Check if off-diagonal elements of D0 and all elements of D1 are non-negative
        if not (np.all(np.diagonal(self.D0) <= 0) and 
                np.all(self.D0 - np.diag(np.diagonal(self.D0)) >= 0) and 
                np.all(self.D1 >= 0)):
            raise ValueError("Invalid MAP matrices")
        
        # Check if row sums of Q are zero
        Q = self.D0 + self.D1
        if not np.allclose(Q.sum(axis=1), 0):
            raise ValueError("Row sums of Q must be zero")
    
    def _compute_stationary_distribution(self):
        """Compute the stationary probability vector"""
        Q = self.Q
        
        # Construct the linear system for steady state
        A = Q.T
        A = np.vstack([A, np.ones(self.n_states)])  # Add normalization condition
        b = np.zeros(self.n_states + 1)
        b[-1] = 1  # Set normalization condition
        
        # Solve the system
        pi = linalg.solve(A.T @ A, A.T @ b)
        return pi
    
    def _compute_arrival_rate(self):
        """Compute the fundamental arrival rate"""
        return float(self.pi @ self.D1 @ np.ones(self.n_states))
    
    def generate_arrivals(self, T, seed=None):
        """
        Generate arrival times using the MAP up to time T
        
        Parameters:
        T (float): Time horizon
        seed (int): Random seed for reproducibility
        
        Returns:
        list: Arrival times
        """
        if seed is not None:
            np.random.seed(seed)
        
        arrivals = []
        current_time = 0
        current_state = np.random.choice(self.n_states, p=self.pi)
        
        while current_time < T:
            # Calculate total rate out of current state
            total_rate = -self.D0[current_state, current_state]
            
            # Generate time until next transition
            time_to_next = np.random.exponential(1/total_rate)
            current_time += time_to_next
            
            if current_time > T:
                break
            
            # Calculate transition probabilities
            P = np.zeros((2, self.n_states))
            P[0] = self.D0[current_state] / total_rate  # Non-arrival transitions
            P[1] = self.D1[current_state] / total_rate  # Arrival transitions
            
            # Determine if arrival occurs and next state
            rand = np.random.random()
            cumsum = np.cumsum([P[0].sum(), P[1].sum()])
            is_arrival = rand > cumsum[0]
            
            if is_arrival:
                arrivals.append(current_time)
                probs = P[1] / P[1].sum()
            else:
                probs = P[0] / P[0].sum()
            
            current_state = np.random.choice(self.n_states, p=probs)
        
        return arrivals

def example_usage():
    # Example two-state MAP
    D0 = np.array([[-3, 1],
                   [2, -4]])
    D1 = np.array([[1.5, 0.5], 
                   [1.0, 1.0]]) # note: -3 = 1 + 1.5 + 0.5 (all the elements in the same row of two matrices (not each matrix) sum to 0)
    
    map_process = MarkovArrivalProcess(D0, D1)
    print(f"Stationary distribution: {map_process.pi}")
    print(f"Arrival rate: {map_process.lambda_}")
    
    # Generate arrivals for time period T=10
    arrivals = map_process.generate_arrivals(T=10, seed=42)
    print(f"Arrival times: {arrivals}")

if __name__ == "__main__":
    example_usage()