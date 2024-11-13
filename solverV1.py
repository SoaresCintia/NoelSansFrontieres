import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

class SVEMatchingSolver:
    def __init__(self, 
                 num_students: int,
                 num_hosts: int,
                 num_time_periods: int,
                 student_availability: np.ndarray,
                 host_availability: np.ndarray,
                 compatibility_matrix: np.ndarray):
        """
        Initialize the matching solver using Gurobi.
        
        Args:
            num_students: Number of students
            num_hosts: Number of hosts
            num_time_periods: Number of time periods
            student_availability: Binary matrix (S x T) of student availability
            host_availability: Binary matrix (H x T) of host availability
            compatibility_matrix: Binary matrix (S x H) of student-host compatibility
        """
        self.S = num_students
        self.H = num_hosts
        self.T = num_time_periods
        
        self.student_avail = student_availability
        self.host_avail = host_availability
        self.compatibility = compatibility_matrix
        
        # Validate input dimensions
        assert student_availability.shape == (self.S, self.T), "Invalid student availability dimensions"
        assert host_availability.shape == (self.H, self.T), "Invalid host availability dimensions"
        assert compatibility_matrix.shape == (self.S, self.H), "Invalid compatibility matrix dimensions"
        
        # Initialize Gurobi model
        self.model = gp.Model("SVE_Matching")
        
        # Initialize variables and constraints
        self._setup_problem()
    
    def create_preference_scores(self,
        language_match: np.ndarray,
        activity_match: np.ndarray,
        weights: Dict[str, float] = {"language": 0.7, "activity": 0.3}
    ) -> np.ndarray:
        """
        Create preference scores matrix
        
        Args:
            language_match: Binary matrix (S x H) indicating language compatibility
            activity_match: Binary matrix (S x H) indicating activity preference match
            weights: Dictionary of weights for different preference types
            
        Returns:
            np.ndarray: Preference scores matrix (S x H x T)
        """
        assert language_match.shape == (self.S, self.H), "Invalid language match dimensions"
        assert activity_match.shape == (self.S, self.H), "Invalid activity match dimensions"
        
        scores = (weights["language"] * language_match + 
                 weights["activity"] * activity_match)
        
        # Expand to include time dimension
        return np.repeat(scores[:, :, np.newaxis], self.T, axis=2)

    def _setup_problem(self):
        """Setup the optimization problem variables and constraints."""
        # Create 3D dictionary for x variables (student-host-time assignments)
        self.x = {}
        for s in range(self.S):
            for h in range(self.H):
                for t in range(self.T):
                    if (self.student_avail[s,t] and 
                        self.host_avail[h,t] and 
                        self.compatibility[s,h]):
                        self.x[s,h,t] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f'x_{s}_{h}_{t}'
                        )
        
        # Create 2D dictionary for e variables (host-time events)
        self.e = {}
        for h in range(self.H):
            for t in range(self.T):
                if self.host_avail[h,t]:
                    self.e[h,t] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f'e_{h}_{t}'
                    )
        
        self.model.update()
        
        # Add constraints only if we have variables to constrain
        if not self.x:
            raise ValueError("No feasible assignments possible with given constraints")
            
        # 1. Each student assigned at most once
        for s in range(self.S):
            student_assignments = [self.x[s,h,t] for h in range(self.H) 
                                 for t in range(self.T) if (s,h,t) in self.x]
            if student_assignments:  # Only add constraint if student has possible assignments
                self.model.addConstr(
                    gp.quicksum(student_assignments) <= 1,
                    name=f'student_{s}_once'
                )
        
        # 2. Each host has at most one event
        for h in range(self.H):
            host_events = [self.e[h,t] for t in range(self.T) if (h,t) in self.e]
            if host_events:  # Only add constraint if host has possible events
                self.model.addConstr(
                    gp.quicksum(host_events) <= 1,
                    name=f'host_{h}_once'
                )
        
        # 3. Event size constraints (2-5 students)
        for h in range(self.H):
            for t in range(self.T):
                if (h,t) in self.e:
                    students_at_event = [self.x[s,h,t] for s in range(self.S) 
                                       if (s,h,t) in self.x]
                    if students_at_event:  # Only add constraints if event has possible students
                        self.model.addConstr(
                            2 * self.e[h,t] <= gp.quicksum(students_at_event),
                            name=f'min_students_{h}_{t}'
                        )
                        self.model.addConstr(
                            gp.quicksum(students_at_event) <= 5 * self.e[h,t],
                            name=f'max_students_{h}_{t}'
                        )
        
        # Add valid inequalities
        if self.e:
            self.model.addConstr(
                gp.quicksum(self.e[h,t] for h, t in self.e) <= np.ceil(self.S/2),
                name='valid_ineq_1'
            )
        
        if self.x:
            self.model.addConstr(
                gp.quicksum(self.x[s,h,t] for s,h,t in self.x) <= min(self.S, 5*self.H),
                name='valid_ineq_2'
            )
    
    def solve_phase1(self) -> Tuple[float, dict, dict]:
        """
        Solve phase 1: Maximize total matches
        
        Returns:
            tuple: (objective value, x assignment dict, e event dict)
        """
        if not self.x:
            raise ValueError("No feasible assignments possible")
            
        # Objective: Maximize total matches
        self.model.setObjective(
            gp.quicksum(self.x[s,h,t] for s,h,t in self.x),
            GRB.MAXIMIZE
        )
        
        # Set Gurobi parameters for better performance
        self.model.setParam('MIPGap', 0.01)  # 1% optimality gap
        self.model.setParam('TimeLimit', 300)  # 5 minute time limit
        
        # Solve the model
        self.model.optimize()
        
        if self.model.status != GRB.OPTIMAL and self.model.status != GRB.TIME_LIMIT:
            raise RuntimeError(f"Optimization failed with status {self.model.status}")
        
        # Extract solutions
        x_sol = {(s,h,t): var.X for (s,h,t), var in self.x.items()}
        e_sol = {(h,t): var.X for (h,t), var in self.e.items()}
        
        return self.model.ObjVal, x_sol, e_sol
    
    def solve_phase2(self, phase1_matches: float, preferences: np.ndarray) -> Tuple[float, dict, dict]:
        """
        Solve phase 2: Optimize preferences while maintaining maximum matches
        
        Args:
            phase1_matches: Number of matches from phase 1
            preferences: Preference scores matrix (S x H x T)
            
        Returns:
            tuple: (objective value, x assignment dict, e event dict)
        """
        # Validate preference matrix dimensions
        assert preferences.shape == (self.S, self.H, self.T), "Invalid preference matrix dimensions"
        
        # Add constraint to maintain number of matches
        self.model.addConstr(
            gp.quicksum(self.x[s,h,t] for s,h,t in self.x) == phase1_matches,
            name='maintain_matches'
        )
        
        # Objective: Maximize preference score
        self.model.setObjective(
            gp.quicksum(preferences[s,h,t] * self.x[s,h,t] 
                       for s,h,t in self.x if preferences[s,h,t] > 0),
            GRB.MAXIMIZE
        )
        
        # Solve the model
        self.model.optimize()
        
        if self.model.status != GRB.OPTIMAL and self.model.status != GRB.TIME_LIMIT:
            raise RuntimeError(f"Phase 2 optimization failed with status {self.model.status}")
        
        # Extract solutions
        x_sol = {(s,h,t): var.X for (s,h,t), var in self.x.items()}
        e_sol = {(h,t): var.X for (h,t), var in self.e.items()}
        
        return self.model.ObjVal, x_sol, e_sol

def format_solution(x_sol: dict, e_sol: dict, S: int, H: int, T: int) -> pd.DataFrame:
    """Format the solution into a readable DataFrame"""
    matches = []
    for (s,h,t), val in x_sol.items():
        if val > 0.5:  # Account for numerical precision
            matches.append({
                'Student': f'Student_{s}',
                'Host': f'Host_{h}',
                'Time': f'Period_{t}',
                'Group_Size': sum(x_sol.get((s2,h,t), 0) > 0.5 
                                for s2 in range(S))
            })
    return pd.DataFrame(matches)

def main():
    # Example usage with smaller test case
    S, H, T = 15, 5, 4  # Smaller problem for testing
    
    # Generate random test data
    np.random.seed(42)
    student_avail = np.random.binomial(1, 0.7, (S, T))
    host_avail = np.random.binomial(1, 0.5, (H, T))
    compatibility = np.random.binomial(1, 0.8, (S, H))
    
    try:
        # Create and solve the matching problem
        solver = SVEMatchingSolver(S, H, T, student_avail, host_avail, compatibility)
        
        # Phase 1: Maximize matches
        matches, x_sol, e_sol = solver.solve_phase1()
        print(f"Phase 1 - Total matches: {matches}")
        
        # Generate preference scores
        language_match = np.random.binomial(1, 0.3, (S, H))
        activity_match = np.random.binomial(1, 0.4, (S, H))
        preferences = solver.create_preference_scores(language_match, activity_match)
        
        # Phase 2: Optimize preferences
        pref_score, x_final, e_final = solver.solve_phase2(matches, preferences)
        print(f"Phase 2 - Preference score: {pref_score}")
        
        # Format and display solution
        solution_df = format_solution(x_final, e_final, S, H, T)
        print("\nSolution Summary:")
        print(f"Total matches: {len(solution_df)}")
        print(f"Number of hosts used: {solution_df['Host'].nunique()}")
        print(f"Average group size: {solution_df['Group_Size'].mean():.2f}")
        
        return solution_df
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    solution = main()


# 1 step: read student and host data from csv files (pandas script)
# 2 step:
#  hard constraints - time/compatiability (english student, animals), number of students a hjost accepts
#  soft constraints - preferences (language, culture, activity)
# 3 step: Objective is to maximize the number of matches and preference score 
# preference score: langue 50%, acticity 50%, later (transport, campus, size of the group) 
# 
# Python - Gurobi - Cvxpy - small search for other alternatives. "GLPK SCIP HIGHS" solvers CVXPY


#Use the fake data from stephanie. 
# 1. create pandas script to read data
#    - nettoyer les donnees (clean data)
#    - ensure data types and format is compatible with solver (for categorical or text data) Is order an underlying assumption of categorical data for OR models (for example animals)
# 2. Add student grouping preferences to model (group students as "1 person" if they want to be togther, and represent the 1 student as the "worst case" for each item) group students from the beginning to poentialy reduce the search space
# 
# 3. Implement model with solver. code the constraints. test with small dataset (track time)  use the fake data.

# 4. augment the fake data to resemble the real dta size and do tests. 

# Idea not certain- Factors for match to fail immediately 
# 1. For transportation, if employee answers no to Greater MTL area and can't help student, and if the student has no car then it can't be a match.
# 2. language incompatitbility (enlighs student)
# 3. Animal allergies
# 4. Time incompatibilities 
# So we can manually pre-process the data to remove (Robin: just remove a variable)

# 