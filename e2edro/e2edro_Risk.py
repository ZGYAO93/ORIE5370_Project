# E2E DRO Module
#
####################################################################################################
## Import libraries
####################################################################################################
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import e2edro.RiskFunctions as rf
import e2edro.LossFunctions as lf
import e2edro.PortfolioClasses as pc
import e2edro.DataLoad as dl

import psutil
num_cores = psutil.cpu_count()
torch.set_num_threads(num_cores)
if psutil.MACOS:
    num_cores = 0

####################################################################################################
# CvxpyLayers: Differentiable optimization layers (nominal and distributionally robust)
####################################################################################################
#---------------------------------------------------------------------------------------------------
# base_mod: CvxpyLayer that declares the portfolio optimization problem
#---------------------------------------------------------------------------------------------------
def base_mod(n_y, n_obs, prisk):
    """Base optimization problem declared as a CvxpyLayer object

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    
    Parameters
    ep: (n_obs x n_y) matrix of residuals 
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
    returns)
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)

    Objective
    Minimize -y_hat @ z
    """
    # Variables
    z = cp.Variable((n_y,1), nonneg=True)

    # Parameters
    y_hat = cp.Parameter(n_y)
    
    # Constraints
    constraints = [cp.sum(z) == 1]

    # Objective function
    objective = cp.Minimize(-y_hat @ z)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[y_hat], variables=[z])


#---------------------------------------------------------------------------------------------------
# nominal: CvxpyLayer that declares the portfolio optimization problem
#---------------------------------------------------------------------------------------------------
def nominal(n_y, n_obs, prisk):
    """Nominal optimization problem declared as a CvxpyLayer object

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar
    obj_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Parameters
    ep: (n_obs x n_y) matrix of residuals 
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
    returns)
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)

    Objective
    Minimize (1/n_obs) * cp.sum(obj_aux) - gamma * mu_aux
    """
    # Variables
    z = cp.Variable((n_y,1), nonneg=True)
    c_aux = cp.Variable()
    obj_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    
    # Constraints
    constraints = [cp.sum(z) == 1,
                    mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [obj_aux[i] >= prisk(z, c_aux, ep[i])]

    # Objective function
    objective = cp.Minimize((1/n_obs) * cp.sum(obj_aux) - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma], variables=[z])

#---------------------------------------------------------------------------------------------------
# Total Variation: sum_t abs(p_t - q_t) <= delta
#---------------------------------------------------------------------------------------------------
def tv(n_y, n_obs, prisk):
    """DRO layer using the 'Total Variation' distance to define the probability ambiguity set.
    From Ben-Tal et al. (2013).
    Total Variation: sum_t abs(p_t - q_t) <= delta

    Inputs
    n_y: Number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar. Allows us to p-linearize the derivation of the variance
    lambda_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    eta_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    obj_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.

    Parameters
    ep: (n_obs x n_y) matrix of residuals 
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
    returns)
    delta: Scalar. Maximum distance between p and q.
    gamma: Scalar. Trade-off between conditional expected return and model error.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)
    All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al.
    (2013).

    Objective
    Minimize eta_aux + delta * lambda_aux + (1/n_obs) * sum(beta_aux) - gamma * y_hat @ z
    """

    # Variables
    z = cp.Variable((n_y,1), nonneg=True)
    c_aux = cp.Variable()
    lambda_aux = cp.Variable(nonneg=True)
    eta_aux = cp.Variable()
    beta_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    delta = cp.Parameter(nonneg=True)
    
    # Constraints
    constraints = [cp.sum(z) == 1,
                    beta_aux >= -lambda_aux,
                    mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [beta_aux[i] >= prisk(z, c_aux, ep[i]) - eta_aux]
        constraints += [lambda_aux >= prisk(z, c_aux, ep[i]) - eta_aux]

    # Objective function
    objective = cp.Minimize(eta_aux + delta * lambda_aux + (1/n_obs) * cp.sum(beta_aux)
                            - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])

#---------------------------------------------------------------------------------------------------
# Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta
#---------------------------------------------------------------------------------------------------
def hellinger(n_y, n_obs, prisk):
    """DRO layer using the Hellinger distance to define the probability ambiguity set.
    from Ben-Tal et al. (2013).
    Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar. Allows us to p-linearize the derivation of the variance
    lambda_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    xi_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    beta_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.
    s_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable SOC constraint.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Parameters
    ep: (n_obs x n_y) matrix of residuals 
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
    returns)
    delta: Scalar. Maximum distance between p and q.
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)
    All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al.
    (2013).

    Objective
    Minimize xi_aux + (delta-1) * lambda_aux + (1/n_obs) * sum(beta_aux) - gamma * y_hat @ z
    """

    # Variables
    z = cp.Variable((n_y,1), nonneg=True)
    c_aux = cp.Variable()
    lambda_aux = cp.Variable(nonneg=True)
    xi_aux = cp.Variable()
    beta_aux = cp.Variable(n_obs, nonneg=True)
    tau_aux = cp.Variable(n_obs, nonneg=True)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    delta = cp.Parameter(nonneg=True)

    # Constraints
    constraints = [cp.sum(z) == 1,
                    mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [xi_aux + lambda_aux >= prisk(z, c_aux, ep[i]) + tau_aux[i]]
        constraints += [beta_aux[i] >= cp.quad_over_lin(lambda_aux, tau_aux[i])]
    
    # Objective function
    objective = cp.Minimize(xi_aux + (delta-1) * lambda_aux + (1/n_obs) * cp.sum(beta_aux) 
                            - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)
    
    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])

####################################################################################################
# All price prediction opt fuctions(Guangyu)
####################################################################################################

#---------------------------------------------------------------------------------------------------
# Markowitz and variants: CvxpyLayer that declares the portfolio optimization problem
#---------------------------------------------------------------------------------------------------

def markowitz_mod(n_y, n_obs, prisk, variant="standard", **kwargs):
    """Markowitz optimization problem with multiple variants as a CvxpyLayer object
    
    Inputs:
    -------
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    variant: String indicating which variant to use:
        - "standard": Standard mean-variance optimization 
        - "min_variance": Minimum variance portfolio
        - "max_sharpe": Maximum Sharpe ratio portfolio
        - "risk_parity": Equal risk contribution portfolio
        - "target_return": Efficient frontier portfolio with target return
    
    Optional Parameters (depending on variant):
    ------------------------------------------
    risk_aversion: Float, trade-off parameter for standard variant
    rf_rate: Float, risk-free rate for max_sharpe variant
    target_ret: Float, target return for target_return variant
    
    Returns:
    --------
    CvxpyLayer object configured for the selected portfolio optimization variant
    """
    # Common variables for all variants
    z = cp.Variable((n_y, 1), nonneg=True)
    
    # Common constraints for all variants
    constraints = [cp.sum(z) == 1]  # Budget constraint
    
    # Implement different variants
    if variant == "standard":
        # Standard mean-variance optimization
        risk_aversion = kwargs.get("risk_aversion", 1.0)
        
        # Variables
        z = cp.Variable((n_y, 1), nonneg=True)
        
        # Parameters
        y_hat = cp.Parameter(n_y)  # Expected returns
        # Instead of using cov_matrix directly, we'll use its square root
        cov_sqrt = cp.Parameter((n_y, n_y))  # Square root of covariance matrix
        
        # Constraints
        constraints = [cp.sum(z) == 1]  # Budget constraint
        
        # Objective: maximize return - risk_aversion * variance
        # The key transformation: replace cp.quad_form(z, cov_matrix) with cp.sum_squares(cov_sqrt @ z)
        # This is mathematically equivalent but DPP-compliant
        objective = cp.Minimize(-y_hat @ z + risk_aversion * cp.sum_squares(cov_sqrt @ z))
        
        # Create problem
        problem = cp.Problem(objective, constraints)
        
        # Create the layer
        layer = CvxpyLayer(problem, parameters=[y_hat, cov_sqrt], variables=[z])
        
        # Wrapper class to handle the covariance decomposition
        class StandardMarkowitzLayer(torch.nn.Module):
            def __init__(self, cvxpy_layer):
                super().__init__()
                self.layer = cvxpy_layer
                
            def forward(self, expected_returns, cov_matrix):
                # Convert Parameter objects to Tensors if needed
                if isinstance(expected_returns, torch.nn.Parameter):
                    expected_returns = expected_returns.data
                if isinstance(cov_matrix, torch.nn.Parameter):
                    cov_matrix = cov_matrix.data
                
                # Compute the square root of the covariance matrix
                try:
                    # Try Cholesky decomposition first (more efficient)
                    L = torch.linalg.cholesky(cov_matrix)
                except:
                    # Fall back to eigendecomposition if Cholesky fails
                    # (e.g., if matrix is not positive definite)
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
                    # Ensure positivity of eigenvalues for numerical stability
                    eigenvalues = torch.clamp(eigenvalues, min=1e-6)
                    # Compute the square root matrix
                    L = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.t()
                
                # Solve the optimization problem with expected returns and cov_sqrt
                weights = self.layer(expected_returns, L)[0]
                return weights
        
        return StandardMarkowitzLayer(layer)
    
    elif variant == "min_variance":
        # Minimum variance portfolio
        
        # Parameters
        cov_matrix = cp.Parameter((n_y, n_y), PSD=True)
        
        # Objective: minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(z, cov_matrix))
        
        # Create problem
        problem = cp.Problem(objective, constraints)
        return CvxpyLayer(problem, parameters=[cov_matrix], variables=[z])
    
    elif variant == "max_sharpe":
        # Maximum Sharpe ratio portfolio
        # Note: This is a reformulation using a change of variables
        rf_rate = kwargs.get("rf_rate", 0.0)
        
        # Parameters
        y_hat = cp.Parameter(n_y)
        cov_matrix = cp.Parameter((n_y, n_y), PSD=True)
        
        # We use a change of variables approach to handle the non-convex Sharpe ratio
        # Let y = z/k where k is a scalar. Then we maximize (y_hat @ y - rf_rate * sum(y))
        # subject to quad_form(y, cov_matrix) = 1 and y >= 0
        y = cp.Variable((n_y, 1), nonneg=True)
        
        # Objective: maximize excess return with fixed risk
        objective = cp.Maximize(y_hat @ y - rf_rate * cp.sum(y))
        
        # Constraints: unit risk and non-negative weights
        max_sharpe_constraints = [cp.quad_form(y, cov_matrix) == 1]
        
        # Create problem 
        problem = cp.Problem(objective, max_sharpe_constraints)
        
        # We'll need to normalize the output to get portfolio weights
        class MaxSharpeLayer(torch.nn.Module):
            def __init__(self, cvxpy_layer):
                super().__init__()
                self.layer = cvxpy_layer
                
            def forward(self, y_hat, cov_matrix):
                # Solve the optimization problem
                y = self.layer(y_hat, cov_matrix)[0]
                
                # Normalize to get weights that sum to 1
                z = y / torch.sum(y)
                return z
                
        cvxpy_layer = CvxpyLayer(problem, parameters=[y_hat, cov_matrix], variables=[y])
        return MaxSharpeLayer(cvxpy_layer)
    
    elif variant == "target_return":
        # Efficient frontier portfolio with target return
        target_ret = kwargs.get("target_ret", 0.0)
        
        # Parameters
        y_hat = cp.Parameter(n_y)
        cov_matrix = cp.Parameter((n_y, n_y), PSD=True)
        target_return = cp.Parameter()
        
        # Objective: minimize variance for a given target return
        objective = cp.Minimize(cp.quad_form(z, cov_matrix))
        
        # Additional constraint: achieve target return
        target_return_constraints = constraints + [y_hat @ z >= target_return]
        
        # Create problem
        problem = cp.Problem(objective, target_return_constraints)
        return CvxpyLayer(problem, parameters=[cov_matrix, y_hat, target_return], variables=[z])
    
    else:
        raise ValueError(f"Unknown variant: {variant}. Available variants: 'standard', 'min_variance', 'max_sharpe', 'risk_parity', 'target_return'")

#---------------------------------------------------------------------------------------------------
# CVaR_mod and variants
#---------------------------------------------------------------------------------------------------

def cvar_portfolio_mod(n_y, n_obs, prisk, variant="min_cvar", **kwargs):
    """
    CVaR-based portfolio optimization variants as a CvxpyLayer object
    
    Inputs:
    -------
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    variant: String indicating which variant to use:
        - "min_cvar": Minimize CVaR (most conservative)
        - "mean_cvar": Balance between return and CVaR (like mean-variance)
        - "cvar_constrained": Maximize return subject to CVaR constraint
        - "cvar_ratio": Maximize return-to-CVaR ratio (like Sharpe ratio)
    
    Optional Parameters:
    ------------------
    alpha: Confidence level for CVaR (default: 0.95)
    risk_aversion: Trade-off parameter for mean_cvar variant
    cvar_limit: Upper bound on CVaR for constrained variant
    
    Returns:
    --------
    CvxpyLayer object configured for the selected CVaR optimization variant
    """
    # Common variables for all variants
    z = cp.Variable((n_y, 1), nonneg=True)  # Portfolio weights
    
    # Common constraints for all variants
    constraints = [cp.sum(z) == 1]  # Budget constraint
    
    # Extract parameters
    alpha = kwargs.get("alpha", 0.95)  # Default 95% confidence level
    
    # Variables for CVaR calculation
    # For sample-based CVaR computation, we need:
    # Sample returns matrix
    returns = cp.Parameter((n_obs, n_y)) 
    # Value-at-Risk variable
    var = cp.Variable(1)
    # Auxiliary variables for CVaR calculation
    aux_vars = cp.Variable(n_obs)
    
    # CVaR calculation constraints
    cvar_constraints = constraints + [
        aux_vars >= 0,
        aux_vars >= -returns @ z - var
    ]
    
    # Calculate CVaR: VaR + 1/((1-alpha)*n_obs) * sum of auxiliary variables
    # This is based on the Rockafellar & Uryasev formulation
    cvar_expr = var + (1/((1-alpha)*n_obs)) * cp.sum(aux_vars)
    
    # Implement different variants
    if variant == "min_cvar":
        # Simple minimum CVaR portfolio
        objective = cp.Minimize(cvar_expr)
        problem = cp.Problem(objective, cvar_constraints)
        return CvxpyLayer(problem, parameters=[returns], variables=[z])
    
    elif variant == "mean_cvar":
        # Mean-CVaR optimization (similar to mean-variance but with CVaR)
        risk_aversion = kwargs.get("risk_aversion", 1.0)
        # Expected returns vector
        expected_returns = cp.Parameter(n_y) 
        
        # Objective: maximize expected return - risk_aversion * CVaR
        objective = cp.Minimize(-expected_returns @ z + risk_aversion * cvar_expr)
        problem = cp.Problem(objective, cvar_constraints)
        return CvxpyLayer(problem, parameters=[returns, expected_returns], variables=[z])
    
    elif variant == "cvar_constrained":
        # Return maximization subject to CVaR constraint
        # Default 10% CVaR limit
        cvar_limit = kwargs.get("cvar_limit", 0.1) 
         # Expected returns vector 
        expected_returns = cp.Parameter(n_y) 
        # CVaR upper bound
        cvar_bound = cp.Parameter(1, nonneg=True)  
        
        # Add CVaR constraint
        cvar_bound_constraints = cvar_constraints + [cvar_expr <= cvar_bound]
        
        # Objective: maximize expected return
        objective = cp.Maximize(expected_returns @ z)
        problem = cp.Problem(objective, cvar_bound_constraints)
        return CvxpyLayer(problem, parameters=[returns, expected_returns, cvar_bound], variables=[z])
    
    elif variant == "cvar_ratio":
        # Maximize return-to-CVaR ratio (similar to Sharpe ratio)
        # This requires a non-convex formulation, so we use a change of variables approach
        
        # This is more complex to implement with CVaR than with standard deviation
        # use a fractional programming approach
        
        class CVaRRatioLayer(torch.nn.Module):
            def __init__(self, n_y, n_obs, alpha=0.95, max_iter=20):
                super().__init__()
                self.n_y = n_y
                self.n_obs = n_obs
                self.alpha = alpha
                self.max_iter = max_iter
                
                # Create a mean-CVaR layer that we'll use iteratively
                self.mean_cvar_layer = cvar_portfolio_mod(
                    n_y, n_obs, variant="mean_cvar", alpha=alpha)
            
            def forward(self, returns, expected_returns):
                # Initial guess for lambda (risk aversion parameter)
                lambda_t = 1.0
                z_t = torch.ones(self.n_y, 1, device=returns.device) / self.n_y
                
                # Bisection search to find optimal lambda for CVaR ratio maximization
                for _ in range(self.max_iter):
                    # Solve mean-CVaR problem with current lambda
                    z_new = self.mean_cvar_layer(returns, expected_returns * lambda_t)[0]
                    
                    # Calculate expected return and CVaR for new weights
                    expected_ret = torch.matmul(expected_returns.view(1, -1), z_new)
                    
                    # Compute CVaR (this is simplified - in practice you'd use the aux vars)
                    portfolio_returns = torch.matmul(returns, z_new)
                    sorted_returns, _ = torch.sort(portfolio_returns, dim=0)
                    var_index = int(self.n_obs * (1 - self.alpha))
                    var_t = -sorted_returns[var_index]
                    cvar_t = -torch.mean(sorted_returns[:var_index+1])
                    
                    # Calculate ratio
                    ratio = expected_ret / cvar_t
                    
                    # Update lambda based on bisection
                    if torch.abs(expected_ret - lambda_t * cvar_t) < 1e-6:
                        break
                    
                    lambda_t = expected_ret / cvar_t
                    z_t = z_new
                
                return z_t
        
        return CVaRRatioLayer(n_y, n_obs, alpha=alpha)
    
    else:
        raise ValueError(f"Unknown variant: {variant}. Available variants: 'min_cvar', 'mean_cvar', 'cvar_constrained', 'cvar_ratio'")

####################################################################################################
# All risk related opt fuctions(Jiayi)
####################################################################################################
def pred_sigma(y_hat, n_asset):
    """
    Convert the lower triangle elements into a symmetric positive definite matrix
    
    INPUTS:
    -------
    y_hat -> torch.Tensor: lower triangle elements of shape [batch_size, (n_asset*(n_asset+1))/2]
    n_asset -> int: number of assets
    
    OUTPUTS:
    --------
    sigma: symmetric positive definite matrix [batch_size, n_asset, n_asset]
    """
    # Check dimensions
    if len(y_hat.shape) != 2:
        raise ValueError(f"Expected 2D tensor, got shape {y_hat.shape}")
    
    n_data = y_hat.shape[0]
    expected_features = (n_asset * (n_asset + 1)) // 2
    
    if y_hat.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features for {n_asset} assets, got {y_hat.shape[1]}")
    
    device = y_hat.device
    dtype = y_hat.dtype
    
    # Create lower triangular matrix
    L = torch.zeros(n_data, n_asset, n_asset, device=device, dtype=dtype)
    
    # Get indices for lower triangular part
    tril_indices = torch.tril_indices(row=n_asset, col=n_asset, device=device)
    
    # Fill lower triangular part
    L[:, tril_indices[0], tril_indices[1]] = y_hat
    
    # Ensure positive diagonal (critical for positive definiteness)
    diag_indices = torch.arange(n_asset, device=device)
    L[:, diag_indices, diag_indices] = torch.nn.functional.softplus(L[:, diag_indices, diag_indices])
    
    # Compute sigma via L*L^T which guarantees positive semi-definiteness
    sigma = torch.bmm(L, L.transpose(1, 2))
    
    # Add regularization term (with proper broadcasting)
    reg_term = 1e-3 * torch.eye(n_asset, device=device, dtype=dtype).unsqueeze(0).expand(n_data, -1, -1)
    sigma = sigma + reg_term
    
    # Print shape info for debugging
    #print(f"pred_sigma input: {y_hat.shape}, output: {sigma.shape}")
    
    return sigma
# Risk budget optimization layer
def risk_budget(n_y):
    y = cp.Variable((n_y, 1), nonneg=True)  # allocation
    b = cp.Parameter((n_y, 1), nonneg=True)  # predicted risk budget
    c = cp.Parameter()
    Sigma = cp.Parameter((n_y, n_y), PSD=True)  # covariance matrix

    objective = cp.Minimize(cp.sum_squares(Sigma @ y))  # quadratic objective
    constraints = [b.T @ cp.log(y) >= c]  # DPP-compliant constraint
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[Sigma, b, c], variables=[y])

def risk_budget_mod(n_y):
    y = cp.Variable((n_y, 1), nonneg=True)  # allocation
    b = cp.Parameter((n_y, 1), nonneg=True)  # predicted risk budget
    c = cp.Parameter()
    Sigma = cp.Parameter((n_y, n_y), PSD=True)  # covariance matrix
    gamma = cp.Parameter(nonneg=True)  
    objective = cp.Minimize(cp.sum_squares(Sigma @ y) - gamma * cp.sum(cp.entr(y)))

    #objective = cp.Minimize(cp.sum_squares(Sigma @ y))  # quadratic objective
    constraints = [b.T @ cp.log(y) >= c]  # DPP-compliant constraint
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[Sigma, b, c, gamma], variables=[y])

def min_variance(n_y):
    Sigma = cp.Parameter((n_y, n_y), PSD=True)
    z = cp.Variable((n_y, 1), nonneg=True)
    objective = cp.Minimize(cp.sum_squares(Sigma @ z))
    constraints = [cp.sum(z) == 1]
    problem = cp.Problem(objective, constraints)
    return CvxpyLayer(problem, parameters=[Sigma], variables=[z])

def risk_parity(n_y): # same as risk_budgeting portfolio
    y = cp.Variable((n_y, 1), nonneg=True)  # allocation
    b = cp.Parameter((n_y, 1), nonneg=True)  # predicted risk budget
    c = cp.Parameter()
    Sigma = cp.Parameter((n_y, n_y), PSD=True)  # covariance matrix

    objective = cp.Minimize(cp.sum_squares(Sigma @ y))  # quadratic objective
    constraints = [b.T @ cp.log(y) >= c]  # DPP-compliant constraint
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[Sigma, b, c], variables=[y])
####################################################################################################
# E2E neural network module
####################################################################################################
class e2e_net(nn.Module):
    """End-to-end DRO learning neural net module.
    """
    def __init__(self, n_x, n_asset, n_obs, opt_layer='nominal', prisk='p_var', perf_loss='sharpe_loss',
                pred_model='linear', pred_loss_factor=0.5, perf_period=13, train_pred=True, train_gamma=True, train_delta=True, set_seed=None, cache_path='./cache/',variant="standard"):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        opt_layer: String. Determines which CvxpyLayer-object to call for the optimization layer
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss_factor: Trade-off between prediction loss function and performance loss function.
            Set 'pred_loss_factor=None' to define the loss function purely as 'perf_loss'
        perf_period: Number of lookahead realizations used in 'perf_loss()'
        train_pred: Boolean. Choose if the prediction layer is learnable (or keep it fixed)
        train_gamma: Boolean. Choose if the risk appetite parameter gamma is learnable
        train_delta: Boolean. Choose if the robustness parameter delta is learnable
        set_seed: (Optional) Int. Set the random seed for replicability

        Output
        e2e_net: nn.Module object 
        """
        super(e2e_net, self).__init__()

        # Set random seed (to be used for replicability of numerical experiments)
        if set_seed is not None:
            torch.manual_seed(set_seed)
            self.seed = set_seed

        self.n_x = n_x
        self.n_obs = n_obs
        self.n_asset = n_asset
        # Prediction loss function
        if pred_loss_factor is not None:
            self.pred_loss_factor = pred_loss_factor
            self.pred_loss = torch.nn.MSELoss()
        else:
            self.pred_loss = None

        # Define performance loss
        self.perf_loss = eval('lf.'+perf_loss)

        # Number of time steps to evaluate the task loss
        self.perf_period = perf_period

        # Register 'gamma' (risk-return trade-off parameter)
        self.gamma = nn.Parameter(torch.FloatTensor(1).uniform_(0.02, 0.1))
        self.gamma.requires_grad = train_gamma
        self.gamma_init = self.gamma.item()

        # Record the model design: nominal, base or DRO
        if opt_layer == 'nominal':
            self.model_type = 'nom'
        elif opt_layer == 'base_mod':
            self.gamma.requires_grad = False
            self.model_type = 'base_mod' 
        elif opt_layer == 'markowitz_mod':
            self.gamma.requires_grad = False
            self.model_type = 'markowitz_mod' 

        elif opt_layer == 'risk_budget' or opt_layer == 'risk_budget_mod':
            self.model_type = opt_layer
            self.gamma.requires_grad = False
            self.n_y = n_asset  # number of output from neural network should align with number of asset (budget)
            self.pred_layer = nn.Sequential(
                nn.Linear(n_x, 32),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(32, self.n_y),
                nn.Softmax(dim=-1)  # risk budget should sum up to 1
            )
        elif opt_layer == 'min_variance' or opt_layer == 'risk_parity':
            self.model_type = opt_layer
            self.gamma.requires_grad = False
            self.n_y = int(n_asset*(n_asset+1)/2)
            self.pred_layer = nn.Sequential(
                nn.Linear(n_x, 32),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(32, self.n_y)
                # no softmax layer
            )

        # LAYER: Optimization model
        jiayi_models = ['risk_budget','risk_budget_mod','min_variance','risk_parity']
        self.opt_layer = eval(opt_layer)(n_asset)
        
          
        
        # Store reference path to store model data
        self.cache_path = cache_path

        # Store initial model
        if train_gamma and train_delta:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model
        elif train_delta and not train_gamma:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model + '_TrainGamma'+str(train_gamma)
        elif train_gamma and not train_delta:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model + '_TrainDelta'+str(train_delta)
        elif not train_gamma and not train_delta:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model + '_TrainGamma'+str(train_gamma) + '_TrainDelta'+str(train_delta)
        torch.save(self.state_dict(), self.init_state_path)

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, x, Sigma, c=torch.tensor([0.1])):
        solver_args = {
            'solve_method': 'ECOS', 
            'max_iters': 500,  # Increase from 120
            'abstol': 1e-6,    # Relax from 1e-7
            'reltol': 1e-5,    # Add relative tolerance
            'verbose': False
        }
        if self.model_type == 'risk_budget':
            b = self.pred_layer(x)
            b = b.unsqueeze(-1)  # (batch_size, n_y, 1)
            Sigma_exp = Sigma.expand(x.shape[0], -1, -1)  # batch compatible
            c_exp = c.expand(x.shape[0])
            y_opt, = self.opt_layer(Sigma_exp, b, c_exp)
            y_opt_normalized = y_opt / y_opt.sum(dim=1, keepdim=True)
            return b.squeeze(-1), y_opt_normalized
        
        elif self.model_type == 'min_variance':
            b = self.pred_layer(x)
            Sigma = pred_sigma(b,self.n_asset)
            Sigma_exp = Sigma.expand(x.shape[0], -1, -1)  # batch compatible
            c_exp = c.expand(x.shape[0])
            y_opt, = self.opt_layer(Sigma_exp)
            y_opt_normalized = y_opt / y_opt.sum(dim=1, keepdim=True)
            return b.squeeze(-1), y_opt_normalized
        
        elif self.model_type == 'risk_parity':
            b = self.pred_layer(x)
            Sigma = pred_sigma(b,self.n_asset)
            Sigma_exp = Sigma.expand(x.shape[0], -1, -1)  # batch compatible
            c = torch.tensor([1.0])
            c_exp = c.expand(x.shape[0])
            b = torch.full((b.shape[0], self.n_asset, 1), 1.0 / self.n_asset, dtype=Sigma.dtype)
            for _ in range(50):
                y_star, = self.opt_layer(Sigma_exp, b, c_exp, solver_args=solver_args)
                #y_star = y_star.squeeze(-1) if y_star.dim() > 2 else y_star
                # Compute realised contributions
                RC = y_star * (Sigma @ y_star)
                RC = RC / RC.sum(dim=-2, keepdim=True)
                dispersion = RC.std(dim=-2).max()
                b = RC.detach()               # update budgets (stop‑grad for stability)
                if dispersion < 1e-6:
                    break
            y_opt_normalized = y_star / y_star.sum(dim=1, keepdim=True)
            print(y_opt_normalized.shape)
            return b.squeeze(-1), y_opt_normalized
        
        elif self.model_type == 'risk_budget_mod':  
            b = self.pred_layer(x)
            b = b.unsqueeze(-1)  # (batch_size, n_y, 1)
            Sigma_exp = Sigma.expand(x.shape[0], -1, -1)  # batch compatible
            c_exp = c.expand(x.shape[0])
            gamma = torch.tensor([0.1])
            gamma_exp = gamma.expand(x.shape[0])
            y_opt, = self.opt_layer(Sigma_exp, b, c_exp, gamma_exp)
            y_opt_normalized = y_opt / y_opt.sum(dim=1, keepdim=True)
            return b.squeeze(-1), y_opt_normalized
            
    #-----------------------------------------------------------------------------------------------
    # net_cv: Cross validation of the e2e neural net for hyperparameter tuning
    #-----------------------------------------------------------------------------------------------
    def net_cv(self, X, Y, lr_list, epoch_list, n_val=4):
        """Neural net cross-validation module for risk-based models

        Inputs
        X: Features. TrainTest object of feature timeseries data
        Y: Realizations. TrainTest object of asset time series data
        epochs: number of training passes
        lr_list: List of candidate learning rates
        epoch_list: List of candidate number of epochs
        n_val: Number of validation folds from the training dataset
        
        Output
        Trained model
        """
        results = pc.CrossVal()
        X_temp = dl.TrainTest(X.train, X.n_obs, [1, 0])
        Y_temp = dl.TrainTest(Y.train, Y.n_obs, [1, 0])
        
        for epochs in epoch_list:
            for lr in lr_list:
                
                # Train the neural network
                print('================================================')
                print(f"Training E2E {self.model_type} model: lr={lr}, epochs={epochs}")
                
                val_loss_tot = []
                for i in range(n_val-1,-1,-1):

                    # Partition training dataset into training and validation subset
                    split = [round(1-0.2*(i+1),2), 0.2]
                    X_temp.split_update(split)
                    Y_temp.split_update(split)

                    X_train, Y_train = X_temp.train(), Y_temp.train()
                    X_val, Y_val = X_temp.test(), Y_temp.test()
                    
                    if hasattr(self, 'init_state_path'):
                        self.load_state_dict(torch.load(self.init_state_path))
                    else:
                        # Save initial state if not already saved
                        self.init_state_path = f"./cache/{self.model_type}_initial_state_{int(time.time())}.pt"
                        torch.save(self.state_dict(), self.init_state_path)
                        self.load_state_dict(torch.load(self.init_state_path))

                    # Compute covariance matrix for risk-based models
                    if self.model_type in ['risk_parity', 'risk_budget', 'risk_budget_mod', 'min_variance']:
                        # change to tensors
                        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
                        Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
                        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
                        Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.float32)
                        
                        # Get cov matrix
                        Y_centered = Y_train_tensor - Y_train_tensor.mean(dim=0)
                        sigma_train = (Y_centered.T @ Y_centered) / (len(Y_train_tensor) - 1)
                        
                        Y_val_centered = Y_val_tensor - Y_val_tensor.mean(dim=0)
                        sigma_val = (Y_val_centered.T @ Y_val_centered) / (len(Y_val_tensor) - 1)
                        
                        c = torch.tensor([0.0])
                        
                        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
                        
                        # Training loop
                        for epoch in range(epochs):
                            optimizer.zero_grad()
                            
                            if self.model_type == 'risk_parity':
                                pred, weights = self(X_train_tensor, sigma_train, c)
                                print(f"Debug - weights shape before: {weights.shape}")
                                # If weights is [batch_size, 1], reshape it to [batch_size, n_asset]
                                if weights.shape[1] == 1:
                                    weights = weights.expand(-1, self.n_asset)
                                # If weights is [batch_size, n_asset, 1], reshape it
                                elif weights.dim() == 3 and weights.shape[2] == 1:
                                    weights = weights.squeeze(-1)
                                print(f"Debug - weights shape after: {weights.shape}")

                                # risk contributions
                                RC = weights * (sigma_train @ weights.unsqueeze(-1)).squeeze(-1)
                                RC = RC / RC.sum(dim=1, keepdim=True)
                                target = torch.full_like(RC, 1.0 / self.n_asset)
                                
                                loss = torch.nn.functional.mse_loss(RC, target)
                            else:
                                pred, weights = self(X_train_tensor, sigma_train, c)
                                if self.model_type == 'risk_budget' or self.model_type == 'risk_budget_mod':
                                    # Target = predicted risk budget
                                    loss = torch.nn.functional.mse_loss(pred, torch.full_like(pred, 1.0 / self.n_asset))
                                else:
                                    # Add reshaping logic here
                                    if weights.shape[1] == 1:
                                        weights = weights.expand(-1, self.n_asset)
                                    elif weights.dim() == 3 and weights.shape[2] == 1:
                                        weights = weights.squeeze(-1)
                                    port_variance = torch.sum(weights * (sigma_train @ weights.unsqueeze(-1)).squeeze(-1), dim=1)
                                    loss = port_variance.mean()
                            
                            loss.backward()
                            
                            optimizer.step()
                        
                        # Evaluate on validation set
                        with torch.no_grad():
                            if self.model_type == 'risk_parity':
                                pred_val, weights_val = self(X_val_tensor, sigma_val, c)
                                # risk contribution
                                RC_val = weights_val * (sigma_val @ weights_val.unsqueeze(-1)).squeeze(-1)
                                RC_val = RC_val / RC_val.sum(dim=1, keepdim=True)
                                target_val = torch.full_like(RC_val, 1.0 / self.n_asset)
                                val_loss = torch.nn.functional.mse_loss(RC_val, target_val).item()
                            else:
                                pred_val, weights_val = self(X_val_tensor, sigma_val, c)
                                
                                if self.model_type == 'risk_budget' or self.model_type == 'risk_budget_mod':
                                    val_loss = torch.nn.functional.mse_loss(
                                        pred_val, torch.full_like(pred_val, 1.0 / self.n_asset)
                                    ).item()
                                else: # min variance
                                    # Add reshaping logic here
                                    if weights_val.shape[1] == 1:
                                        weights_val = weights_val.expand(-1, self.n_asset)
                                    elif weights_val.dim() == 3 and weights_val.shape[2] == 1:
                                        weights_val = weights_val.squeeze(-1)
                                    port_variance = torch.sum(weights_val * (sigma_val @ weights_val.unsqueeze(-1)).squeeze(-1), dim=1)
                                    val_loss = port_variance.mean().item()
                    

                    val_loss_tot.append(val_loss)
                    print(f"Fold: {n_val-i} / {n_val}, val_loss: {val_loss}")

                # Store results
                results.val_loss.append(np.mean(val_loss_tot))
                results.lr.append(lr)
                results.epochs.append(epochs)
                print(f"Average validation loss: {np.mean(val_loss_tot):.6f}")
                print('================================================')

        # Convert results to dataframe
        self.cv_results = results.df()
        self.cv_results.to_pickle(self.init_state_path+'_results.pkl')

        # Select and store the optimal hyperparameters
        idx = self.cv_results.val_loss.idxmin()
        self.lr = self.cv_results.lr[idx]
        self.epochs = self.cv_results.epochs[idx]

        # Print optimal parameters
        print(f"CV E2E {self.model_type} with hyperparameters: lr={self.lr}, epochs={self.epochs}")
        

    #-----------------------------------------------------------------------------------------------
    # net_roll_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4, lr=None, epochs=None):
        """Neural net rolling window out-of-sample test for risk parity models

        Inputs
        X: Features. TrainTest object with feature timeseries data
        Y: Realizations. TrainTest object with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)
        lr: Learning rate for test. If 'None', the optimal learning rate is loaded
        epochs: Number of epochs for test. If 'None', the optimal # of epochs is loaded

        Output 
        self.portfolio: add the backtest results to the e2e_net object
        """
        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(Y.test)-Y.n_obs, self.n_asset, Y.test.index[Y.n_obs:])

        # Store initial train/test split
        init_split = Y.split

        # Window size
        win_size = init_split[1] / n_roll

        c = torch.tensor([0.0])

        split = [0, 0]
        t = 0
        for i in range(n_roll):
            print(f"Out-of-sample window: {i+1} / {n_roll}")

            split[0] = init_split[0] + win_size * i
            if i < n_roll-1:
                split[1] = win_size
            else:
                split[1] = 1 - split[0]

            X.split_update(split), Y.split_update(split)
            
            # For risk parity models, we handle the data differently
            if self.model_type in ['risk_parity', 'risk_budget', 'risk_budget_mod', 'min_variance']:
                X_train = torch.tensor(X.train.values, dtype=torch.float32)
                Y_train = torch.tensor(Y.train.values, dtype=torch.float32)
                X_test = torch.tensor(X.test.values, dtype=torch.float32)
                Y_test = torch.tensor(Y.test.values, dtype=torch.float32)
                
                # Compute covariance matrix 
                Y_centered = Y_train - Y_train.mean(dim=0)
                sigma_train = (Y_centered.T @ Y_centered) / (len(Y_train) - 1)
                
                Y_test_centered = Y_test - Y_test.mean(dim=0)
                sigma_test = (Y_test_centered.T @ Y_test_centered) / (len(Y_test) - 1)
                
                # Reset model parameters
                if hasattr(self, 'init_state_path'):
                    self.load_state_dict(torch.load(self.init_state_path))
                
                # optimizer
                optimizer = torch.optim.Adam(self.parameters(), lr=lr if lr is not None else self.lr)
                
                # Training 
                for epoch in range(epochs if epochs is not None else self.epochs):
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    if self.model_type == 'risk_parity':
                        pred, weights = self(X_train, sigma_train, c)
                        
                        # risk contributions
                        RC = weights * (sigma_train @ weights.unsqueeze(-1)).squeeze(-1)
                        RC = RC / RC.sum(dim=1, keepdim=True)
                        target = torch.full_like(RC, 1.0 / self.n_asset)
                        
                        loss = torch.nn.functional.mse_loss(RC, target)
                    else:
                        pred, weights = self(X_train, sigma_train, c)
                        
                        if self.model_type == 'risk_budget' or self.model_type == 'risk_budget_mod':
                            # Target = predicted risk budget
                            loss = torch.nn.functional.mse_loss(pred, torch.full_like(pred, 1.0 / self.n_asset))
                        else: # min variance
                            port_variance = torch.sum(weights * (sigma_train @ weights.unsqueeze(-1)).squeeze(-1), dim=1)
                            loss = port_variance.mean()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    optimizer.step()
                
                # Test model
                with torch.no_grad():
                    # Get predictions for each test point
                    for j in range(len(X_test)):
                        x_t = X_test[j:j+1]
                        
                        y_perf = Y_test[j] if j < len(Y_test) else torch.zeros(self.n_asset)
                        
                        _, z_star = self(x_t, sigma_test, c)
                        
                        portfolio.weights[t] = z_star[0].squeeze().detach().cpu().numpy()

                        weights_tensor = torch.tensor(portfolio.weights[t], dtype=y_perf.dtype, device=y_perf.device)
                        portfolio.rets[t] = float(y_perf @ weights_tensor)
                        #portfolio.rets[t] = float(y_perf @ torch.tensor(portfolio.weights[t]))
                        
                        t += 1
            

        # Reset dataset
        X, Y = X.split_update(init_split), Y.split_update(init_split)

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio
        
        return portfolio

    def quick_test(self, X, Y, holding_period=14):
        """
        Tests performance with a specified holding period (e.g., 30 days)
        
        For each day:
        1. Get model weights
        2. Calculate return over the next 'holding_period' days
        3. Repeat for each day in the test set
        """
        # Get test data without the sliding window - we need all price data
        X_test = X.test
        Y_test = Y.test
        
        # Convert to torch tensors
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)
        
        # Get test dates
        test_dates = Y_test.index
        
        # Determine how many test points we can evaluate
        # We need enough future data for the holding period
        max_start_idx = len(test_dates) - holding_period
        
        # Create portfolio object
        portfolio = pc.backtest(max_start_idx, self.n_asset, test_dates[:max_start_idx])
        
        print(f"Testing {self.n_asset} assets with {holding_period}-day holding period across {max_start_idx} start dates")
        
        # Compute covariance matrix for the entire dataset
        Y_centered = Y_test_tensor - Y_test_tensor.mean(dim=0)
        sigma = torch.matmul(Y_centered.T, Y_centered) / (len(Y_centered) - 1)
        sigma = sigma + torch.eye(self.n_asset) * 1e-3
        
        # Constraint parameter
        c = torch.tensor([0.0], dtype=torch.float32)
        
        # For each possible start date
        with torch.no_grad():
            for t in range(max_start_idx):
                # Get features for current day
                x_t = X_test_tensor[t:t+1]
                
                # Forward pass to get weights
                _, weights = self(x_t, sigma, c)
                
                # Get portfolio weights
                z_star = weights[0] if weights.dim() > 1 else weights
                #z_star = z_star.squeeze()

                raw_weights = z_star.squeeze().detach().cpu().numpy()
                # Ensure weights sum to 1
                normalized_weights = raw_weights / np.sum(raw_weights) if np.sum(raw_weights) != 0 else raw_weights
                # Assign normalized weights
                portfolio.weights[t] = normalized_weights

                # # Store portfolio weights
                # portfolio.weights[t] = z_star.detach().cpu().numpy()
                
                # Calculate return over holding period:
                # 1. Get prices at start of period
                start_prices = Y_test_tensor[t]
                
                # 2. Get prices at end of period
                end_prices = Y_test_tensor[t + holding_period]
                
                # 3. Calculate percentage change over period
                holding_period_returns = (end_prices - start_prices) / start_prices
                
                weights_tensor = torch.tensor(portfolio.weights[t], dtype=holding_period_returns.dtype, device=holding_period_returns.device)
                portfolio_return = float(torch.sum(weights_tensor * holding_period_returns))

                # 4. Calculate portfolio return (weighted sum of asset returns)
                #portfolio_return = float(torch.sum(z_star * holding_period_returns))
                
                # 5. Store the return (annualize if needed)
                portfolio.rets[t] = portfolio_return
                
        # Calculate portfolio statistics
        try:
            portfolio.stats()
            print(f"\nPerformance over {holding_period}-day holding periods:")
            print(f"Mean Return: {portfolio.mu:.4f}")
            print(f"Volatility: {portfolio.vol:.4f}")
            print(f"Sharpe Ratio: {portfolio.sharpe:.4f}")
        except Exception as e:
            print(f"Error in stats calculation: {e}")
            # Manual calculation
            returns = np.array(portfolio.rets)
            print(f"Mean Return: {np.mean(returns):.4f}")
            print(f"Volatility: {np.std(returns):.4f}")
            if np.std(returns) > 0:
                print(f"Sharpe Ratio: {np.mean(returns)/np.std(returns):.4f}")
        
        self.portfolio = portfolio
        return portfolio

    def non_overlapping_risk_test(self, X, Y, holding_period=14):
        """
        Tests performance of risk budget models using non-overlapping windows.
        Each holding period uses weights calculated once at the beginning of the period.
        
        Parameters:
        -----------
        model : torch.nn.Module
            The trained risk budget model
        X : TrainTest
            Features data object
        Y : TrainTest
            Asset prices data object
        holding_period : int
            Number of days to hold each portfolio allocation
            
        Returns:
        --------
        portfolio : object
            Portfolio object with daily performance data
        """
        # Get test data without the sliding window - we need all price data
        X_test = X.test
        Y_test = Y.test
        
        # Convert to torch tensors
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)
        
        # Get test dates
        test_dates = Y_test.index
        
        # Calculate how many non-overlapping periods fit in the test set
        available_days = len(test_dates)
        num_periods = available_days // holding_period
        
        # Total number of days we'll track in our portfolio
        total_days = num_periods * holding_period
        
        # Make sure we don't exceed available data
        if total_days > available_days:
            total_days = available_days
        
        # Get number of assets
        n_assets = self.n_asset
        
        # Create portfolio object for storing daily results
        portfolio = pc.backtest(total_days - 1, n_assets, test_dates[:total_days - 1])
        
        # Arrays to store weights and daily returns
        all_weights = np.zeros((total_days - 1, n_assets))
        daily_returns = np.zeros(total_days - 1)
        
        print(f"Testing {n_assets} assets with {holding_period}-day rebalancing across {num_periods} periods")
        
        # Initialize portfolio value series (starting at 1.0)
        portfolio_values = np.ones(total_days)
        
        # Compute covariance matrix for the entire dataset
        Y_centered = Y_test_tensor - Y_test_tensor.mean(dim=0)
        sigma = torch.matmul(Y_centered.T, Y_centered) / (len(Y_centered) - 1)
        sigma = sigma + torch.eye(n_assets) * 1e-3
        
        # Constraint parameter
        c = torch.tensor([0.0], dtype=torch.float32)
        
        with torch.no_grad():
            # For each non-overlapping holding period
            for period in range(num_periods):
                # Calculate the starting index for this period
                period_start_idx = period * holding_period
                period_end_idx = min(period_start_idx + holding_period, total_days)
                
                print(f"Processing period {period+1}/{num_periods} (days {period_start_idx}-{period_end_idx-1})")
                
                try:
                    # Get features for current day (first day of the period)
                    x_t = X_test_tensor[period_start_idx:period_start_idx+1]
                    
                    # Forward pass to get weights
                    _, weights = self(x_t, sigma, c)
                    
                    # Get portfolio weights
                    z_star = weights[0] if weights.dim() > 1 else weights
                    raw_weights = z_star.squeeze().detach().cpu().numpy()
                    
                    # Ensure weights sum to 1
                    if np.sum(raw_weights) != 0:
                        normalized_weights = raw_weights / np.sum(raw_weights)
                    else:
                        normalized_weights = np.ones(n_assets) / n_assets  # Equal weights fallback
                    
                    # Store these weights for each day in the holding period
                    for t in range(period_start_idx, period_end_idx - 1):  # -1 because we need next day's data
                        day_index = t  # For indexing into portfolio arrays
                        if day_index < len(all_weights):
                            all_weights[day_index] = normalized_weights
                    
                    # Calculate daily returns during the holding period
                    for t in range(period_start_idx, period_end_idx - 1):  # -1 because we need next day's data
                        # Calculate daily return from t to t+1
                        curr_prices = Y_test_tensor[t]
                        next_prices = Y_test_tensor[t+1]
                        daily_price_returns = (next_prices - curr_prices) / curr_prices
                        
                        # Portfolio daily return
                        weights_tensor = torch.tensor(normalized_weights, dtype=daily_price_returns.dtype, 
                                                      device=daily_price_returns.device)
                        daily_return = float(torch.sum(weights_tensor * daily_price_returns))
                        
                        # Store daily return
                        daily_returns[t] = np.clip(daily_return, -0.25, 0.25)  # Clip extreme values
                        
                        # Update portfolio value
                        portfolio_values[t+1] = portfolio_values[t] * (1 + daily_return)
                    
                except Exception as e:
                    print(f"Error at period {period+1}: {e}")
                    # Use equal weights for this period
                    normalized_weights = np.ones(n_assets) / n_assets
                    
                    # Fill in weights and zeros for returns
                    for t in range(period_start_idx, period_end_idx - 1):
                        if t < len(all_weights):
                            all_weights[t] = normalized_weights
                            daily_returns[t] = 0.0
        
        # Store results in portfolio object
        portfolio.weights = all_weights
        portfolio.rets = daily_returns
        
        # Calculate portfolio statistics
        try:
            portfolio.stats()
            print(f"\nPerformance with {holding_period}-day rebalancing:")
            print(f"Mean Daily Return: {portfolio.mu:.4f}")
            print(f"Daily Volatility: {portfolio.vol:.4f}")
            print(f"Sharpe Ratio: {portfolio.sharpe:.4f}")
            
            # Calculate annualized metrics
            annual_factor = 252  # Trading days in a year
            annual_return = (1 + portfolio.mu)**annual_factor - 1
            annual_vol = portfolio.vol * np.sqrt(annual_factor)
            annual_sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            
            print(f"\nAnnualized Metrics:")
            print(f"Annual Return: {annual_return:.4f}")
            print(f"Annual Volatility: {annual_vol:.4f}")
            print(f"Annual Sharpe Ratio: {annual_sharpe:.4f}")
            
        except Exception as e:
            print(f"Error in stats calculation: {e}")
            # Manual calculation
            returns = np.array(daily_returns)
            valid_returns = returns[returns != 0]  # Filter out zeros
            
            if len(valid_returns) > 0:
                mean_return = np.mean(valid_returns)
                std_return = np.std(valid_returns)
                
                print(f"Mean Daily Return: {mean_return:.4f}")
                print(f"Daily Volatility: {std_return:.4f}")
                
                if std_return > 0:
                    print(f"Sharpe Ratio: {mean_return/std_return:.4f}")
                    
                    # Annualized metrics
                    annual_factor = 252  # Trading days in a year
                    annual_return = (1 + mean_return)**annual_factor - 1
                    annual_vol = std_return * np.sqrt(annual_factor)
                    annual_sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    print(f"\nAnnualized Metrics:")
                    print(f"Annual Return: {annual_return:.4f}")
                    print(f"Annual Volatility: {annual_vol:.4f}")
                    print(f"Annual Sharpe Ratio: {annual_sharpe:.4f}")
            else:
                print("No valid returns to calculate statistics")
        
        # Store portfolio values for further analysis
        portfolio.values = portfolio_values
        
        return portfolio
    #-----------------------------------------------------------------------------------------------
    # load_cv_results: Load cross validation results
    #-----------------------------------------------------------------------------------------------
    def load_cv_results(self, cv_results):
        """Load cross validation results

        Inputs
        cv_results: pd.dataframe containing the cross validation results

        Outputs
        self.lr: Load the optimal learning rate
        self.epochs: Load the optimal number of epochs
        """

        # Store the cross validation results within the object
        self.cv_results = cv_results

        # Select and store the optimal hyperparameters
        idx = cv_results.val_loss.idxmin()
        self.lr = cv_results.lr[idx]
        self.epochs = cv_results.epochs[idx]