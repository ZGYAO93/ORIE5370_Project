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
def pred_sigma(y_hat,n_asset):
    """
    Convert the lower triangle elements into a symmetric matrix
    We define/assume the output of the pred_layer represents the covariance between assets
    
    INPUTS:
    -------
    y_hat -> torch.Tensor: here y_hat represents the lower triangle part of the covariance matrix ([1,(n+1)n/2])
    n_asset -> int: number of assets
    
    OUTPUTS:
    --------
    sigma: symmetric covariance matrix (n,n)
    """
    n_data = y_hat.size(0)
    L = torch.zeros((n_data, n_asset, n_asset))  # just for one data

    tril_indices = torch.tril_indices(row=n_asset, col=n_asset)
    L[:,tril_indices[0], tril_indices[1]] = y_hat
    # make the diagonal positive
    L[:,range(n_asset), range(n_asset)] = torch.nn.functional.softplus(L[:,range(n_asset), range(n_asset)])
    sigma = torch.bmm(L, L.transpose(1, 2)) + 1e-3 * torch.eye(n_asset)
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
####################################################################################################
# E2E neural network module
####################################################################################################
class e2e_net(nn.Module):
    """End-to-end DRO learning neural net module.
    """
    def __init__(self, n_x, n_y, n_obs, opt_layer='nominal', prisk='p_var', perf_loss='sharpe_loss',
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
        self.n_y = n_y
        self.n_obs = n_obs

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
            self.n_y = n_y  # number of output from neural network should align with number of asset (budget)
            self.pred_layer = nn.Sequential(
                nn.Linear(n_x, 32),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(32, self.n_y),
                nn.Softmax(dim=-1)  # risk budget should sum up to 1
            )
        elif opt_layer == 'min_variance' or opt_layer == 'risk_parity':
            self.model_type = opt_layer
            self.gamma.requires_grad = False
            self.n_y = int(n_y*(n_y+1)/2)
            self.pred_layer = nn.Sequential(
                nn.Linear(n_x, 32),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(32, self.n_y)
                # no softmax layer
            )
        else:
            # Register 'delta' (ambiguity sizing parameter) for DR layer
            if opt_layer == 'hellinger':
                ub = (1 - 1/(n_obs**0.5)) / 2
                lb = (1 - 1/(n_obs**0.5)) / 10
            else:
                ub = (1 - 1/n_obs) / 2
                lb = (1 - 1/n_obs) / 10
            self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(lb, ub))
            self.delta.requires_grad = train_delta
            self.delta_init = self.delta.item()
            self.model_type = 'dro'

        # LAYER: Prediction model
        self.pred_model = pred_model
        if pred_model == 'linear':
            # Linear prediction model
            self.pred_layer = nn.Linear(n_x, n_y)
            self.pred_layer.weight.requires_grad = train_pred
            self.pred_layer.bias.requires_grad = train_pred
        elif pred_model == '2layer':
            # Neural net with 2 hidden layers 
            self.pred_layer = nn.Sequential(nn.Linear(n_x, int(0.5*(n_x+n_y))),
                      nn.ReLU(),
                      nn.Linear(int(0.5*(n_x+n_y)), n_y),
                      nn.ReLU(),
                      nn.Linear(n_y, n_y))
        elif pred_model == '3layer':
            # Neural net with 3 hidden layers 
            self.pred_layer = nn.Sequential(nn.Linear(n_x, int(0.5*(n_x+n_y))),
                      nn.ReLU(),
                      nn.Linear(int(0.5*(n_x+n_y)), int(0.6*(n_x+n_y))),
                      nn.ReLU(),
                      nn.Linear(int(0.6*(n_x+n_y)), n_y),
                      nn.ReLU(),
                      nn.Linear(n_y, n_y))

        # LAYER: Optimization model
        jiayi_models = ['risk_budget','risk_budget_mod','min_variance','risk_parity']
        if opt_layer == 'markowitz_mod':
            self.opt_layer = eval(opt_layer)(n_y, n_obs, eval('rf.'+prisk), variant)
        elif opt_layer in jiayi_models:
            self.opt_layer = eval(opt_layer)(n_y)
        else:
            self.opt_layer = eval(opt_layer)(n_y, n_obs, eval('rf.'+prisk))
          
        
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
    def forward(self, X, Y):
        """Forward pass of the NN module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        # Multiple predictions Y_hat from X
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        ep = Y - Y_hat[:-1]
        y_hat = Y_hat[-1]
        # Optimization solver arguments (from CVXPY for ECOS/SCS solver)
        solver_args = {'solve_method': 'ECOS', 'max_iters': 120, 'abstol': 1e-7}
        #solver_args = {'solve_method': 'SCS', 'eps': 1e-7, 'acceleration_lookback': 5,
        #'max_iters':20000}
        # solver_args = {
        #     'solve_method': 'CLARABEL',
        #     'max_iter': 1000,             # maximum number of iterations
        #     'tol_feas': 1e-7,             # feasibility tolerance
        #     'tol_infeas_abs': 1e-7,       # absolute infeasibility tolerance
        #     'tol_gap_abs': 1e-7           # absolute gap tolerance
        # }
        # Optimize z per scenario
        Y_hat_centered = Y_hat - Y_hat.mean(dim=0)
        cov_matrix = (Y_hat_centered.T @ Y_hat_centered) / (Y_hat.shape[0] - 1)
        #print(cov_matrix)
        # Determine whether nominal or dro model
        if self.model_type == 'nom':
            z_star, = self.opt_layer(ep, y_hat, self.gamma, solver_args=solver_args)
        elif self.model_type == 'dro':
            z_star, = self.opt_layer(ep, y_hat, self.gamma, self.delta, solver_args=solver_args)
        elif self.model_type == 'base_mod':
            z_star, = self.opt_layer(y_hat, solver_args=solver_args)
        elif self.model_type == 'markowitz_mod':
            z_star = self.opt_layer(y_hat, cov_matrix)
        return z_star, y_hat

    #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, train_set, val_set=None, epochs=None, lr=None):
        """Neural net training module
        
        Inputs
        train_set: SlidingWindow object containing features x, realizations y and performance
        realizations y_perf
        val_set: SlidingWindow object containing features x, realizations y and performance
        realizations y_perf
        epochs: Number of training epochs
        lr: learning rate

        Output
        Trained model
        (Optional) val_loss: Validation loss
        """

        # Assign number of epochs and learning rate
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        # Define the optimizer and its parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Number of elements in training set
        n_train = len(train_set)

        # Train the neural network
        for epoch in range(epochs):
                
            # TRAINING: forward + backward pass
            train_loss = 0
            optimizer.zero_grad() 
            for t, (x, y, y_perf) in enumerate(train_set):

                # Forward pass: predict and optimize
                z_star, y_hat = self(x.squeeze(), y.squeeze())

                # Loss function
                if self.pred_loss is None:
                    loss = (1/n_train) * self.perf_loss(z_star, y_perf.squeeze())
                else:
                    loss = (1/n_train) * (self.perf_loss(z_star, y_perf.squeeze()) + 
                    (self.pred_loss_factor/self.n_y) * self.pred_loss(y_hat, y_perf.squeeze()[0]))

                # Backward pass: backpropagation
                loss.backward()

                # Accumulate loss of the fully trained model
                train_loss += loss.item()
        
            # Update parameters
            optimizer.step()

            # Ensure that gamma, delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name=='gamma':
                    param.data.clamp_(0.0001)
                if name=='delta':
                    param.data.clamp_(0.0001)

        # Compute and return the validation loss of the model
        if val_set is not None:

            # Number of elements in validation set
            n_val = len(val_set)

            val_loss = 0
            with torch.no_grad():
                for t, (x, y, y_perf) in enumerate(val_set):

                    # Predict and optimize
                    z_val, y_val = self(x.squeeze(), y.squeeze())
                
                    # Loss function
                    if self.pred_loss_factor is None:
                        loss = (1/n_val) * self.perf_loss(z_val, y_perf.squeeze())
                    else:
                        loss = (1/n_val) * (self.perf_loss(z_val, y_perf.squeeze()) + 
                        (self.pred_loss_factor/self.n_y)*self.pred_loss(y_val, y_perf.squeeze()[0]))
                    
                    # Accumulate loss
                    val_loss += loss.item()

            return val_loss

    #-----------------------------------------------------------------------------------------------
    # net_cv: Cross validation of the e2e neural net for hyperparameter tuning
    #-----------------------------------------------------------------------------------------------
    def net_cv(self, X, Y, lr_list, epoch_list, n_val=4):
        """Neural net cross-validation module

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

                    # Construct training and validation DataLoader objects
                    train_set = DataLoader(pc.SlidingWindow(X_temp.train(), Y_temp.train(), 
                                                            self.n_obs, self.perf_period))
                    val_set = DataLoader(pc.SlidingWindow(X_temp.test(), Y_temp.test(), 
                                                            self.n_obs, self.perf_period))

                    # Reset learnable parameters gamma and delta
                    self.load_state_dict(torch.load(self.init_state_path))

                    if self.pred_model == 'linear':
                        # Initialize the prediction layer weights to OLS regression weights
                        X_train, Y_train = X_temp.train(), Y_temp.train()
                        X_train.insert(0,'ones', 1.0)

                        X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                        Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
                    
                        Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                        Theta = Theta.T
                        del X_train, Y_train

                        with torch.no_grad():
                            self.pred_layer.bias.copy_(Theta[:,0])
                            self.pred_layer.weight.copy_(Theta[:,1:])

                    val_loss = self.net_train(train_set, val_set=val_set, lr=lr, epochs=epochs)
                    val_loss_tot.append(val_loss)

                    print(f"Fold: {n_val-i} / {n_val}, val_loss: {val_loss}")

                # Store results
                results.val_loss.append(np.mean(val_loss_tot))
                results.lr.append(lr)
                results.epochs.append(epochs)
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
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)
        lr: Learning rate for test. If 'None', the optimal learning rate is loaded
        epochs: Number of epochs for test. If 'None', the optimal # of epochs is loaded

        Output 
        self.portfolio: add the backtest results to the e2e_net object
        """

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(Y.test())-Y.n_obs, self.n_y, Y.test().index[Y.n_obs:])

        # Store trained gamma and delta values 
        if self.model_type == 'nom':
            self.gamma_trained = []
        elif self.model_type == 'dro':
            self.gamma_trained = []
            self.delta_trained = []

        # Store the squared L2-norm of the prediction weights and their difference from OLS weights
        if self.pred_model == 'linear':
            self.theta_L2 = []
            self.theta_dist_L2 = []

        # Store initial train/test split
        init_split = Y.split

        # Window size
        win_size = init_split[1] / n_roll

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
            train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), self.n_obs, 
                                                    self.perf_period))
            test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0))

            # Reset learnable parameters gamma and delta
            self.load_state_dict(torch.load(self.init_state_path))

            if self.pred_model == 'linear':
                # Initialize the prediction layer weights to OLS regression weights
                X_train, Y_train = X.train(), Y.train()
                X_train.insert(0,'ones', 1.0)

                X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
            
                Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                Theta = Theta.T
                del X_train, Y_train

                with torch.no_grad():
                    self.pred_layer.bias.copy_(Theta[:,0])
                    self.pred_layer.weight.copy_(Theta[:,1:])

            # Train model using all available data preceding the test window
            self.net_train(train_set, lr=lr, epochs=epochs)

            # Store trained values of gamma and delta
            if self.model_type == 'nom':
                self.gamma_trained.append(self.gamma.item())
            elif self.model_type == 'dro':
                self.gamma_trained.append(self.gamma.item())
                self.delta_trained.append(self.delta.item())

            # Store the squared L2 norm of theta and distance between theta and OLS weights
            if self.pred_model == 'linear':
                theta_L2 = (torch.sum(self.pred_layer.weight**2, axis=()) + 
                            torch.sum(self.pred_layer.bias**2, axis=()))
                theta_dist_L2 = (torch.sum((self.pred_layer.weight - Theta[:,1:])**2, axis=()) + 
                                torch.sum((self.pred_layer.bias - Theta[:,0])**2, axis=()))
                self.theta_L2.append(theta_L2)
                self.theta_dist_L2.append(theta_dist_L2)

            # Test model
            with torch.no_grad():
                for j, (x, y, y_perf) in enumerate(test_set):
                
                    # Predict and optimize
                    z_star, _ = self(x.squeeze(), y.squeeze())

                    # Store portfolio weights and returns for each time step 't'
                    portfolio.weights[t] = z_star.squeeze()
                    portfolio.rets[t] = y_perf.squeeze() @ portfolio.weights[t]
                    t += 1

        # Reset dataset
        X, Y = X.split_update(init_split), Y.split_update(init_split)

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio

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