import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    # Check if x is a scalar (int, float, np.float64, etc.)
    is_scalar = np.isscalar(x)
    
    # Ensure input is a NumPy float array
    x_arr = np.asarray(x, dtype=float)
    
    # Numerically stable calculation using np.where to prevent exp() overflow
    result = np.where(
        x_arr >= 0,
        1.0 / (1.0 + np.exp(-x_arr)),
        np.exp(x_arr) / (1.0 + np.exp(x_arr))
    )
    
    if is_scalar:
        return float(result.item())
    
    return result
    pass