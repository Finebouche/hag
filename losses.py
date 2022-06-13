import torch
import numpy as np

def mse(target_signal,input_signal):
    """
    rmse(input_signal, target_signal)-> error
    MSE calculation.
    Calculates the mean square error (MSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    target_signal = target_signal.view(target_signal.numel())
    input_signal = input_signal.view(input_signal.numel())

    error = (target_signal - input_signal) ** 2
    return error.mean()

def nmse(target_signal,input_signal):
    """
    nmse(input_signal, target_signal)-> error
    NMSE calculation.
    Calculates the normalized mean square error (NMSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """

    if len(target_signal) == 1:
        raise NotImplementedError('The NRMSE is not defined for signals of length 1 since they have no variance.')

    if isinstance(target_signal, np.ndarray):
        input_signal = torch.from_numpy(input_signal)
        target_signal = torch.from_numpy(target_signal)
    var = torch.std(target_signal)** 2

    return mse(target_signal,input_signal) / var

def nrmse(target_signal,input_signal):
    """
    nrmse(input_signal, target_signal)-> error
    NRMSE calculation.
    Calculates the normalized root mean square error (NRMSE) of the input signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """

    if len(target_signal) == 1:
        raise NotImplementedError('The NRMSE is not defined for signals of length 1 since they have no variance.')

    return torch.sqrt(nmse(target_signal,input_signal))
