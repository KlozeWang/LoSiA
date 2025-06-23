import torch
from torch.utils.cpp_extension import load
from typing import Tuple, Optional, List, Literal
from .strategy import Row2Column, Column2Row, PureGreedy

# CUDA extension for efficient two-dimensional index selection
index_select = load(
  name='index_select_dim2',
  sources=["../losia_torch/index_select_dim2.cu"],
  build_directory='.',
  verbose=True,
  extra_cuda_cflags=["--expt-extended-lambda"],
)       

class LoSiAProjector:
    """A class that implements the LoSiA projection method
    for efficient subnet gradient compression and adaptation, along with parameter importance estimation.
    
    Args:
        shape (Tuple[int, int]): Original shape of the parameter matrix
        keys_shape (Tuple[int, int]): Shape of the core subnet
        device: Device to store tensors on (e.g., 'cuda' or 'cpu')
        dtype: Data type for tensors
        
        taylor_type (Literal['param_mix', 'param_second', 'param_first']): 
            Type of Taylor approximation to use for importance measurement
        beta1 (float): Exponential decay rate for average importance I(.)
        beta2 (float): Exponential decay rate for average uncertainties U(.)  
        scale (float): Scaling factor for projected gradients
        layer: Optional layer connection requried only in LoSiA-Pro
    """
    def __init__(
        self,
        shape: Tuple[int, int],
        keys_shape: Tuple[int, int], 
        device, 
        dtype,

        taylor_type: Literal['param_mix', 'param_second', 'param_first'] = 'param_mix', 
        beta1: float = 0.85, 
        beta2: float = 0.85, 
        scale: float = 1.0,
        layer = None
    ):
        self.shape = shape
        self.keys_shape = keys_shape
        self.scale = scale

        self.keys_x = torch.randperm(shape[0])[:keys_shape[0]].to(device)
        torch.sort(self.keys_x)

        self.keys_y = torch.randperm(shape[1])[:keys_shape[1]].to(device)
        torch.sort(self.keys_y)

        self.taylor = taylor_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.ipt = None
        self.exp_avg_ipt = None
        self.exp_avg_unc = None

        self.strategy = [ PureGreedy(self.keys_shape), Row2Column(self.keys_shape), Column2Row(self.keys_shape) ]

        self.device = device
        self.dtype = dtype

        if layer is not None \
            and hasattr(layer, 'setxy') \
            and callable(getattr(layer, 'setxy')):
            layer.setxy(self.keys_x, self.keys_y)

    def project(self, full_grad: torch.Tensor) -> torch.Tensor:
        """Extracts subnet parameter gradients in full gradients 
        based on selected neurons storing in keys.
        
        Args:
            full_grad (torch.Tensor): Original gradient tensor
            
        Returns:
            torch.Tensor: Compressed subnet gradient
        """
        grad_proj = index_select.index_select_dim2(
                        full_grad, 
                        self.keys_x, 
                        self.keys_y, 
                        16
                    )
        grad_proj = grad_proj * self.scale
        return grad_proj

    def project_back(self):
        """Returns the current key indices as a meshgrid.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Meshgrid of x and y indices
        """
        return torch.meshgrid(self.keys_x, self.keys_y, indexing='ij')

    def get_key(self):
        return self.keys_x, self.keys_y

    def record(self, p: torch.Tensor, layer) -> torch.Tensor:
        """Records and updates importance calculation metrics for the given parameter tensor.
        
        Args:
            p (torch.Tensor): Parameter tensor
        """
        if layer is not None \
            and hasattr(layer, 'setxy') \
            and callable(getattr(layer, 'setxy')):
            layer.setxy(None, None)
        
        if self.ipt is None:
            self.ipt = torch.zeros_like(p)
            self.exp_avg_ipt = torch.zeros_like(p) 
            self.exp_avg_unc = torch.zeros_like(p)
        
        with torch.no_grad():
            self.ipt = p * p.grad

            if self.taylor in ['param_second']:
                self.ipt = (self.ipt * self.ipt).abs().detach()
            elif self.taylor in ['param_mix']:
                self.ipt = (self.ipt - 0.5 * self.ipt * self.ipt).abs().detach()
            else:
                self.ipt = self.ipt.abs().detach()

            self.exp_avg_ipt = self.beta1 * self.exp_avg_ipt + \
                                (1 - self.beta1) * self.ipt

            self.exp_avg_unc = self.beta2 * self.exp_avg_unc + \
                                (1 - self.beta2) * (self.ipt - self.exp_avg_ipt).abs()
    
    def reframing(self, layer):
        """Recomputes the optimal key indices based on importance scores.
        
        Args:
            layer: Optional layer to notify about key updates in LoSiA-Pro
        """
        ipt_score = (self.exp_avg_ipt * self.exp_avg_unc).to(dtype=torch.float32)
        max_score = -1e5

        print("reselect")

        for algo in self.strategy:
            rows, cols = algo.importance_locate(ipt_score)
            sum = ipt_score[rows, :][: , cols].sum()
            if sum > max_score:
                max_score = sum
                self.keys_x = rows
                self.keys_y = cols
        
        if layer is not None \
            and hasattr(layer, 'setxy') \
            and callable(getattr(layer, 'setxy')):
            layer.setxy(self.keys_x, self.keys_y)
        
        self.ipt = None
        self.exp_avg_ipt = None
        self.exp_avg_unc = None