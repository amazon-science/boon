import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import FNO1d
from ..corrections import (
    dirkernelcorrection1d,
    neukernelcorrection1d,
    perkernelcorrection1d,
    )
from ..operator import BOON_1d

DEFAULT_BDY = {'val':None, 'diff_fn':None}

bdy_correction1d = {
    'dirichlet':dirkernelcorrection1d,
    'periodic': perkernelcorrection1d,
    'neumann':neukernelcorrection1d,
}
class BOON_FNO1d(BOON_1d):
    def __init__(
        self,
        width,
        base_no,
        lb = 0,
        ub = 1,
        bdy_type = 'dirichlet'): # dirichlet, neumann, periodic
        super().__init__(width=width, base_no=base_no, lb=lb, ub=ub,
            bdy_type=bdy_type)

        assert isinstance(base_no, FNO1d), (
            'BOON-FNO1d only accepts FNO 1D as base_no')
        
        self.conv_correction0 = bdy_correction1d[bdy_type](self.base_no.conv0)
        self.conv_correction1 = bdy_correction1d[bdy_type](self.base_no.conv1)
        self.conv_correction2 = bdy_correction1d[bdy_type](self.base_no.conv2)
        self.conv_correction3 = bdy_correction1d[bdy_type](self.base_no.conv3)
        
        self.w_correction0 = bdy_correction1d[bdy_type](self.base_no.w0)
        self.w_correction1 = bdy_correction1d[bdy_type](self.base_no.w1)
        self.w_correction2 = bdy_correction1d[bdy_type](self.base_no.w2)
        self.w_correction3 = bdy_correction1d[bdy_type](self.base_no.w3)
        
        
    def forward(self, x, bdy_left=DEFAULT_BDY, 
                    bdy_right=DEFAULT_BDY):
        
        non_bdy = self.get_non_bdy(x, bdy_left, bdy_right)
        grid = self.get_grid(x.shape, x.device)
        
        x = torch.cat((x, grid), dim=-1)     
        x = self.fc0(x)
        
        x = x.permute(0, 2, 1)
                
        x1 = self.conv_correction0(x, bdy_left, bdy_right)
        x2 = self.w_correction0(x, bdy_left, bdy_right)
        x = x1 + x2
        x[:, :, non_bdy] = F.gelu(x[:, :, non_bdy].clone())

        
        x1 = self.conv_correction1(x, bdy_left, bdy_right)
        x2 = self.w_correction1(x, bdy_left, bdy_right)
        x = x1 + x2
        x[:, :, non_bdy] = F.gelu(x[:, :, non_bdy].clone())
        
        
        x1 = self.conv_correction2(x, bdy_left, bdy_right) 
        x2 = self.w_correction2(x, bdy_left, bdy_right)
        x = x1 + x2
        x[:, :, non_bdy] = F.gelu(x[:, :, non_bdy].clone())


        x1 = self.conv_correction3(x, bdy_left, bdy_right)
        x2 = self.w_correction3(x, bdy_left, bdy_right)
        x = x1 + x2
        x = x.permute(0, 2, 1)
        
        x = self.fc1(x)
        x[:, non_bdy, :] = F.gelu(x[:, non_bdy, :].clone())
        
        x = self.fc2(x)
        x = self.strict_enforce_bdy(x, bdy_left, bdy_right)

        return x
