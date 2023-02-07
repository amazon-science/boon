import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import FNO2d
from ..corrections import (
    dirkernelcorrection2d,
     neukernelcorrection2d,
     perkernelcorrection2d,
    )
from ..operator import BOON_2d

DEFAULT_BDY = {'val':None, 'diff_fn':None}

bdy_correction2d = {
     'dirichlet':dirkernelcorrection2d,
     'periodic': perkernelcorrection2d,
     'neumann':neukernelcorrection2d,
}


# 1D space with time
class BOON_FNO2d(BOON_2d):
    def __init__(
        self,
        width,
        base_no,
        lb = 0,
        ub = 1,
        bdy_type = 'dirichlet'): # dirichlet, periodic, neumann
        super().__init__(width=width, base_no=base_no, lb=lb, ub=ub,
                            bdy_type=bdy_type)
        
        assert isinstance(base_no, FNO2d), (
            'BOON-FNO2d only accepts FNO 2D as base_no')

        self.padding = 9
        self.bdy_type = bdy_type
        
        self.conv_correction0 = bdy_correction2d[bdy_type](self.base_no.conv0)
        self.conv_correction1 = bdy_correction2d[bdy_type](self.base_no.conv1)
        self.conv_correction2 = bdy_correction2d[bdy_type](self.base_no.conv2)
        self.conv_correction3 = bdy_correction2d[bdy_type](self.base_no.conv3)
        
        self.w_correction0 = bdy_correction2d[bdy_type](self.base_no.w0)
        self.w_correction1 = bdy_correction2d[bdy_type](self.base_no.w1)
        self.w_correction2 = bdy_correction2d[bdy_type](self.base_no.w2)
        self.w_correction3 = bdy_correction2d[bdy_type](self.base_no.w3)
        
        
    def forward(self, x, bdy_left=DEFAULT_BDY, 
                    bdy_right=DEFAULT_BDY):
        
        non_bdy = self.get_non_bdy(x, bdy_left, bdy_right)
        grid = self.get_grid(x.shape, x.device)

        bdy_left_padded = bdy_left.copy()
        bdy_right_padded = bdy_right.copy()

        if self.bdy_type != 'periodic':
            bdy_left_padded['val'] = F.pad(bdy_left['val'], [0, self.padding])
            bdy_right_padded['val'] = F.pad(bdy_right['val'], [0, self.padding])

        x = torch.cat((x, grid), dim=-1)     
        x = self.fc0(x)
        
        x = x.permute(0, 3, 1, 2)

        if self.bdy_type != 'periodic':
            x = F.pad(x, [0,self.padding])
                
        x1 = self.conv_correction0(x,bdy_left_padded, bdy_right_padded)
        x2 = self.w_correction0(x,bdy_left_padded, bdy_right_padded)
        x = x1 + x2
        x[:, :, non_bdy, :] = F.gelu(
            x[:, :, non_bdy, :].clone())

        
        x1 = self.conv_correction1(x,bdy_left_padded, bdy_right_padded)
        x2 = self.w_correction1(x,bdy_left_padded, bdy_right_padded)
        x = x1 + x2
        x[:, :, non_bdy, :] = F.gelu(
            x[:, :, non_bdy, :].clone())
        
        
        x1 = self.conv_correction2(x,bdy_left_padded, bdy_right_padded)
        x2 = self.w_correction2(x,bdy_left_padded, bdy_right_padded)
        x = x1 + x2
        x[:, :, non_bdy, :] = F.gelu(
            x[:, :, non_bdy, :].clone())


        x1 = self.conv_correction3(x,bdy_left_padded, bdy_right_padded)
        x2 = self.w_correction3(x,bdy_left_padded, bdy_right_padded)
        x = x1 + x2

        if self.bdy_type != 'periodic':
            x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 1)
        
        x = self.fc1(x)
        x[:, non_bdy, :, :] = F.gelu(
            x[:, non_bdy, :, :].clone())
        
        x = self.fc2(x)
        x = self.strict_enforce_bdy(x, bdy_left, bdy_right)

        return x
