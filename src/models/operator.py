import torch
import torch.nn as nn

import numpy as np

DEFAULT_BDY = {'val':None, 'diff_fn':None}


class add_layers(nn.Module):
    def __init__(self, layers):
        super(add_layers, self).__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        return torch.stack(
            [layer(x) for layer in self.layers], dim=-1).sum(dim=-1)


class BOON_1d(nn.Module):
    def __init__(
        self,
        base_no,
        width,
        lb,
        ub,
        bdy_type):
        super().__init__()

        self.base_no = base_no
        self.width = width
        self.lb = lb # lower value of the domain
        self.ub = ub # upper value of the domain
        self.bdy_type = bdy_type

        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    
    
    def get_non_bdy(self, x, left=None, right=None):
        N = x.shape[1]
        non_bdy = torch.arange(N)
        if self.bdy_type == 'periodic':
#             return np.s_[:] # torch autograd giving issue
            return non_bdy
        else: # TODO: Do we need left['val'] and right['val'] here?
            left_interior = 1 if isinstance(left, torch.Tensor) else None
            right_interior = -1 if isinstance(right, torch.Tensor) else None

            return non_bdy[left_interior:right_interior]
    

    def strict_enforce_bdy(self, x, left=None, right=None):
        if self.bdy_type == 'dirichlet':
            if left is not None:
                x[:, 0]=left['val']
            if right is not None:
                x[:,-1]=right['val']
        elif self.bdy_type == 'periodic':
            bdy_val = 0.5*x[:,0,:] + 0.5*x[:,-1,:]
            x[:, 0,:] = bdy_val
            x[:,-1,:] = bdy_val
        elif self.bdy_type == 'neumann':
            if left is not None:
                inv_c0_l, _diff_l = left['diff_fn'](x.permute(0, 2, 1))
                x[:, 0,:] = left['val']*inv_c0_l - _diff_l
            if right is not None:
                inv_c0_r, _diff_r = right['diff_fn'](x.permute(0, 2, 1))
                x[:,-1,:] = right['val']*inv_c0_r - _diff_r
        else:
            raise NotImplementedError
        return x
    
        
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(self.lb, self.ub, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


class BOON_2d(nn.Module):
    def __init__(
            self,
            base_no,
            width,
            lb,
            ub,
            bdy_type):
        super().__init__()

        self.base_no = base_no
        self.width = width
        self.lb = lb
        self.ub = ub
        self.bdy_type = bdy_type

        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def get_non_bdy(self, x, left=None, right=None):
        N = x.shape[1]
        non_bdy = torch.arange(N)
        if self.bdy_type == 'periodic':
            #             return np.s_[:] # torch autograd giving issue
            return non_bdy
        else:
            left_interior = 1 if isinstance(left['val'], torch.Tensor) else None
            right_interior = -1 if isinstance(right['val'], torch.Tensor) else None

            return non_bdy[left_interior:right_interior]

    def strict_enforce_bdy(self, x, left=None, right=None, num_smooth=5):
        if self.bdy_type == 'dirichlet':
            if self.bdy_type == 'dirichlet':
                if left is not None:
                    x[:, 0, :, 0] = left['val'][:, 0, :]
                if right is not None:
                    x[:, -1, :, 0] = right['val'][:, 0, :]
            # Apply mollifier
            # TODO: Add custom smoothing stencil input
            x[:, 1:num_smooth, :] = (x[:, 0:num_smooth - 1, :] + x[:, 1:num_smooth, :] + x[:, 2:num_smooth + 1, :]) / 3
            x[:, -num_smooth:-1, :] = (x[:, -num_smooth - 1:-2, :] + x[:, -num_smooth:-1, :] + x[:, -num_smooth + 1:, :]) / 3
        elif self.bdy_type == 'periodic':
            bdy_val = 0.5 * x[:, 0, :, :] + 0.5 * x[:, -1, :, :]
            x[:, 0, :, :] = bdy_val
            x[:, -1, :, :] = bdy_val
        elif self.bdy_type == 'neumann':
            if left is not None:
                inv_c0_l, _diff_l = left['diff_fn'](x.permute(0, 2, 3, 1))
                x[:, 0, :, 0] = left['val'][:, 0, :] * inv_c0_l - _diff_l.squeeze()
            if right is not None:
                inv_c0_r, _diff_r = right['diff_fn'](x.permute(0, 2, 3, 1))
                x[:, -1, :, 0] = right['val'][:, 0, :] * inv_c0_r - _diff_r.squeeze()
        else:
            raise NotImplementedError
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(self.lb, self.ub, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        # TODO: Time dim may want to pass a separate lb_y, ub_y
        gridy = torch.tensor(np.linspace(self.lb, self.ub, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class BOON_3d(nn.Module):
    def __init__(
        self,
        base_no,
        width,
        lb,
        ub,
        bdy_type):
        super().__init__()

        self.base_no = base_no
        self.width = width
        self.lb = lb
        self.ub = ub
        self.bdy_type = bdy_type

        self.fc0 = nn.Linear(4, self.width) # input channel is 4: (u(x), (x,y,z))
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    
    
    def get_non_bdy(self, x, left=None, right=None, top=None, down=None):
        Nx, Ny = x.shape[1], x.shape[2]
        if self.bdy_type == 'periodic':
#             return np.s_[:] # torch autograd giving issue
            return (0, Nx), (0, Ny)
        else:
            left_interior = 1 if isinstance(left['val'], torch.Tensor) else None
            right_interior = -1 if isinstance(right['val'], torch.Tensor) else None
            top_interior = 1 if isinstance(top['val'], torch.Tensor) else None
            down_interior = -1 if isinstance(down['val'], torch.Tensor) else None

            return (left_interior, right_interior), (top_interior, down_interior)
    

    def strict_enforce_bdy(self, x, left=None, right=None, top=None, down=None):
        if self.bdy_type == 'dirichlet':
            if left['val'] is not None:
                x[:, 0, :, :, 0]=left['val'][:,0,:,:] # squeeze expanded dimension at second axes
            if right['val'] is not None:
                x[:,-1, :, :, 0]=right['val'][:,0,:,:]
            if top['val'] is not None:
                x[:, :, 0, :, 0]=top['val'][:,0,:,:]
            if down['val'] is not None:
                x[:, :,-1, :, 0]=down['val'][:,0,:,:]
        elif self.bdy_type == 'periodic':
            bdy_val_x = 0.5*x[:,0] + 0.5*x[:,-1]
            x[:, 0] = bdy_val_x
            x[:,-1] = bdy_val_x
            bdy_val_y = 0.5*x[:,:,0] + 0.5*x[:,:,-1]
            x[:,:, 0] = bdy_val_y
            x[:,:,-1] = bdy_val_y
            
        elif self.bdy_type == 'neumann':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return x
    
        
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(self.lb, self.ub, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        # TODO: y dim may want to pass a separate lb_y, ub_y and lb_z, ub_z (time)
        gridy = torch.tensor(np.linspace(self.lb, self.ub, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(self.lb, self.ub, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
