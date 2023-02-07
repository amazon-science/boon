import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################
#  1d kernel correction
################################################################
DEFAULT_BDY = {'val':None, 'diff_fn':None}


class gaussian_elim_layer1d(nn.Module):
    def __init__(
        self,
        layer,
        tol = 1e-7):
        super().__init__()
        self.layer = layer
        self.tol = tol
    
    
    def forward(self, x):
        raise NotImplementedError
    
    
    def gauss_elimination(self, x, left, right):
        
        e1 = torch.zeros_like(x)
        e1[:, :, 0] = 1
        eN = torch.zeros_like(x)
        eN[:, :, -1] = 1

        K00 = self.layer(e1)[0,0,0]
        KN0 = self.layer(e1)[0,0,-1]
        if torch.abs(K00) < self.tol:K00 = self.tol

        KNN = self.layer(eN)[0,0,-1]
        K0N = self.layer(eN)[0,0,0]
        if torch.abs(KNN) < self.tol:KNN = self.tol

        if left is not None: # first apply left and then right bdy
    #                              opposite could also be done (may result in slight different results)
            Kx = self.layer(x)
            T2_left_x = x.clone()            
            tilde_K00 = K00 - KN0*K0N/KNN
            if torch.abs(tilde_K00) < self.tol:tilde_K00 = self.tol
            T2_left_x[:, :, 0] = 2*x[:, :, 0] - (1/tilde_K00)*(
                                    Kx[:, :, 0] - Kx[:, :, -1]*K0N/KNN 
                                    + K0N*x[:, :, -1])
        else:
            T2_left_x = x

        Ky = self.layer(T2_left_x)

        if right is not None: # left bdy is already corrected, if exists
            T2_right_y = T2_left_x.clone()
            T2_right_y[:, :, -1] = 2*T2_right_y[:, :, -1] - Ky[:, :, -1]/KNN
        else:
            T2_right_y = T2_left_x

        T = self.layer(T2_right_y)
        return T


class gaussian_elim_layer2d(nn.Module):
    def __init__(
        self,
        layer,
        tol = 1e-4):
        super().__init__()
        self.layer = layer
        self.tol = tol

    def forward(self):
        raise NotImplementedError

    def gauss_elimination(self, x, left, right):

        e1 = torch.zeros_like(x)
        eN = torch.zeros_like(x)

        e1[:, :, 0, :] = 1
        eN[:, :, -1, :] = 1

        K00 = self.layer(e1)[0, 0, 0, :]
        KN0 = self.layer(e1)[0, 0, -1, :]
        K0N = self.layer(eN)[0, 0, 0, :]
        KNN = self.layer(eN)[0, 0, -1, :]

        indices_K00_less_tol = torch.nonzero(K00 < self.tol)
        indices_KNN_less_tol = torch.nonzero(KNN < self.tol)

        if len(indices_K00_less_tol) != 0:
            K00[indices_K00_less_tol] = self.tol

        if len(indices_KNN_less_tol) != 0:
            KNN[indices_KNN_less_tol] = self.tol

        if left is not None:  # first apply left and then right bdy
        #                              opposite could also be done (may result in slight different results)
            Kx = self.layer(x)
            T2_left_x = x.clone()

            tilde_K00 = K00 - KN0 * K0N / KNN

            indices_tilde_K00_less_tol = torch.nonzero(tilde_K00 < self.tol)

            if len(indices_tilde_K00_less_tol) != 0:
                tilde_K00[indices_tilde_K00_less_tol] = self.tol

            T2_left_x[:, :, 0, :] = 2 * x[:, :, 0, :] - (1 / tilde_K00) * (
                        Kx[:, :, 0, :] - Kx[:, :, -1, :] * K0N / KNN + K0N * x[:, :, -1, :])

        else:
            T2_left_x = x

        Ky = self.layer(T2_left_x)

        if right is not None: # left bdy is already corrected, if exists
            T2_right_y = T2_left_x.clone()
            T2_right_y[:, :, -1, :] = 2 * T2_right_y[:, :, 0, :] - Ky[:, :, -1, :] / KNN
        else:
            T2_right_y = T2_left_x

        T = self.layer(T2_right_y)
        return T


class gaussian_elim_layer3d(nn.Module):
    def __init__(
        self,
        layer,
        tol = 1e-4):
        super().__init__()
        self.layer = layer
        self.tol = tol
    
    
    def forward(self, x):
        raise NotImplementedError
    
    
    def gauss_elimination(self, x):
        
        e1_x = torch.zeros_like(x)
        e1_x[:, :, 0, :, :] = 1
        eN_x = torch.zeros_like(x)
        eN_x[:, :, -1, :, :] = 1

        K00_x = self.layer(e1_x)[0,0,0,0,:]
        KN0_x = self.layer(e1_x)[0,0,-1,0,:]
        idx_K00_below_tol = torch.nonzero(torch.abs(K00_x)<self.tol)
        if len(idx_K00_below_tol) > 0:
            K00_x[idx_K00_below_tol] = self.tol

        KNN_x = self.layer(eN_x)[0,0,-1,0,:]
        K0N_x = self.layer(eN_x)[0,0,0,0,:]
        idx_KNN_below_tol = torch.nonzero(torch.abs(KNN_x)<self.tol)
        if len(idx_KNN_below_tol)>0:
            KNN_x[idx_KNN_below_tol] = self.tol
            
            
        tilde_K00_x = K00_x - KN0_x*K0N_x/KNN_x
        
        idx_tilde_K00x_below_tol = torch.nonzero(torch.abs(tilde_K00_x)<self.tol)
        if len(idx_tilde_K00x_below_tol)>0:
            tilde_K00_x[idx_tilde_K00x_below_tol] = self.tol
            
        Kx = self.layer(x)
        T2_left_x = x.clone()
        
        T2_left_x[:,:,0,:,:] = 2*x[:,:,0,:,:] - (1/tilde_K00_x)*(
                                    Kx[:,:,0,:,:] - Kx[:,:,-1,:,:]*K0N_x/KNN_x
                                    + K0N_x*x[:,:,-1,:,:])
        
        Ky = self.layer(T2_left_x)
        T2_right_y = T2_left_x.clone()
        
        T2_right_y[:,:,-1,:,:] = 2*T2_left_x[:,:,0,:,:] - Ky[:,:,-1,:,:]/KNN_x

        T = self.layer(T2_right_y)
        return T
        

class dirkernelcorrection1d(gaussian_elim_layer1d):
    def __init__(self, layer):
        super().__init__(layer=layer)
        """
        1D Corrected Layer. It modifies the kernel to enforce a Dirichlet boundary condition.   
        """

    def forward(self, x, left=DEFAULT_BDY, right=DEFAULT_BDY):
        if left['val'] is None and right['val'] is None:  # no bdy correction
            return self.layer(x)

        T = self.gauss_elimination(x, left['val'], right['val'])

        T[..., 0] = left['val']
        T[..., -1] = right['val']

        return T


class dirkernelcorrection2d(gaussian_elim_layer2d):
    def __init__(self, layer):
        super().__init__(layer=layer)
        """
        2D Corrected Layer. It modifies the kernel to enforce a Dirichlet boundary condition.   
        """

    def forward(self, x, left=DEFAULT_BDY, right=DEFAULT_BDY,
                num_smooth: int = 5):  # TODO: update default to no smoothing

        if left['val'] is None and right['val'] is None:  # no bdy correction
            return self.layer(x)

        T = self.gauss_elimination(x, left['val'], right['val'])

        T[:, :, 0, :] = left['val']
        T[:, :, -1, :] = right['val']

        # Apply mollifier for smoothing at the boundary
        T[:, :, 1:num_smooth, :] = (T[:, :, 0:num_smooth - 1, :] + T[:, :, 1:num_smooth, :] + T[:, :,
                                                                                              2:num_smooth + 1,
                                                                                              :]) / 3
        T[:, :, -num_smooth:-1, :] = (T[:, :, -num_smooth - 1:-2, :] + T[:, :, -num_smooth:-1, :] + T[:, :, -num_smooth+1:, :]) / 3
        return T


class dirkernelcorrection3d(gaussian_elim_layer3d):
    def __init__(self, layer):
        super().__init__(layer=layer)
        """
        1D Corrected Layer. It modifies the kernel to enforce a Dirichlet boundary condition.   
        """

    def forward(self, x, left=DEFAULT_BDY, right=DEFAULT_BDY,
                top=DEFAULT_BDY, down=DEFAULT_BDY):
        if (left['val'] is None and right['val'] is None and
                top['val'] is None and down['val'] is None):  # no bdy correction
            return self.layer(x)

        T = self.gauss_elimination(x)

        T[:, :, 0, :, :] = left['val']
        T[:, :, -1, :, :] = right['val']
        T[:, :, :, 0, :] = top['val']
        T[:, :, :, -1, :] = down['val']

        return T


class neukernelcorrection1d(gaussian_elim_layer1d):
    def __init__(self, layer):
        super().__init__(layer=layer)
        """
        1D Corrected Layer. It modifies the kernel to enforce a Neumann boundary condition.   
        """
        
    def forward(self, x , left=DEFAULT_BDY, right=DEFAULT_BDY):

        if left['val'] is None and right['val'] is None: # no bdy correction
            return self.layer(x)
        
        T = self.gauss_elimination(x, None, right['val'])

        inv_c0_l, _diff_l = left['diff_fn'](T)
        inv_c0_r, _diff_r = right['diff_fn'](T)
        
        T[..., 0] = left['val']  * inv_c0_l - _diff_l
        T[...,-1] = right['val'] * inv_c0_r - _diff_r
        
        return T/2  # 0.5 factor has shown to be better learning the model


class neukernelcorrection2d(gaussian_elim_layer2d):
    def __init__(self, layer):
        super().__init__(layer=layer)
        """
        1D Corrected Layer. It modifies the kernel to enforce a Neumann boundary condition.   
        """

    def forward(self, x, left=DEFAULT_BDY, right=DEFAULT_BDY):

        if left['val'] is None and right['val'] is None:  # no bdy correction
            return self.layer(x)

        # TODO: Fix getting OOM error when pass in left['val']
        T = self.gauss_elimination(x, left['val'], right['val'])

        inv_c0_l, _diff_l = left['diff_fn'](T.permute(0,1,3,2))
        inv_c0_r, _diff_r = right['diff_fn'](T.permute(0,1,3,2))

        T[:, :, 0, :] = left['val'] * inv_c0_l - _diff_l
        T[:, :, -1, :] = right['val'] * inv_c0_r - _diff_r

        return T/2  # 0.5 factor has shown to be better learning the model
    
    
class perkernelcorrection1d(nn.Module):
    def __init__(self, layer):
        super().__init__()
        
        """
        1D Corrected Layer. It modifies the kernel to enforce a periodic boundary condition.   
        """
        
        self.layer = layer
        self.alpha = 0.5  # may not need other possible value, alpha+beta=1, alpha, beta >=0
        self.beta = 0.5
        
    def forward(self, x, *args):
        x = self.layer(x)
        bdy_val = self.alpha*x[...,0] + self.beta*x[...,-1]
        x[..., 0] = bdy_val
        x[...,-1] = bdy_val
        
        return x


class perkernelcorrection2d(nn.Module):
    def __init__(self, layer):
        super().__init__()

        """
        2D Corrected Layer. It modifies the kernel to enforce a periodic boundary condition.   
        """

        self.layer = layer
        self.alpha = 0.5  # may not need other possible value, alpha+beta=1, alpha, beta >=0
        self.beta = 0.5

    def forward(self, x, *args):
        x = self.layer(x)
        bdy_val = self.alpha * x[:, :, 0, :] + self.beta * x[:, :, -1, :] # shape (bs, channel_dim, N, T)
        x[:, :, 0, :] = bdy_val
        x[:, :, -1, :] = bdy_val

        return x

