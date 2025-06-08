import torch

class BoxInpaintingOperator:
    def __init__(self, box_size=32, box_center=None, noise_sigma=0.0, device="cuda"):
        """Initialize box inpainting operator.
        
        Args:
            box_size (int): Size of square box to mask out
            box_center (tuple, optional): Center coordinates (x,y) of box. If None, uses center of image.
        """
        self.box_size = box_size
        self.box_center = box_center
        self.noise_sigma = noise_sigma
        self.device = device
        self.name = "inpainting"
        
    def degradation(self, x):
        """Apply box mask to image.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Masked image with box region set to 0
        """
        B, C, H, W = x.shape
        
        # Default to center if not specified
        if self.box_center is None:
            self.box_center = (H//2, W//2)
            
        # Calculate box boundaries
        x_min = self.box_center[0] - self.box_size//2
        x_max = x_min + self.box_size
        y_min = self.box_center[1] - self.box_size//2 
        y_max = y_min + self.box_size
        
        # Create mask (1 everywhere except box region)
        mask = torch.ones_like(x)
        mask[:, :, x_min:x_max, y_min:y_max] = 0
        
        # Apply mask
        masked_x = x * mask

        # Add gaussian noise scaled by sigma
        if self.noise_sigma > 0:
            masked_x = masked_x + torch.randn_like(x) * self.noise_sigma
        
        return masked_x

    def degradation_transpose(self, x):
        """Same as degradation for box inpainting"""
        return self.degradation(x)
    
    def get_mask(self, shape):
        """Get the binary mask.
        
        Args:
            shape (tuple): Shape of input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Binary mask (1 everywhere except box region)
        """
        mask = torch.ones(shape)
        
        if self.box_center is None:
            self.box_center = (shape[2]//2, shape[3]//2)
            
        x_min = self.box_center[0] - self.box_size//2
        x_max = x_min + self.box_size
        y_min = self.box_center[1] - self.box_size//2
        y_max = y_min + self.box_size
        
        mask[:, :, x_min:x_max, y_min:y_max] = 0
        
        return mask.to(self.device)


class SuperResolutionOperator:
    """Operator for super-resolution by downsampling images."""
    
    def __init__(self, scale_factor, noise_sigma=0.0, device="cuda"):
        """Initialize super-resolution operator.
        
        Args:
            scale_factor (int): Factor by which to downsample the image
            sigma (float): Standard deviation of Gaussian noise to add
        """
        self.scale_factor = scale_factor
        self.noise_sigma = noise_sigma
        self.device = device
        self.name = "superresolution"
        
    def degradation(self, x):
        """Apply downsampling operation.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Downsampled image
        """
        # Downsample using average pooling
        downsampled = torch.nn.functional.avg_pool2d(x, self.scale_factor)

        # Add gaussian noise scaled by sigma
        if self.noise_sigma > 0:
            downsampled = downsampled + torch.randn_like(downsampled) * self.noise_sigma
            
        return downsampled

    def degradation_transpose(self, x):
        """Upsample back to original size.
        
        Args:
            x (torch.Tensor): Downsampled image tensor
            
        Returns:
            torch.Tensor: Upsampled image
        """
        # Get original size
        B, C, h, w = x.shape
        H, W = h * self.scale_factor, w * self.scale_factor
        
        # Upsample back to original size using nearest neighbor
        upsampled = torch.nn.functional.interpolate(
            x, 
            size=(H, W),
            mode='nearest'
        )
            
        return upsampled
    
    def get_mask(self, shape):
        """Get the binary mask - not applicable for super-resolution.
        
        Args:
            shape (tuple): Shape of input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Tensor of ones since mask concept doesn't apply
        """
        return torch.ones(shape).to(self.device)

class GaussianDeblurOperator:
    """Operator for Gaussian deblurring."""
    
    def __init__(self, kernel_size, sigma, noise_sigma=0.0, device='cuda'):
        """Initialize Gaussian blur operator.
        
        Args:
            kernel_size (int): Size of Gaussian kernel
            sigma (float): Standard deviation of Gaussian kernel
            device (str): Device to place tensors on
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.noise_sigma = noise_sigma
        self.device = device
        self.name = "deblur"
        
        # Create Gaussian kernel
        kernel = self._get_gaussian_kernel2d(kernel_size, sigma)
        self.kernel = kernel.repeat(3, 1, 1, 1).to(device)
        
    def degradation(self, x):
        """Apply Gaussian blur operation.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Blurred image
        """
        # Apply gaussian blur using conv2d
        blurred = torch.nn.functional.conv2d(
            x.to(self.device), 
            self.kernel.to(x.dtype),
            padding=self.kernel_size//2,
            groups=x.shape[1]
        )

        # Add gaussian noise scaled by sigma if specified
        if self.noise_sigma > 0:
            blurred = blurred + torch.randn_like(blurred) * self.noise_sigma
            
        return blurred

    def degradation_transpose(self, x):
        """Same as degradation for Gaussian blur"""
        return x#self.degradation(x)
        
    def get_mask(self, shape):
        """Get the binary mask - not applicable for deblurring.
        
        Args:
            shape (tuple): Shape of input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Tensor of ones since mask concept doesn't apply
        """
        return torch.ones(shape, device=self.device)
        
    def _get_gaussian_kernel2d(self, kernel_size, sigma):
        """Create 2D Gaussian kernel.
        
        Args:
            kernel_size (int): Size of square kernel
            sigma (float): Standard deviation
            
        Returns:
            torch.Tensor: 2D Gaussian kernel
        """
        x = torch.linspace(-kernel_size//2 + 0.5, kernel_size//2 - 0.5, kernel_size)
        x = x.view(1, -1).repeat(kernel_size, 1)
        y = x.t()
        
        kernel = torch.exp(-(x.pow(2) + y.pow(2))/(2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, kernel_size, kernel_size)

class IdentityOperator:
    def __init__(self, noise_sigma=0.0, device='cuda'):
        self.noise_sigma = noise_sigma
        self.device = device
        
    def degradation(self, x):
        return x

    def degradation_transpose(self, x):
        return x
    
    def get_mask(self, shape):
        return torch.ones(shape, device=self.device)


class SamplingConfig:
    def __init__(self):
        self.inverse_problem = "none"
        self.noise_sigma = 0.0
        self.kernel_size = 11
        self.blur_sigma = 5.0
        self.mask_size = 128
        self.scale_factor = 4

def get_inverse_operator(config, device='cuda'):
    """Create inverse operator based on config.

    Args:
        config: Configuration object containing inverse problem settings
        device: Device to place operator on (default: 'cuda')

    Returns:
        InverseOperator: Configured operator based on config.inverse_problem type
    """
    inverse_problem = config.inverse_problem
    noise_sigma = config.noise_sigma

    if inverse_problem == 'deblur':
        operator = GaussianDeblurOperator(
            kernel_size=config.kernel_size,
            sigma=config.blur_sigma,
            noise_sigma=noise_sigma,
            device=device
        )
    elif inverse_problem == 'box_inpaint':
        operator = BoxInpaintingOperator(
            box_size=config.mask_size,
            noise_sigma=noise_sigma,
            device=device
        )
    elif inverse_problem == 'super_resolution':
        operator = SuperResolutionOperator(
            scale_factor=config.scale_factor,
            noise_sigma=noise_sigma,
            device=device
        )
    else:
        raise ValueError(f"Unknown inverse problem type: {inverse_problem}")

    return operator

if __name__ == "__main__":
    # Create config and set params
    config = SamplingConfig()
    config.inverse_problem = "deblur"
    
    # Create operator
    operator = get_inverse_operator(config)
    
    # Test on random image
    test_shape = (1, 3, 256, 256)
    test_img = torch.randn(test_shape).cuda()
    
    # Apply forward and backward ops
    degraded = operator.degradation(test_img)
    restored = operator.degradation_transpose(degraded)
    print(f"Input shape: {test_img.shape}")
    print(f"Degraded shape: {degraded.shape}")
    print(f"Restored shape: {restored.shape}")
    print(f"Kernel size: {operator.kernel_size}")
    print(f"Sigma: {operator.sigma}")
    print(f"Device: {operator.device}")
    
    # Get mask
    mask = operator.get_mask(test_shape)
    print(f"Mask shape: {mask.shape}")