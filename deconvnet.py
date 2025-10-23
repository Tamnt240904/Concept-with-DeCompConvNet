import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

class DeconvNet:
    """
    Implementation của thuật toán DeconvNet để visualize 
    các pixel input ảnh hưởng đến activation của một channel cụ thể
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: Pretrained AlexNet model
            target_layer: Tên của layer muốn visualize (vd: 'features.6' cho conv3)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        
        # Lưu activations, gradients và switches
        self.activations = {}
        self.feature_maps = {}  # Lưu feature maps trước mỗi layer
        self.switches = {}  # Lưu vị trí max pooling indices
        self.layer_outputs = []  # Lưu output của mỗi layer theo thứ tự
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks để lưu activations và switches"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
                module.register_forward_hook(self._save_output(name))
                
    def _save_output(self, name):
        """Hook để lưu output và switches của mỗi layer"""
        def hook(module, input, output):
            # Lưu input vào layer này (output của layer trước)
            self.feature_maps[name] = input[0].detach()
            
            # Lưu output
            if isinstance(module, nn.MaxPool2d):
                # Với MaxPool2d, tính toán indices để lưu switches
                # Sử dụng max_pool2d_with_indices để lấy cả output và indices
                pooled_output, indices = nn.functional.max_pool2d(
                    input[0],
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    return_indices=True
                )
                self.switches[name] = indices
                self.activations[name] = pooled_output
            else:
                self.activations[name] = output.detach()
            
            # Lưu info về layer này
            self.layer_outputs.append({
                'name': name,
                'type': type(module).__name__,
                'output_shape': output.shape
            })
            
            # Check nếu đây là target layer
            if name == self.target_layer:
                return output
                
        return hook
    
    def _deconv_relu(self, x):
        """Deconvolution qua ReLU: chỉ giữ các giá trị dương"""
        return torch.clamp(x, min=0)
    
    def _deconv_pool_with_switches(self, x, switches, output_size):
        """
        Unpooling sử dụng switches: đặt giá trị vào đúng vị trí 
        đã được chọn trong max pooling, các vị trí khác = 0
        
        Args:
            x: Input tensor cần unpool
            switches: Indices từ max pooling
            output_size: Size của output sau unpooling
        """
        # Sử dụng max_unpool2d với switches
        unpooled = nn.functional.max_unpool2d(
            x,
            switches,
            kernel_size=3,  # Sẽ được update từ layer info
            stride=2,
            output_size=output_size
        )
        return unpooled
    
    def _deconv_conv(self, x, conv_layer):
        """
        Deconvolution qua conv layer: sử dụng transposed convolution
        với cùng weights
        """
        # Lấy parameters từ conv layer
        weight = conv_layer.weight
        bias = None
        stride = conv_layer.stride
        padding = conv_layer.padding
        
        # Transpose convolution
        deconv = nn.functional.conv_transpose2d(
            x,
            weight,
            bias=bias,
            stride=stride,
            padding=padding
        )
        return deconv
    
    def visualize_channel(self, input_image, channel_idx):
        """
        Visualize pixel nào ảnh hưởng đến channel cụ thể
        
        Args:
            input_image: Input tensor [1, 3, H, W]
            channel_idx: Index của channel muốn visualize
            
        Returns:
            reconstruction: Reconstructed input showing influential pixels
        """
        # Reset saved data
        self.activations = {}
        self.feature_maps = {}
        self.switches = {}
        self.layer_outputs = []
        
        # Forward pass
        with torch.no_grad():
            self.model(input_image)
        
        # Lấy activation của target layer
        target_activation = self.activations[self.target_layer]
        
        # Tạo mask: chỉ giữ 1 channel, zero out các channel khác
        mask = torch.zeros_like(target_activation)
        mask[0, channel_idx, :, :] = target_activation[0, channel_idx, :, :]
        
        # Backward reconstruction qua các layers
        reconstruction = self._reconstruct(mask, input_image.shape[-2:])
        
        return reconstruction
    
    def _reconstruct(self, activation_map, target_size):
        """
        Reconstruct từ activation map về input space
        Đi ngược qua các layers của network
        """
        current = activation_map
        
        # Lấy danh sách các layers từ input đến target layer
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
                layers.append((name, module))
                if name == self.target_layer:
                    break
        
        # Đi ngược từ target layer về input
        for i, (name, layer) in enumerate(reversed(layers)):
            if isinstance(layer, nn.ReLU):
                # Deconv qua ReLU
                current = self._deconv_relu(current)
                
            elif isinstance(layer, nn.Conv2d):
                # Deconv qua Conv
                current = self._deconv_conv(current, layer)
                
            elif isinstance(layer, nn.MaxPool2d):
                # Unpool sử dụng switches
                if name in self.switches:
                    switches = self.switches[name]
                    # Lấy size của feature map trước pooling
                    if name in self.feature_maps:
                        output_size = self.feature_maps[name].shape[-2:]
                    else:
                        # Estimate nếu không có
                        h, w = current.shape[-2:]
                        stride_h = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
                        stride_w = layer.stride if isinstance(layer.stride, int) else layer.stride[1]
                        output_size = (h * stride_h, w * stride_w)
                    
                    # Unpool với switches
                    kernel_size = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
                    stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
                    
                    current = nn.functional.max_unpool2d(
                        current,
                        switches,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=layer.padding,
                        output_size=output_size
                    )
        
        # Resize về target size nếu cần
        if current.shape[-2:] != target_size:
            current = nn.functional.interpolate(
                current, 
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        return current
    
    def visualize_top_activations(self, input_image, channel_idx, top_k=9):
        """
        Visualize top-k vị trí có activation mạnh nhất
        
        Args:
            input_image: Input tensor
            channel_idx: Channel index
            top_k: Số lượng activation mạnh nhất để visualize
        """
        # Reset và forward pass
        self.activations = {}
        self.feature_maps = {}
        self.switches = {}
        self.layer_outputs = []
        
        with torch.no_grad():
            self.model(input_image)
        
        target_activation = self.activations[self.target_layer]
        
        # Lấy channel cụ thể
        channel_activation = target_activation[0, channel_idx, :, :]
        
        # Tìm top-k activations
        flat_activation = channel_activation.flatten()
        top_values, top_indices = torch.topk(flat_activation, top_k)
        
        # Convert indices về 2D coordinates
        h, w = channel_activation.shape
        top_positions = [(idx.item() // w, idx.item() % w) for idx in top_indices]
        
        # Visualize từng position
        reconstructions = []
        for i, (y, x) in enumerate(top_positions):
            # Reset switches cho mỗi reconstruction
            saved_switches = {k: v.clone() for k, v in self.switches.items()}
            
            mask = torch.zeros_like(target_activation)
            mask[0, channel_idx, y, x] = target_activation[0, channel_idx, y, x]
            recon = self._reconstruct(mask, input_image.shape[-2:])
            reconstructions.append(recon)
            
            # Restore switches
            self.switches = saved_switches
        
        return reconstructions, top_positions, top_values


# Example usage
def main():
    # Load pretrained AlexNet
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()
    
    # Tạo DeconvNet visualizer cho conv3 (features.6 trong AlexNet)
    deconv = DeconvNet(alexnet, target_layer='features.6')
    
    # Load và preprocess image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Giả sử bạn có một image
    img = Image.open('bird.png').convert('RGB')  # Convert to RGB here
    input_tensor = transform(img).unsqueeze(0)
    
    # Tạo random input để demo
    # input_tensor = torch.randn(1, 3, 224, 224)
    
    # Visualize top-9 activations của channel 20
    top_recons, positions, values = deconv.visualize_top_activations(
        input_tensor, 
        channel_idx=20, 
        top_k=9
    )
    
    # Plot results - Top 9 activations
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Top-9 Activations của Channel 20 (Conv3)', fontsize=16)
    
    for idx, (recon, pos, val) in enumerate(zip(top_recons, positions, values)):
        ax = axes[idx // 3, idx % 3]
        
        # Convert tensor to numpy for visualization
        img_np = recon[0].permute(1, 2, 0).detach().numpy()
        # Normalize để hiển thị
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        ax.imshow(img_np)
        ax.set_title(f'Rank #{idx+1}\nPos: {pos}\nActivation: {val:.3f}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('deconvnet_top9_activations.png', dpi=150, bbox_inches='tight')
    print("Top-9 activations visualization saved!")
    
    # Optional: Visualize toàn bộ channel (tổng hợp tất cả activations)
    print("\nVisualizing entire channel...")
    full_reconstruction = deconv.visualize_channel(input_tensor, channel_idx=20)
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    img_full = full_reconstruction[0].permute(1, 2, 0).detach().numpy()
    img_full = (img_full - img_full.min()) / (img_full.max() - img_full.min() + 1e-8)
    ax2.imshow(img_full)
    ax2.set_title('Full Channel 20 Reconstruction (tất cả activations)')
    ax2.axis('off')
    plt.savefig('deconvnet_full_channel.png', dpi=150, bbox_inches='tight')
    print("Full channel reconstruction saved!")


if __name__ == '__main__':
    main()