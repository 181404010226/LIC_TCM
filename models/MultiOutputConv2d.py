import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiOutputConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, m, stride=1, padding='same', bias=True):
        super(MultiOutputConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # n
        self.m = m
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Define the weight matrix for fully connected transformation from n*n to m*m
        self.fc = nn.Linear(in_channels * kernel_size * kernel_size, out_channels * m * m, bias=bias)

    def forward(self, x):
        if self.padding == 'same':
            # Calculate padding to maintain the same spatial dimensions
            padding_total = self.kernel_size - 1
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            padding = (padding_left, padding_right, padding_left, padding_right)
            x = F.pad(x, padding)
        elif isinstance(self.padding, int):
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        elif isinstance(self.padding, tuple):
            x = F.pad(x, self.padding)
        else:
            raise ValueError("Padding must be 'same', an integer, or a tuple")

        # Extract sliding local blocks
        unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        # unfold shape: (batch_size, in_channels * kernel_size * kernel_size, L)
        L = unfold.size(-1)

        # Transpose to (batch_size, L, in_channels * kernel_size * kernel_size)
        unfold = unfold.transpose(1, 2)

        # Apply the fully connected layer
        out = self.fc(unfold)
        # out shape: (batch_size, L, out_channels * m * m)

        # Reshape to (batch_size, out_channels, m, m, H, W) assuming 'same' padding
        batch_size, L, out_channels_m2 = out.size()
        H = W = int(L ** 0.5)  # Assuming square input

        out = out.view(batch_size, H, W, self.out_channels, self.m, self.m)
        # Permute to (batch_size, out_channels, H, m, W, m)
        out = out.permute(0, 3, 1, 4, 2, 5)
        # Reshape to (batch_size, out_channels, H*m, W*m)
        out = out.contiguous().view(batch_size, self.out_channels, H * self.m, W * self.m)

        return out
    
class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.multi_conv = MultiOutputConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            m=2,
            stride=1,
            padding='same',
            bias=True
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.multi_conv(x)
        x = self.relu(x)
        return x

# 简化的测试模型
class SimpleTestModel(nn.Module):
    def __init__(self):
        super(SimpleTestModel, self).__init__()
        self.multi_conv = MultiOutputConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            m=2,
            stride=1,
            padding='same',
            bias=False  # 为了简化计算，我们不使用偏置
        )

    def forward(self, x):
        return self.multi_conv(x)

# 测试模型
if __name__ == "__main__":
    # 创建一个 4x4 的序列填充输入张量
    input_tensor = torch.arange(1, 17).float().view(1, 1, 4, 4)
    print("Input tensor:")
    print(input_tensor)

    # 初始化模型
    model = SimpleTestModel()

    # 设置固定的权重以便于验证
    with torch.no_grad():
        weight = torch.arange(1, 37).float().view(4, 9) / 10  # 9 = 3*3 (kernel_size^2), 4 = 2*2 (m^2)
        model.multi_conv.fc.weight.data = weight

    print("\nFully connected layer weights:")
    print(model.multi_conv.fc.weight.data)

    # 运行模型
    output = model(input_tensor)

    print("\nOutput tensor shape:", output.shape)
    print("Output tensor:")
    print(output)

    # 手动计算第一个输出值以供验证
    first_window = input_tensor[0, 0, :3, :3].flatten()
    first_output = torch.matmul(model.multi_conv.fc.weight.data, first_window)
    print("\nManually calculated first 2x2 output:")
    print(first_output.view(2, 2))

        # 手动计算所有滑动窗口的输出
    manual_output = torch.zeros(8, 8)
    padded_input = F.pad(input_tensor.squeeze(), (1, 1, 1, 1))  # 添加same padding

    for i in range(4):
        for j in range(4):
            window = padded_input[i:i+3, j:j+3].flatten()
            window_output = torch.matmul(model.multi_conv.fc.weight.data, window)
            manual_output[2*i:2*i+2, 2*j:2*j+2] = window_output.view(2, 2)

    print("\nManually calculated output:")
    print(manual_output)

    # 验证手动计算和模型输出是否一致
    is_close = torch.isclose(output.squeeze(), manual_output, rtol=1e-4, atol=1e-4)
    print("\nManual calculation matches model output:", is_close.all().item())