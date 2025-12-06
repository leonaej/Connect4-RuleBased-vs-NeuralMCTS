

import torch
import sys

print("="*70)
print("CUDA/GPU availability checks")
print("="*70)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA available: {cuda_available}")

if cuda_available:
    print("\n CUDA available")
    print(f"   - CUDA version: {torch.version.cuda}")
    print(f"   - Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"     - Name: {torch.cuda.get_device_name(i)}")
        print(f"     - Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"     - Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Test a simple tensor operation on GPU
    print("\n" + "-"*70)
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(" GPU test successful! Tensor operations work correctly.")
        print(f"   Result shape: {z.shape}, Device: {z.device}")
    except Exception as e:
        print(f" GPU test failed: {e}")
        cuda_available = False
    
    # Test network on GPU
    print("\n" + "-"*70)
    print("Testing AlphaZero network on GPU...")
    try:
        from agents.alphazero.az_network import AZNetwork
        net = AZNetwork().cuda()
        test_input = torch.randn(4, 2, 6, 7).cuda()  # Batch of 4 boards
        policy, value = net(test_input)
        print(" Network test successful!")
        print(f"   Policy shape: {policy.shape}, device: {policy.device}")
        print(f"   Value shape: {value.shape}, device: {value.device}")
    except Exception as e:
        print(f" Network test failed: {e}")
        cuda_available = False

else:
    print("\n CUDA not available.")
   
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if cuda_available:
    print("\nGPU training is NOT available.")
  
else:
    print("\n GPU training is NOT available.")
  
print("\n" + "="*70)