import torch
import time

def main():
    for i in range(10):
        print(f"Iteration {i}")
        time.sleep(0.1)

print("Without profiler:")
main()

print("\nWith profiler:")
with torch.cuda.profiler.profile():
    main()