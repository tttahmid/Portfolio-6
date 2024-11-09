import os

def iterative_training(data_yaml, initial_weights, max_iterations=10):
    for iteration in range(1, max_iterations + 1):
        print(f"Starting iteration {iteration}...")
        weights = f"iteration_{iteration}_best.pt" if iteration > 1 else initial_weights
        os.system(f"python train.py --img 640 --batch 16 --epochs 100 --data {data_yaml} --weights {weights}")
      
