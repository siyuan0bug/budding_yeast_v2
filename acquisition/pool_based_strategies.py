"""
Pool-based acquisition strategies.

Maintains a massive candidate pool of parameters and selects from it.
"""
import time
import numpy as np
import wandb

class RandomPoolBased:
    """
    Randomly selects candidates from a predefined massive parameter pool.
    """
    def __init__(self, pool_params, pool_dicts):
        """
        Args:
            pool_params: (Total, 141) numpy array of candidate parameters
            pool_dicts: list of dicts corresponding to pool_params
        """
        self.pool_params = pool_params
        self.pool_dicts = pool_dicts
        self.num_total = len(pool_dicts)
        
        # Keep track of which indices are still available
        self.available_indices = list(range(self.num_total))
        np.random.shuffle(self.available_indices)
        
    def generate(self, model, param_gen, simulator, al_iter, save_dir, batch_size):
        """
        Select batch_size candidates, run ODE simulator, and save them.
        """
        t = time.time()
        
        if len(self.available_indices) < batch_size:
            print(f"Warning: Only {len(self.available_indices)} candidates left in pool. Selecting all remaining.")
            batch_size = len(self.available_indices)
            
        if batch_size == 0:
            return

        # Pop indices from the end of the shuffled list
        selected_indices = [self.available_indices.pop() for _ in range(batch_size)]
        
        sel_dicts = [self.pool_dicts[i] for i in selected_indices]
        sel_params = self.pool_params[selected_indices]
        
        print(f"RandomPoolBased: Selected {batch_size} candidates. Remaining in pool: {len(self.available_indices)}")
        print(f"selection time {time.time() - t:.2f}s")
        wandb.log({"al/al_iter": al_iter, "al/sel_time": time.time() - t, "al/pool_remaining": len(self.available_indices)})
        
        # Run Simulator on selected params
        print(f"Running Simulator on {batch_size} candidates...")
        trajs, valid_indices = simulator.simulate_batch(sel_dicts)
        
        if len(valid_indices) > 0:
            from budding_yeast_v2.data.param_gen import save_ar_batch
            valid_sel_params = sel_params[valid_indices]
            save_ar_batch(trajs, valid_sel_params, save_dir, al_iter, 0)
        
        print(f"RandomPoolBased: Completed generation for iter {al_iter}.")
