"""
Random acquisition strategy for Active Learning.
"""

import time
from budding_yeast_v2.data.param_gen import save_ar_batch

class RandomSelection:
    def __init__(self, schedule, sim_batch_size):
        self.schedule = schedule
        self.sim_batch_size = sim_batch_size

    def generate(self, prob_model, param_gen, simulator, al_iter, iter_folder):
        """
        Generate new samples randomly and run the simulator.
        """
        num_batches = self.schedule(al_iter)
        print(f"RandomSelection: Generating {num_batches} batches of size {self.sim_batch_size}...")
        
        t0 = time.time()
        for i in range(num_batches):
            p_vec, p_dicts = param_gen.get_pde_params(self.sim_batch_size)
            trajs, valid_indices = simulator.simulate_batch(p_dicts)
            if len(valid_indices) > 0:
                valid_params = p_vec[valid_indices]
                save_ar_batch(trajs, valid_params, iter_folder, al_iter, i)
        
        print(f"RandomSelection: Completed in {time.time() - t0:.2f}s")
