import numpy as np
import pandas as pd
from itertools import product

s_values = np.round(np.arange(2.000, 9.001, 1), 2)
t_s_ratios = np.round(np.arange(0.080, 0.161, 0.02), 3)
d_values = sorted(set([2400, 7850] + list(np.arange(1900, 2501, 200))))
w_values = np.round(np.arange(5.000, 8.001, 1), 1)
n_blocks_values = np.arange(8, 61, 4)

output_file = "arch_dataset_inputs.csv"
chunk_size = 500_000
buffer = []

columns = ["Width", "Span", "Thickness", "Unit Weight", "Blocks", "Load Position"]

with open(output_file, "w") as f:
    f.write(",".join(columns) + "\n")
    
    for i, (s, t_s, d, w, n) in enumerate(product(s_values, t_s_ratios, d_values, w_values, n_blocks_values)):
        t = round(t_s * s, 3)
        for load_pos in range(2, n ):
            buffer.append([w, s, t, d, n, load_pos])
            if len(buffer) >= chunk_size:
                df_chunk = pd.DataFrame(buffer, columns=columns)
                df_chunk.to_csv(f, header=False, index=False)
                buffer = []
    
    if buffer:
        df_chunk = pd.DataFrame(buffer, columns=columns)
        df_chunk.to_csv(f, header=False, index=False)
