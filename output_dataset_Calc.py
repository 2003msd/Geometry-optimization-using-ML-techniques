import numpy as np
import pandas as pd

def masonry_analysis(row):
    
    Sp = float(row["Span"])
    ts_ratio = float(row["Thickness"])
    tck = ts_ratio * Sp
    wdth = float(row["Width"])  
    UwM = float(row["Unit Weight"])
    n = int(row["Blocks"])
    live_block = int(row["Load Position"]) - 1  
    Wl = 50  

    # Arch ring geometry
    ri = Sp / 2
    re = ri + tck
    rc = (re + ri) / 2
    angblck = np.pi / n
    ArchA = 0.5 * (np.pi * re**2 - np.pi * ri**2)  # Semi-circular arch
    blkA = ArchA / n
    Pi = wdth * blkA * UwM  # weight of each block

    # Block centroids
    xc, yc = np.zeros(n), np.zeros(n)
    for i in range(n):
        theta = angblck * (i + 0.5)
        xc[i] = rc * np.cos(theta)
        yc[i] = rc * np.sin(theta)

    x_A = rc * np.cos(np.pi)
    x_D = rc * np.cos(0)

    min_lambda = float('inf')
    best_B = best_C = -1

    for b in range(1, n - 2):
        for c in range(b + 1, n - 1):
            work_ab = sum(Pi * (xc[i] - x_A) for i in range(0, b + 1))
            work_bc = sum(Pi * (xc[i] - xc[b]) for i in range(b + 1, c + 1))
            work_cd = sum(Pi * (xc[i] - x_D) for i in range(c + 1, n))
            total_internal_work = work_ab + work_bc + work_cd

            if live_block <= b:
                delta_yl = xc[live_block] - x_A
            elif b < live_block <= c:
                delta_yl = xc[live_block] - xc[b]
            else:
                delta_yl = xc[live_block] - x_D

            if delta_yl != 0:
                lambda_val = -total_internal_work / (Wl * delta_yl)
                if lambda_val > 0 and lambda_val < min_lambda:
                    min_lambda = lambda_val
                    best_B, best_C = b, c

    # Prepare result
    result = {
        "Width": wdth,
        "Span": Sp,
        "Thickness": ts_ratio,
        "Unit Weight": UwM,
        "Blocks": n,
        "Load Position": live_block + 1,  # back to 1-based for output
    }

    if best_B != -1:
        
        result.update({
            "Lambda": round(min_lambda, 4),
            "Hinge A": 1,
            "Hinge B": best_B + 1,
            "Live Load Hinge": live_block + 1,
            "Hinge C": best_C + 1,
            "Hinge D": n
        })
        
    return result

def main():
    input_file = "arch_dataset_inputs.csv"
    output_file = "arch_dataset_outputs.csv"

    df = pd.read_csv(input_file)
    results = []

    for _, row in df.iterrows():
        result = masonry_analysis(row)
        results.append(result)

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"✅ Output saved to {output_file}")

if __name__ == "__main__":
    main()
