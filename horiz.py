import time
import numpy as np
import pandas as pd
import joblib
import traceback
from math import isfinite

HAS_DEAP = False
HAS_PYSWARM = False
HAS_OPTUNA = False
HAS_TABULATE = False

try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except Exception:
    pass

try:
    from pyswarm import pso
    HAS_PYSWARM = True
except Exception:
    pass

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    pass

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except Exception:
    pass

from scipy.optimize import differential_evolution, dual_annealing, minimize
import tensorflow as tf

MODEL_PATH = "ann_model"
X_SCALER_PATH = "X_scaler_horizontal.pkl"
Y_SCALER_PATH = "y_scaler_horizontal.pkl"
DATASET_PATH = "horizontal_dataset.csv"

print("Loading ANN model and scalers...")
model = tf.keras.models.load_model(MODEL_PATH)
X_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)
print("Loaded model and scalers.")

TH_MIN_GLOBAL = 0.08
TH_MAX_GLOBAL = 2.0
PENALTY = 1e6

def predict_from_ann(n, diameter, thickness, density):
    X = np.array([[n, diameter, thickness, density]])
    Xs = X_scaler.transform(X)
    ys = model.predict(Xs, verbose=0)
    y = y_scaler.inverse_transform(ys)[0]
    alpha, bcon, ccon = float(y[0]), float(y[1]), float(y[2])
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    bcon_clamped = float(np.clip(bcon, 1.0, n))
    ccon_clamped = float(np.clip(ccon, 1.0, n))
    return (alpha, bcon, ccon), (alpha_clamped, bcon_clamped, ccon_clamped)

def is_feasible_clamped(alpha, bcon, ccon, n):
    b_int = int(round(bcon))
    c_int = int(round(ccon))
    if not (0.0 <= alpha <= 1.0):
        return False
    if not (1 <= b_int < c_int <= n):
        return False
    return True

def objective_thickness(thickness, fixed, verbose=False):
    if isinstance(thickness, (list, tuple, np.ndarray)):
        t = float(np.asarray(thickness).ravel()[0])
    else:
        t = float(thickness)
    n, diameter, density = fixed
    if t < TH_MIN_GLOBAL or t > TH_MAX_GLOBAL:
        return PENALTY + max(0.0, (TH_MIN_GLOBAL - t) if t < TH_MIN_GLOBAL else t - TH_MAX_GLOBAL)
    (_, _, _), (alpha_c, bcon_c, ccon_c) = predict_from_ann(n, diameter, t, density)
    feasible = is_feasible_clamped(alpha_c, bcon_c, ccon_c, n)
    if feasible:
        return t
    else:
        viol = 0.0
        if not (0.0 <= alpha_c <= 1.0):
            viol += abs(min(0.0, alpha_c)) + abs(max(0.0, alpha_c - 1.0))
        b_int = int(round(bcon_c)); c_int = int(round(ccon_c))
        if not (1 <= b_int < c_int <= n):
            if b_int < 1: viol += (1 - b_int)
            if c_int > n: viol += (c_int - n)
            if b_int >= c_int: viol += (b_int - c_int + 1)
        return PENALTY + viol * 1e3

def grid_search_min_thickness(fixed, th_min=TH_MIN_GLOBAL, th_max=None, steps=100):
    if th_max is None:
        th_max = TH_MAX_GLOBAL
    thickness_values = np.linspace(th_min, th_max, steps)
    best = None
    for t in thickness_values:
        val = objective_thickness(t, fixed)
        if val < PENALTY:
            best = (t, *predict_from_ann(fixed[0], fixed[1], t, fixed[2])[1])
            break
    if best is None:
        return None
    return best

def random_search_min_thickness(fixed, trials=200, th_min=TH_MIN_GLOBAL, th_max=None):
    if th_max is None:
        th_max = TH_MAX_GLOBAL
    best_t = None
    best_pred = None
    for _ in range(trials):
        t = np.random.uniform(th_min, th_max)
        val = objective_thickness(t, fixed)
        if val < PENALTY:
            if (best_t is None) or (t < best_t):
                best_t = t
                best_pred = predict_from_ann(fixed[0], fixed[1], t, fixed[2])[1]
    if best_t is None:
        return None
    return (best_t, *best_pred)

def simulated_annealing_min_thickness(fixed, th_min=TH_MIN_GLOBAL, th_max=None, maxiter=100):
    if th_max is None:
        th_max = TH_MAX_GLOBAL
    bounds = [(th_min, th_max)]
    try:
        res = dual_annealing(lambda x: objective_thickness(x, fixed), bounds=bounds, maxiter=maxiter)
        t = float(res.x[0])
        if objective_thickness(t, fixed) < PENALTY:
            return (t, *predict_from_ann(fixed[0], fixed[1], t, fixed[2])[1])
        else:
            return None
    except Exception as e:
        print("Simulated Annealing failed:", e)
        return None

def differential_evolution_min_thickness(fixed, th_min=TH_MIN_GLOBAL, th_max=None, maxiter=100):
    if th_max is None:
        th_max = TH_MAX_GLOBAL
    bounds = [(th_min, th_max)]
    try:
        res = differential_evolution(lambda x: objective_thickness(x, fixed), bounds=bounds, maxiter=maxiter, polish=True)
        t = float(res.x[0])
        if objective_thickness(t, fixed) < PENALTY and isfinite(t):
            return (t, *predict_from_ann(fixed[0], fixed[1], t, fixed[2])[1])
        else:
            return None
    except Exception as e:
        print("Differential Evolution failed:", e)
        return None

def nelder_mead_min_thickness(fixed, x0=None, th_min=TH_MIN_GLOBAL, th_max=None):
    if th_max is None:
        th_max = TH_MAX_GLOBAL
    if x0 is None:
        x0 = np.array([(th_min + th_max) / 2.0])
    try:
        res = minimize(lambda x: objective_thickness(x, fixed), x0, method='Nelder-Mead',
                       bounds=[(th_min, th_max)], options={'maxiter': 200})
        t = float(res.x[0])
        if objective_thickness(t, fixed) < PENALTY:
            return (t, *predict_from_ann(fixed[0], fixed[1], t, fixed[2])[1])
        else:
            return None
    except Exception as e:
        print("Nelder-Mead failed:", e)
        return None

def optuna_min_thickness(fixed, trials=50, th_min=TH_MIN_GLOBAL, th_max=None):
    if not HAS_OPTUNA:
        print("Optuna not installed; skipping Bayesian optimization.")
        return None
    if th_max is None:
        th_max = TH_MAX_GLOBAL
    def opt_obj(trial):
        t = trial.suggest_float("th", th_min, th_max)
        return float(objective_thickness(t, fixed))
    study = optuna.create_study(direction="minimize")
    study.optimize(opt_obj, n_trials=trials, show_progress_bar=False)
    best_val = float(study.best_value)
    if best_val < PENALTY:
        t = float(study.best_trial.params["th"])
        return (t, *predict_from_ann(fixed[0], fixed[1], t, fixed[2])[1])
    return None

def ga_min_thickness(fixed, th_min=TH_MIN_GLOBAL, th_max=None, ngen=40, pop_size=40):
    if not HAS_DEAP:
        print("DEAP (GA) not installed; skipping GA (fall back to DE).")
        return differential_evolution_min_thickness(fixed, th_min=th_min, th_max=th_max)
    if th_max is None:
        th_max = TH_MAX_GLOBAL
    lb, ub = th_min, th_max
    try:
        if "FitnessMin" not in creator.__dict__:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMin)
    except Exception:
        pass
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, lb, ub)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    def eval_ind(ind):
        t = float(ind[0])
        val = objective_thickness(t, fixed)
        return ( -1.0*val, ) if val < PENALTY else ( -1.0*(PENALTY + 1e3), )
    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=(ub - lb)/10.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=pop_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=ngen, verbose=False)
    all_vals = [(float(ind[0]), objective_thickness(float(ind[0]), fixed)) for ind in pop]
    all_vals = sorted(all_vals, key=lambda x: x[1])
    best_t, best_val = all_vals[0]
    if best_val < PENALTY:
        return (best_t, *predict_from_ann(fixed[0], fixed[1], best_t, fixed[2])[1])
    return None

def pso_min_thickness(fixed, th_min=TH_MIN_GLOBAL, th_max=None, swarmsize=30, maxiter=80):
    if not HAS_PYSWARM:
        print("pyswarm not installed; skipping PSO (fall back to DE).")
        return differential_evolution_min_thickness(fixed, th_min=th_min, th_max=th_max)
    if th_max is None:
        th_max = TH_MAX_GLOBAL
    lb = [th_min]
    ub = [th_max]
    try:
        t_best, _ = pso(lambda x: objective_thickness(x, fixed), lb, ub, swarmsize=swarmsize, maxiter=maxiter, debug=False)
        t = float(t_best[0])
        if objective_thickness(t, fixed) < PENALTY:
            return (t, *predict_from_ann(fixed[0], fixed[1], t, fixed[2])[1])
    except Exception as e:
        print("PSO failed:", e)
    return None

METHODS = [
    ("Grid Search", grid_search_min_thickness),
    ("Random Search", random_search_min_thickness),
    ("Simulated Annealing", simulated_annealing_min_thickness),
    ("Genetic Algorithm", ga_min_thickness),
    ("Particle Swarm", pso_min_thickness),
    ("Differential Evolution", differential_evolution_min_thickness),
    ("Bayesian Opt (Optuna)", optuna_min_thickness),
    ("Nelder-Mead", nelder_mead_min_thickness),
]

def run_all_methods_for_design(fixed, verbose=False, config=None):
    results = []
    cfg = {
        "grid_steps": 200,
        "random_trials": 300,
        "sa_iter": 200,
        "de_iter": 200,
        "ga_gen": 60,
        "ga_pop": 60,
        "pso_swarm": 40,
        "pso_iter": 80,
        "optuna_trials": 80
    }
    if config:
        cfg.update(config)
    for name, fn in METHODS:
        start = time.time()
        try:
            if name == "Grid Search":
                out = fn(fixed, steps=cfg["grid_steps"])
            elif name == "Random Search":
                out = fn(fixed, trials=cfg["random_trials"])
            elif name == "Simulated Annealing":
                out = fn(fixed, maxiter=cfg["sa_iter"])
            elif name == "Genetic Algorithm":
                out = fn(fixed, ngen=cfg["ga_gen"], pop_size=cfg["ga_pop"])
            elif name == "Particle Swarm":
                out = fn(fixed, swarmsize=cfg["pso_swarm"], maxiter=cfg["pso_iter"])
            elif name == "Differential Evolution":
                out = fn(fixed, maxiter=cfg["de_iter"])
            elif name == "Bayesian Opt (Optuna)":
                out = fn(fixed, trials=cfg["optuna_trials"])
            elif name == "Nelder-Mead":
                out = fn(fixed)
            else:
                out = None
        except Exception as e:
            print(f"Method {name} raised an exception: {e}\n{traceback.format_exc()}")
            out = None
        elapsed = time.time() - start
        if out is None:
            results.append((name, None, None, None, elapsed))
        else:
            t, alpha, bcon, ccon = out
            results.append((name, float(t), float(alpha), float(bcon), float(ccon), elapsed))
    return results

if __name__ == "__main__":
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print("Could not load dataset file:", DATASET_PATH, e)
        df = None
    SAMPLE_K = 5
    if df is not None:
        available = len(df)
        SAMPLE_K = min(SAMPLE_K, available)
        rows = df.sample(SAMPLE_K, random_state=42).reset_index(drop=True)
    else:
        rows = pd.DataFrame([
            {"n": 20, "Diameter (m)": 5.0, "Density (kN/m³)": 25},
            {"n": 30, "Diameter (m)": 3.5, "Density (kN/m³)": 18},
        ])
    all_summary = []
    for idx, row in rows.iterrows():
        n = int(row["n"])
        diameter = float(row["Diameter (m)"])
        density = float(row["Density (kN/m³)"])
        fixed = (n, diameter, density)
        print("\n" + "="*80)
        print(f"Design row {idx+1}: n={n}, Diameter={diameter}, Density={density}")
        print("Running optimizers (this may take time)...")
        results = run_all_methods_for_design(fixed)
        summary_rows = []
        for item in results:
            if item[1] is None:
                summary_rows.append([item[0], "No feasible", "-", "-", "-", f"{item[-1]:.2f}s"])
            else:
                summary_rows.append([item[0], f"{item[1]:.4f}", f"{item[2]:.4f}", f"{item[3]:.0f}", f"{item[4]:.0f}", f"{item[5]:.2f}s"])
                all_summary.append({
                    "design_idx": idx+1, "method": item[0], "thickness": item[1],
                    "alpha": item[2], "bcon": item[3], "ccon": item[4], "time_s": item[5],
                    "n": n, "diameter": diameter, "density": density
                })
        headers = ["Method", "Min Thickness (m)", "Alpha", "Bcon", "Ccon", "Time"]
        if HAS_TABULATE:
            print(tabulate(summary_rows, headers=headers, tablefmt="fancy_grid"))
        else:
