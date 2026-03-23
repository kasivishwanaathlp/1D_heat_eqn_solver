#modules
import numbers
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.WARNING, format="WARNING: %(message)s")

#params-INPUT
from config import (config)
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    diffusivity: float
    rod_length: float
    nodes: int
    time: float
    t1: float
    t2: float
    ti: float
    target_CFL: float
    target_residuals: float
    fps: int

    dx: float
    dt: float
    timesteps: int
    calc_CFL: float

@dataclass(frozen=True)
class Results:
    u_history: np.ndarray
    residuals_history: np.ndarray
    time_history: np.ndarray

#rule-tables
INPUT_CHECKS=[
    ("diffusivity", lambda i:isinstance(i,(int,float)) and i>0, "must be >0", "error"),
    ("rod_length", lambda i:isinstance(i,(int,float)) and i>0, "must be >0", "error"),

    ("nodes", lambda i:isinstance(i,int) and i>=2, "must be and integer >=2", "error"),
    ("nodes", lambda i:isinstance(i,int) and i>=10, "Number of nodes less than 10", "warning"),
    ("nodes", lambda i:isinstance(i,int) and i<=1e6, "Number of nodes greater than 1e6", "warning"),

    ("time", lambda i:isinstance(i,(int,float)) and i>0, "must be >0", "error"),
    ("t1", lambda i:isinstance(i,(int,float)), "must be numeric", "error"),
    ("t2", lambda i:isinstance(i,(int,float)), "must be numeric", "error"),
    ("ti", lambda i:isinstance(i,(int,float)), "must be numeric", "error"),
    ("target_CFL", lambda i:isinstance(i,numbers.Real) and 0<i<=0.5, "must be 0< and <=0.5", "error"),

    ("target_residuals", lambda i:isinstance(i,(float)) and 0<i<1, "must be 0< and <1", "error"),
    ("target_residuals", lambda i:isinstance(i,(float)) and i<=1e-2, "convergence might be loose", "warning"),
    ("target_residuals", lambda i:isinstance(i,(float)) and i>=1e-9, "may cause excessive runtime", "warning"),

    ("fps", lambda i:isinstance(i,int) and i>0, "must be an integer >0", "error"),
]
DERIVED_CHECKS=[
    ("dt", lambda i:isinstance(i,(int,float)) and i>0, "computed dt is non-positive", "error"),

    ("timesteps", lambda i:isinstance(i,(int)) and i>=1, "given simulation time smaller than unit timestep", "error"),
    ("timesteps", lambda i:isinstance(i, (int)) and i<=1e7, "calculated no. of timesteps >1e7, simulation time might be excessive", "warning"),
    ("timesteps", lambda i:isinstance(i, (int)) and i>=1e2, "calculated no. of timesteps <1e2, solution may not reach steady behaviour", "warning"),

    ("calc_CFL", lambda i:isinstance(i,numbers.Real) and 0<i<=0.5, "calculated CFL >0.5", "error"),
]

#CHECKS-input
def input_checks(IP_file, rule_table):
    errors=[]
    warnings=[]
    allowed_keys={rule[0] for rule in rule_table}
    for user_keys in IP_file:
        if user_keys not in allowed_keys:
            warnings.append(f"unknown config key: {user_keys}")
    for key,rule,msg,severity in rule_table:
        if key not in IP_file:
            errors.append(f"Unknown config key: {key}")
            continue
        if not rule(IP_file[key]):
            if severity=="error":
                errors.append(f"{key}:{msg}")
            else:
                warnings.append(f"{key}:{msg}")
    for w in warnings:
        logging.warning(w)
    if errors:
        raise ValueError("\nInput validation failed:\n" + "\n".join(errors))

#params-DERIVED
def params_DERIVED(IP_file):
    dx=(IP_file["rod_length"])/((IP_file["nodes"])-1) #m
    dt=(IP_file["target_CFL"] * dx ** 2)/(IP_file["diffusivity"]) #s
    timesteps=int((IP_file['time'])/dt) #-
    calc_CFL=(IP_file["diffusivity"]*dt)/(dx**2)
    return {"dx":dx, "dt":dt, "timesteps":timesteps, "calc_CFL":calc_CFL}

#CHECKS-derived
def derived_checks(IP_dict, rule_table):
    errors=[]
    warnings=[]
    for key,rule,msg,severity in rule_table:
        if key not in IP_dict:
            errors.append(f"Unknown key: {key}")
            continue
        if not rule(IP_dict[key]):
            if severity=="error":
                errors.append(f"{key}:{msg}")
            else:
                warnings.append(f"{key}:{msg}")
    for w in warnings:
        logging.warning(w)
    if errors:
        raise ValueError("\nDerived parameters validation failed:\n" + "\n".join(errors))

#pre-init
def assemble_params(IP_file):
    input_checks(IP_file, INPUT_CHECKS)
    IP_file_derived=params_DERIVED(IP_file)
    derived_checks(IP_file_derived,DERIVED_CHECKS)
    return Params(**IP_file, **IP_file_derived)
params = assemble_params(config)

#init
def init(params):
    u=np.full(params.nodes, params.ti, dtype=float)
    u[0]=params.t1
    u[-1]=params.t2
    return u

#solver
def solver_loop(var1):
    u = init(var1)

    u_history=[]
    residuals_history=[]
    time_history=[]

    for j in range(var1.timesteps):
        w = u.copy()
        for i in range(1, var1.nodes - 1):
            u[i] = w[i] + (var1.calc_CFL * (w[i + 1] - 2 * w[i] + w[i - 1]))
        residuals = np.max(np.abs(w - u))

        u_history.append(u.copy())
        residuals_history.append(residuals)
        time_history.append(j*params.dt)

        if residuals < var1.target_residuals:
            print(f"solution converged in {j} of {var1.timesteps} iteration(s)")
            break

    return Results(
        u_history=np.array(u_history),
        residuals_history=np.array(residuals_history),
        time_history=np.array(time_history),
    )
results_loop = solver_loop(params)

def solver_vectorized(var1,var2):
    for j in range(var1.timesteps):
        w = var2.copy()
        var2[1:-1]=w[1:-1]+var1.calc_CFL*(w[2:]-2*w[1:-1]+w[:-2])
        var2[0]=var1.t1
        var2[-1]=var1.t2
        residuals = np.max(np.abs(w - var2))
        print(residuals)
        ##TODO: ADD RESIDUALS PLOT HERE!!!

        img.set_data(var2[np.newaxis, :])
        # plt.pause(1/fps)

        if residuals < var1.target_residuals:
            print(f"solution converged in {j} of {var1.timesteps} iteration(s)")
            break
    plt.show()
#results_vec=solver_vectorized(params)

#exports
def exports(resutls_loop, pramas, filename="solution.txt"):
    x = np.linspace(0, params.rod_length, params.nodes)
    u_final = results_loop.u_history[-1]
    u_anal = (params.t1) + (params.t2 - params.t1) * (x / params.rod_length)
    data = np.column_stack((x, u_final, u_anal))
    np.savetxt(filename, data, fmt="%.6f", delimiter="\t", header="x\tu_numerical\tu_analytical")
    print(f"Solution exported to {filename}")
exports(results_loop, params)

#viz
def viz(results_loop, params):
    x=np.linspace(0,params.rod_length,params.nodes)
    u_final=results_loop.u_history[-1]
    u_anal=(params.t1)+(params.t2-params.t1)*(x/params.rod_length)

    errors=u_final-u_anal
    l_inf=np.max(np.abs(errors))
    l2=np.sqrt(np.mean(errors**2))
    ref = np.max(np.abs(u_anal))
    percent_error = (l_inf / ref) * 100
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # ---- Temperature Profile ----
    axs[0].plot(x, u_final, label="Numerical", color="#CE31BE")
    axs[0].plot(x, u_anal, "--", label="Analytical", color="#31CE41")
    error_label = (
        f"L∞ = {l_inf:.2e}, "
        f"L2 = {l2:.2e}, "
        f"% = {percent_error:.3f}%"
    )
    axs[0].plot([], [], ' ', label=error_label)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("Temperature")
    axs[0].set_title("Final Temperature Profile")
    axs[0].legend(loc="best")
    axs[0].grid(True)
    # ---- Residual Plot ----
    axs[1].plot(results_loop.time_history, results_loop.residuals_history, color="#CE31BE")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Residual")
    axs[1].set_title("Residual vs Time")
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()
    return l_inf, l2
viz(results_loop, params)

#main-fn
## TODO: add "if __name__ == "__main__":" part for imports
## TODO: "stress-test"
#wiki
## TODO: compare loop and vectorized
## TODO: look for top 3 tangible differences between loop and vectorized and how to documnet them