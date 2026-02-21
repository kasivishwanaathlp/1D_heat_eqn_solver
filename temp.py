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
        time_history.append(n*params.dt)

        if residuals < var1.target_residuals:
            print(f"solution converged in {j} of {var1.timesteps} iteration(s)")
            break
    return { "u_history":np.array(u_history), "residuals_history":np.array(residuals_history), "time_history":np.array(time_history)}
