def solver_vectorized(var1,var2):
    for j in range(var1.timesteps):
        w = var2.copy()
        var2[1:-1]=w[1:-1]+(var1.calc_CFL*w[2:]-2*w[1:-1]+w[-2])
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