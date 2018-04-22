import GPyOpt
import Part_1a

# create the object function
f_true = Part_1a.TargetFunction()
f_sim = Part_1a.TargetFunction(sd=0.1)
bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': f_true.bounds[0]},
         {'name': 'var_2', 'type': 'continuous', 'domain': f_true.bounds[1]}]
f_true.plot()

myBopt2D = GPyOpt.methods.BayesianOptimization(f_sim.f, domain=bounds, acquisition_type='MPI', normalize_Y=True)

max_iter = 100  # maximum time 40 iterations
max_time = 100  # maximum time 60 seconds

myBopt2D.run_optimization(max_iter, max_time, verbosity=False)
myBopt2D.plot_acquisition()
myBopt2D.plot_convergence()
print(myBopt2D.x_opt)
print(myBopt2D.fx_opt)


