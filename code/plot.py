import matplotlib.pyplot as plt


def plot_solution(times, solution, figsize=(10,6), ax=None, component_kwargs=None):
 
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if component_kwargs is None:
        component_kwargs = [dict() for _ in range(solution.shape[1])]
    
    for sol, kwargs in zip(solution.T, component_kwargs):
        ax.plot(times, sol, **kwargs)
    
    ax.set_xlabel("time [s]")
    ax.spines[["top", "right"]].set_visible(False)

    return ax


def plot_solution_3d(solution, figsize=(8,8), ax=None, **kwargs):

    if ax is None:

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")

    ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return ax


def plot_scatter_3d(points, figsize=(8,8), ax=None, **kwargs):
    
    if ax is None:
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
    
    ax.scatter(points[:,0], points[:, 1], points[:, 2], **kwargs)
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    
    return ax