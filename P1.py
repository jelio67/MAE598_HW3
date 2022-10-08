import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

a = np.array([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]])

x1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
x2 = 1 - x1






''' # HW2P2b
def f_x(x):
    x2 = x[0]
    x3 = x[1]
    f = 5 * x2 ** 2 + 12 * x2 * x3 - 8 * x2 + 10 * x3 ** 2 - 14 * x3 - 5 # function to optimize for this problem
    return f


def g_x(x):
    x2 = x[0]
    x3 = x[1]
    g1 = 10 * x2 + 12 * x3 - 8 # gradient index 1 for this problem
    g2 = 12 * x2 + 20 * x3 - 14 # gradient index 2 for this problem
    g = np.array([g1, g2])
    return g

def H_x(x):
    H = np.array([[10, 12], [12, 20]]) # Hessian for this problem
    return H

def phi_alpha(alpha, t, f, g):
    phi = f - t*g.transpose() @ g*alpha # 1st order approx. of function value, can use @ operator instead of np.matmul
    return phi


def Inexact_Line_Search(XN, t, alpha, max_iter):
    iter = 0
    f = f_x(XN)  # compute function value at current solution XN
    g = g_x(XN)  # compute gradient at current solution XN
    f_x_ag = f_x(XN - alpha * g)  # actual function value at step size alpha
    phi = phi_alpha(alpha, t, f, g)  # 1st order approx. of function value at step size alpha

    while f_x_ag > phi and iter < max_iter:
        f_x_ag = f_x(XN - alpha * g)  # actual function value at new alpha
        phi = phi_alpha(alpha, t, f, g)  # 1st order approx. of function value at new alpha

        alpha = 0.5 * alpha # reduce alpha & test again

        iter += 1 # increment iteration number

    return alpha


def Gradient_Descent(X0, t, alpha0, eps, max_iter):
    results = [] # save some results
    XN = X0 # set solution to initial point
    f = f_x(XN)  # compute function value at initial point
    g = g_x(XN)  # compute gradient at initial point
    iter = 0 # set iteration equal to zero

    results.append([iter, f, g[0], g[1], alpha0, np.linalg.norm(g)])  # save initial results

    while np.linalg.norm(g) > eps and iter < max_iter: # while gradient is not close to zero & iterations are below max

        alpha = Inexact_Line_Search(XN, t, alpha0, max_iter) # Armijo line search, pass in alpha0, not alpha from previous iteration

        XN = XN - alpha*g # new solution at step size alpha

        f = f_x(XN)  # compute function value at new solution
        g = g_x(XN)  # compute gradient at new solution

        iter += 1 # increment iteration number

        results.append([iter, f, g[0], g[1], alpha, np.linalg.norm(g)])

    return XN, results

def Convergence(f_list, f_star, X0, method):
    X2_0 = int(X0[0])
    X3_0 = int(X0[1])
    error = np.zeros(len(f_list))
    for i in range(0, len(f_list)):
        error[i] = abs(f_list[i] - f_star)

    plt.figure()
    plt.ylabel(r'|$f_k$-$f^{*}$|')
    plt.yscale('log')
    plt.xlabel('Iteration #, k')
    if method == 'GD':
        plt.plot(error)
        plt.ylim([1e-13, 10])
        plt.xlim([0, 90])
        plt.savefig('.\\P2b_GD_Results\Convergence_X2_0='+str(X2_0)+'_X3_0='+str(X3_0)+'.jpg', bbox_inches='tight', dpi=300)
    elif method == 'NM':
        plt.plot([0, 1], [4.92857,1e-13])
        plt.ylim([1e-13, 10])
        plt.xlim([0, 1])
        plt.savefig('.\\P2b_NM_Results\Convergence_X2_0='+str(X2_0)+'_X3_0='+str(X3_0)+'.jpg', bbox_inches='tight', dpi=300)


def Newton(X0, eps, max_iter):
    results = []  # save some results
    XN = X0  # set solution to initial point
    # f = f_x(XN)  # compute function value at initial point
    # g = g_x(XN)  # compute gradient at initial point
    # H = H_x(XN)  # Hessian of the objective function
    iter = 0  # set iteration equal to zero
    diff = 1

    # results.append([iter, f, diff])  # save initial results

    while diff > eps and iter < max_iter:  # while the solution is converging & iterations are below max

        f = f_x(XN)  # compute function value at new solution
        g = g_x(XN)  # compute gradient at new solution
        H = H_x(XN)

        XNp1 = XN - np.linalg.inv(H) @ g  # new solution

        diff = abs(np.linalg.norm(XN) - np.linalg.norm(XNp1))

        results.append([iter, f, diff])

        iter += 1  # increment iteration number

        XN = XNp1

    return XN, results



# set params
t = 0.5
alpha = 1
eps = 1e-6
max_iter = 1000

# initial guess
x2_0 = 0
x3_0 = 0
X0 = np.array([[x2_0], [x3_0]])

# Calculate Gradient Descent
XN, results = Gradient_Descent(X0, t, alpha, eps, max_iter)
Results = pd.DataFrame(results, columns=['iter', 'f', 'g[0]', 'g[1]', 'alpha', 'norm(g)'])
Results.to_csv('.\\P2b_GD_Results\Debug_Results_X2_0='+str(x2_0)+'_X3_0='+str(x3_0)+'.csv', index=False)

# plot GD convergence
f_list = np.array(Results['f'])
x_star = np.array([[-1/7], [11/14]])
f_star = f_x(x_star)
Convergence(f_list, f_star, X0, 'GD')

# print GD results
Total_Iterations = len(f_list)-1
print(XN)
print('\n'+str(Total_Iterations))

# Calculate Newton's Method
XN, Results_NM = Newton(X0, eps, max_iter)
Results_NM = pd.DataFrame(Results_NM, columns=['iter', 'f', 'diff'])
Results_NM.to_csv('.\\P2b_NM_Results\Debug_Results_X2_0='+str(x2_0)+'_X3_0='+str(x3_0)+'.csv', index=False)

# plot NM convergence
f_list = np.array(Results_NM['f'])
Convergence(f_list, f_star, X0, 'NM')

# print NM results
Iterations = len(Results_NM)-1
print('\n')
print (XN)
print('\n'+str(Iterations))

'''