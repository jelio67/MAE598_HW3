import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def Calc_P(x1, x2, p_w, p_d, A):
    p_est = x1 * torch.exp(A[0] * (A[1] * x2 / (A[0] * x1 + A[1] * x2)) ** 2) * p_w + x2 * torch.exp(A[1] * (A[0] * x1 / (A[0] * x1 + A[1] * x2)) ** 2) * p_d  # p estimation
    return p_est


def Plot_Results(p, p_est, x1):
    x1 = x1.detach().numpy()
    p = p.detach().numpy()
    p_est = p_est.detach().numpy()

    plt.figure()
    plt.plot(x1, p, label='Exact P')
    plt.plot(x1, p_est, label='Approx. P')
    plt.xlim([0,1])
    plt.xlabel('x1')
    plt.ylabel('P')
    plt.legend()
    plt.savefig('P1-3.jpg', dpi=300, bbox_inches='tight')
    plt.show()


a = np.array([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]]) # initialize "a" matrix

x1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) # set x1 vector as numpy array
x2 = 1 - x1  # satisfy x1 + x2 = 1 (& initialize x2 as numpy array)

x1 = torch.tensor(x1, requires_grad=False, dtype=torch.float32) # convert x1 vector into a pytorch tensor, requires_grad is false since we will not have to diff wrt this var
x2 = torch.tensor(x2, requires_grad=False, dtype=torch.float32) # convert x2 vector into a pytorch tensor

T = 20 # degC

p_w = 10**(a[0, 0] - a[0, 1]/(T + a[0, 2])) # p_sat for water, solved Antoine equation
p_d = 10**(a[1, 0] - a[1, 1]/(T + a[1, 2])) # p_sat for dioxane, solved Antoine equation

p_sol = np.array([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]) # solution vector
p_sol = torch.tensor(p_sol, requires_grad=False, dtype=torch.float32) # convert solution vector into pytorch tensor

A = Variable(torch.tensor([1.0, 1.0]), requires_grad=True) # initial guess of solution, requires_grad is true since we will diff p_est wrt this var

alpha = 0.0001
eps = 0.001

p_est = Calc_P(x1, x2, p_w, p_d, A)
loss = (p_est - p_sol)**2
loss = loss.sum()
loss.backward()
AGRAD = float(torch.norm(A.grad))
iter = 0

while AGRAD >= eps:
    p_est = Calc_P(x1, x2, p_w, p_d, A)

    loss = (p_est - p_sol)**2 # calculate squared error
    loss = loss.sum() # sum the squared error

    loss.backward() # calculates gradient

    with torch.no_grad():
        A -= alpha*A.grad # gradient descent step
        AGRAD = float(torch.norm(A.grad))

        A.grad.zero_() # zero out grad so a new grad is not added too it when backward() is called next

    iter += 1


# display results for part 2!
print('estimation of A_12 & A_21 is:', A)
print('final loss is:', loss.data.numpy())
print('total iterations:', iter)

# Plot results for part 3!
Plot_Results(p_sol, Calc_P(x1, x2, p_w, p_d, A), x1)








