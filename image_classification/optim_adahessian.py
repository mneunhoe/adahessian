#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami, Sheng Shen
# All rights reserved.
# This file is part of AdaHessian library.
#
# AdaHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AdaHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with adahessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import math
import torch
from torch.optim.optimizer import Optimizer
from copy import deepcopy
import numpy as np
import time
import contextlib

@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(gpu_rng_state, device)

def try_update(params, step_size, params_current, search_direction):
        zipped = zip(params, params_current, search_direction)

        for p_next, p_current, search_direction in zipped:
            #p_next.data = p_current - step_size * g_current
            p_next.data = p_current + step_size * search_direction

def check_armijo_conditions(step_size, step_size_old, loss, grad_norm,
                      loss_next, c, beta_b):
    found = 0

    # computing the new break condition
    break_condition = loss_next - \
        (loss - (step_size) * c * grad_norm**2)

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b

    return found, step_size, step_size_old

def check_goldstein_conditions(step_size, loss, grad_norm,
                          loss_next,
                          c, beta_b, beta_f, bound_step_size, eta_max):
	found = 0
	if(loss_next <= (loss - (step_size) * c * grad_norm ** 2)):
		found = 1

	if(loss_next >= (loss - (step_size) * (1 - c) * grad_norm ** 2)):
		if found == 1:
			found = 3 # both conditions are satisfied
		else:
			found = 2 # only the curvature condition is satisfied

	if (found == 0):
		raise ValueError('Error')

	elif (found == 1):
		# step-size might be too small
		step_size = step_size * beta_f
		if bound_step_size:
			step_size = min(step_size, eta_max)

	elif (found == 2):
		# step-size might be too large
		step_size = max(step_size * beta_b, 1e-8)

	return {"found":found, "step_size":step_size}


def reset_step(step_size, n_batches_per_epoch=None, gamma=None, reset_option=1,
               init_step_size=None):
    if reset_option == 0:
        pass

    elif reset_option == 1:
        step_size = step_size * gamma**(1. / n_batches_per_epoch)

    elif reset_option == 2:
        step_size = init_step_size

    return step_size

def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
        p_next.data = p_current - step_size * g_current

def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm

def get_grad_list(params):
    return [p.grad for p in params]

class Adahessian_sls(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1)
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1, line_search = True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power, line_search=line_search)

        super(Adahessian_sls, self).__init__(params, defaults)

    def get_trace(self, gradsH):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        params = self.param_groups[0]['params']

        v = [torch.randint_like(p, high=2, device='cuda') for p in params]
        for v_i in v:
            v_i[v_i == 0] = -1
        hvs = torch.autograd.grad(
            gradsH,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True)

        hutchinson_trace = []
        for hv, vi in zip(hvs, v):
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                tmp_output = torch.abs(hv * vi)
                hutchinson_trace.append(tmp_output) # Hessian diagonal block size is 1 here.
            elif len(param_size) == 4:  # Conv kernel
                tmp_output = torch.abs(torch.sum(torch.abs(
                    hv * vi), dim=[2, 3], keepdim=True)) / vi[0, 1].numel() # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                hutchinson_trace.append(tmp_output)
        
        return hutchinson_trace
    
    
        
    def step(self, gradsH, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        seed = time.time()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure()
        
        loss = None
        if closure is not None:
            loss = closure_deterministic()

        # get the Hessian diagonal
        hut_trace = self.get_trace(gradsH)

        for group in self.param_groups:
            params = group['params']
            p_current = deepcopy(params)
            grad_current = get_grad_list(params)
            grad_norm = compute_grad_norm(grad_current)
            
            for i, p in enumerate(params):
                if p.grad is None:
                    continue
                grad = deepcopy(gradsH[i].data)
                state = self.state[p]
                params_current = p_current[i]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of Hessian diagonal square values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

                beta1, beta2 = group['betas']
                
                lr = group['lr']
                
                weight_decay = group['weight_decay']
                
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(
                    1 - beta2, hut_trace[i], hut_trace[i])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # make the square root, and the Hessian power
                k = group['hessian_power']
                denom = (
                    (exp_hessian_diag_sq.sqrt() ** k) /
                    math.sqrt(bias_correction2) ** k).add_(
                    group['eps'])
                
                if group['line_search']:
                    with torch.no_grad():
                        if grad_norm >= 1e-8:
                            # check if condition is satisfied
                            found = 0
                            step_size_old = lr
                            search_direction = -(exp_avg / bias_correction1 / denom + weight_decay/lr * p.data)
                            for e in range(100):
                                # try a prospective step
                                try_update(p.data, step_size_old, params_current, search_direction)

                                # compute the loss at the next step; no need to compute gradients.
                                loss_next = closure_deterministic()
                                self.state['n_forwards'] += 1

                                # =================================================
                                # Line search
                                if group['line_search_fn'] == "armijo":
                                    armijo_results = check_armijo_conditions(step_size=step_size,
                                                                step_size_old=step_size_old,
                                                                loss=loss,
                                                                grad_norm=grad_norm,
                                                                loss_next=loss_next,
                                                                c=0.25,
                                                                beta_b=0.5)
                                    found, step_size, step_size_old = armijo_results
                                    if found == 1:
                                        lr = step_size
                                        break
                    
                
                # make update
                p.data = p.data - \
                    lr * (exp_avg / bias_correction1 / denom + weight_decay/lr * p.data)

        return loss

class Adahessian_sls2(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1)
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1, line_search = True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power, line_search=line_search)

        super(Adahessian_sls, self).__init__(params, defaults)

    def get_trace(self, gradsH):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        params = self.param_groups[0]['params']

        v = [torch.randint_like(p, high=2, device='cuda') for p in params]
        for v_i in v:
            v_i[v_i == 0] = -1
        hvs = torch.autograd.grad(
            gradsH,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True)

        hutchinson_trace = []
        for hv, vi in zip(hvs, v):
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                tmp_output = torch.abs(hv * vi)
                hutchinson_trace.append(tmp_output) # Hessian diagonal block size is 1 here.
            elif len(param_size) == 4:  # Conv kernel
                tmp_output = torch.abs(torch.sum(torch.abs(
                    hv * vi), dim=[2, 3], keepdim=True)) / vi[0, 1].numel() # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                hutchinson_trace.append(tmp_output)
        
        return hutchinson_trace
    
    
        
    def step(self, gradsH, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        seed = time.time()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure()
        
        loss = None
        if closure is not None:
            loss = closure_deterministic()

        # get the Hessian diagonal
        hut_trace = self.get_trace(gradsH)

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                params = p.data
                params_current = deepcopy(params)
                grad = deepcopy(gradsH[i].data)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of Hessian diagonal square values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

                beta1, beta2 = group['betas']
                
                lr = group['lr']
                
                weight_decay = group['weight_decay']
                
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(
                    1 - beta2, hut_trace[i], hut_trace[i])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # make the square root, and the Hessian power
                k = group['hessian_power']
                denom = (
                    (exp_hessian_diag_sq.sqrt() ** k) /
                    math.sqrt(bias_correction2) ** k).add_(
                    group['eps'])
                
                if group['line_search']:
                    alpha = lr
                    search_direction = -(exp_avg / bias_correction1 / denom + weight_decay/alpha * p.data)
                    
                    line_m = torch.sum(torch.mul(grad, search_direction))
                    
                    prev_loss = loss
                    c = 0.25
                    tau = 0.5
                    search_t = -c*line_m
                    
                    try_update(params, alpha, params_current, search_direction)
                    loss_next = closure_deterministic()
                    ls_it = 1
                    while (prev_loss - loss_next) < (alpha * search_t) and ls_it < 15 :
                            alpha = tau * alpha
                            #search_direction = -(exp_avg / bias_correction1 / denom + weight_decay/alpha * p.data)
                            try_update(params, alpha, params_current, search_direction)
                            loss_next = closure_deterministic()
                            ls_it = ls_it + 1
                    lr = alpha
                
                # make update
                p.data = p.data - \
                    lr * (exp_avg / bias_correction1 / denom + weight_decay/lr * p.data)

        return loss

def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for n, param in model.named_parameters():
        #print(n, param.shape, param.requires_grad, None if param.grad is None else param.grad.shape)
        if not param.requires_grad:# or "decoder.sentence_encoder.embed" in n:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

class Adahessian(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1)
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power)

        super(Adahessian, self).__init__(params, defaults)

    def get_trace(self, gradsH):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        params = self.param_groups[0]['params']

        v = [torch.randint_like(p, high=2, device='cuda') for p in params]
        for v_i in v:
            v_i[v_i == 0] = -1
        hvs = torch.autograd.grad(
            gradsH,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True)

        hutchinson_trace = []
        for hv, vi in zip(hvs, v):
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                tmp_output = torch.abs(hv * vi)
                hutchinson_trace.append(tmp_output) # Hessian diagonal block size is 1 here.
            elif len(param_size) == 4:  # Conv kernel
                tmp_output = torch.abs(torch.sum(torch.abs(
                    hv * vi), dim=[2, 3], keepdim=True)) / vi[0, 1].numel() # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                hutchinson_trace.append(tmp_output)
        
        return hutchinson_trace

    def step(self, gradsH, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # get the Hessian diagonal
        hut_trace = self.get_trace(gradsH)

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = deepcopy(gradsH[i].data)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of Hessian diagonal square values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(
                    1 - beta2, hut_trace[i], hut_trace[i])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # make the square root, and the Hessian power
                k = group['hessian_power']
                denom = (
                    (exp_hessian_diag_sq.sqrt() ** k) /
                    math.sqrt(bias_correction2) ** k).add_(
                    group['eps'])

                # make update
                p.data = p.data - \
                    group['lr'] * (exp_avg / bias_correction1 / denom + group['weight_decay'] * p.data)

        return loss
