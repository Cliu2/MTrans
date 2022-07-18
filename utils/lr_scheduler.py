"""
    Customize optimizer for warmup
"""
from numpy.random.mtrand import gamma
import torch

class WarmupCosineAnnealing:
    def __init__(self, optimizer, max_lr, warmup_rate, T_max, eta_min=0):
        self.warmup_steps = T_max * warmup_rate
        self.current_step=0
        self.cosineannealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max-self.warmup_steps, eta_min)
        self.optimizer = optimizer
        self.max_lr = max_lr

    def step(self):
        if self.current_step < self.warmup_steps:
            # linear wramup
            for g in self.optimizer.param_groups:
                g['lr'] = self.max_lr/self.warmup_steps*self.current_step
        else:
            self.cosineannealing.step()

        self.current_step+=1

    def state_dict(self):
        state_dict = {
            'cosineannealing': self.cosineannealing.state_dict(),
            'current_step': self.current_step,
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.cosineannealing.load_state_dict(state_dict['cosineannealing'])
        self.current_step = state_dict['current_step']


class WarmupExponential:
    def __init__(self, optimizer, max_lr, warmup_rate, epoches, loader_length, eta_min=1e-9, gamma=0.9):
        self.warmup_steps = epoches * loader_length * warmup_rate
        self.current_step=0
        self.exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.eta_min = eta_min
        self.loader_length = loader_length

    def step(self):
        if self.current_step < self.warmup_steps:
            # linear wramup
            for g in self.optimizer.param_groups:
                g['lr'] = self.max_lr/self.warmup_steps*self.current_step
        else:
            if self.current_step % self.loader_length == 0:
                if not self.optimizer.param_groups[0]['lr'] <= self.eta_min:
                    self.exp.step()

        self.current_step+=1

    def state_dict(self):
        state_dict = {
            'exp': self.exp.state_dict(),
            'current_step': self.current_step,
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.exp.load_state_dict(state_dict['exp'])
        self.current_step = state_dict['current_step']

class WarmupStep:
    def __init__(self, optimizer, max_lr, warmup_rate, total_steps, step_size, eta_min=1e-9, gamma=0.5):
        self.warmup_steps = total_steps * warmup_rate
        self.current_step=0
        self.exp = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.eta_min = eta_min

    def step(self):
        if self.current_step < self.warmup_steps:
            # linear wramup
            for g in self.optimizer.param_groups:
                g['lr'] = self.max_lr/self.warmup_steps*self.current_step
        else:
            if not self.optimizer.param_groups[0]['lr'] <= self.eta_min:
                self.exp.step()

        self.current_step+=1

    def state_dict(self):
        state_dict = {
            'exp': self.exp.state_dict(),
            'current_step': self.current_step,
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.exp.load_state_dict(state_dict['exp'])
        self.current_step = state_dict['current_step']