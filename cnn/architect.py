import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    def __init__(self, model, criterion, args):
        gpus = [int(i) for i in args.gpu.split(',')]
        self.is_multi_gpu = True if len(gpus) > 1 else False

        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.criterion = criterion
        self.adas = args.adas

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        self.optimizer = torch.optim.Adam(arch_parameters,
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.gumbel = args.gumbel
        self.grad_clip = args.grad_clip

    def _compute_unrolled_model(self, input, target, lr_vector, network_optimizer):
        logits = self.model(input, self.gumbel)
        loss = self.criterion(logits, target)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        arch_theta = _concat(arch_parameters).data

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))
        model_parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        model_params = filter(lambda p: id(p) not in arch_params, model_parameters) 
        model_theta = _concat(model_params).data
        
        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))
        model_parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        model_params = filter(lambda p: id(p) not in arch_params, model_parameters) 
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in model_params).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(model_theta)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))
        model_parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        model_params = list(filter(lambda p: id(p) not in arch_params, model_parameters))         
        # using gumbel-softmax:
        # for unused ops there will be no gradient and this needs to be handled
        if self.gumbel:
            dtheta = _concat([grad_i + self.network_weight_decay * theta_i if grad_i is not None
                              else self.network_weight_decay * theta_i
                              for grad_i, theta_i in
                              zip(torch.autograd.grad(loss, model_params, allow_unused=True), model_params)])
        # not using gumbel-softmax
        else:
            dtheta = _concat([grad_i + self.network_weight_decay * theta_i
                              for grad_i, theta_i in
                              zip(torch.autograd.grad(loss, model_params), model_params)])

        # Adas
        if self.adas:
            iteration_p = 0
            offset_p = 0
            offset_dp = 0
            arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
            arch_params = list(map(id, arch_parameters))
            model_parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
            model_params = filter(lambda p: id(p) not in arch_params, model_parameters) 
            for p in model_params:                      
                p_length = np.prod(p.size())
                lr = lr_vector[iteration_p]
                d_p = moment[offset_p: offset_p + p_length] + \
                      dtheta[offset_dp: offset_dp + p_length]
                model_theta[offset_p: offset_p + p_length].sub_(d_p, alpha=lr)
                offset_p += p_length
                offset_dp += p_length
                iteration_p += 1
        # original DARTS
        else:
            model_theta.sub_(lr_vector, moment + dtheta)
        
        theta = torch.cat([arch_theta, model_theta])
        unrolled_model = self._construct_model_from_theta(theta)
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, lr, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, lr, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)

        # Add gradient clipping for gumbel-softmax because it leads to gradients with high magnitude
        if self.gumbel:
            arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
            torch.nn.utils.clip_grad_norm_(arch_parameters, self.grad_clip)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        logits = self.model(input_valid, self.gumbel)
        loss = self.criterion(logits, target_valid)

        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, lr, network_optimizer):
        # eqn(6)：dαLval(w',α) ，where w' = w − ξ*dwLtrain(w, α)
        # compute w'
        unrolled_model = self._compute_unrolled_model(input_train, target_train, lr,
                                                      network_optimizer)  # unrolled_model: w -> w'
        # compute Lval: validation loss
        logits = unrolled_model(input_valid, self.gumbel)
        unrolled_loss = self.criterion(logits, target_valid)
        unrolled_loss.backward()
        # compute dαLval(w',α)
        unrolled_arch_parameters = unrolled_model.module.arch_parameters() if self.is_multi_gpu else unrolled_model.arch_parameters()
        dalpha = [v.grad for v in unrolled_arch_parameters]  # grad wrt alpha

        # compute dw'Lval(w',α)
        # gumbel-softmax
        unrolled_arch_parameters = unrolled_model.module.arch_parameters() if self.is_multi_gpu else unrolled_model.arch_parameters()
        unrolled_arch_params = list(map(id, unrolled_arch_parameters))
        unrolled_model_parameters = unrolled_model.module.parameters() if self.is_multi_gpu else unrolled_model.parameters()
        unrolled_model_params = filter(lambda p: id(p) not in unrolled_arch_params, unrolled_model_parameters)
        if self.gumbel:
            vector = []
            for v in unrolled_model_params:
                if v.grad is not None:
                    # used operation by Gumbel-softmax
                    vector.append(v.grad.data)
                else:
                    # unused operation by Gumbel-softmax
                    vector.append(torch.zeros_like(v))
        else:
            vector = [v.grad.data for v in unrolled_model_params]
        
        # Adas: use different etas for different w's
        if self.adas:
            for i, p in enumerate(vector):
                p.mul_(lr[i])

        # eqn(8): (dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
        # where w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # eqn(6)-eqn(8): dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
        for g, ig in zip(dalpha, implicit_grads):
            # g.data.sub_(ig.data, alpha=eta)
            g.data.sub_(ig.data)
        # update α
        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        for v, g in zip(arch_parameters, dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.module.new() if self.is_multi_gpu else self.model.new()
        model_dict = self.model.module.state_dict() if self.is_multi_gpu else self.model.state_dict()

        params, offset = {}, 0
        named_parameters = self.model.module.named_parameters() if self.is_multi_gpu else self.model.named_parameters()
        for k, v in named_parameters:
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)

        if self.is_multi_gpu:
            new_state_dict = OrderedDict()
            for k, v in model_dict.items():
                if 'module' not in k:
                    k = 'module.' + k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k] = v
        else:
            new_state_dict = model_dict

        if self.is_multi_gpu:
            model_new = nn.DataParallel(model_new)
            cudnn.benchmark = True

        model_new.load_state_dict(new_state_dict)

        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        
        # eqn(8): dαLtrain(w+,α) 
        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))
        model_parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        model_params = filter(lambda p: id(p) not in arch_params, model_parameters)
        # compute w+ in eqn(8): w+ = w + dw'Lval(w',α) * epsilon
        for p, v in zip(model_params, vector):
            p.data.add_(v, alpha=R)
        logits = self.model(input, self.gumbel)
        loss = self.criterion(logits, target)
        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_p = torch.autograd.grad(loss, arch_parameters)
        
        # eqn(8): dαLtrain(w-,α) 
        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))
        model_parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        model_params = filter(lambda p: id(p) not in arch_params, model_parameters)
        # compute w- in eqn(8): w- = w - dw'Lval(w',α) * epsilon
        for p, v in zip(model_params, vector):
            p.data.sub_(v, alpha=2 * R)
        logits = self.model(input, self.gumbel)
        loss = self.criterion(logits, target)
        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_n = torch.autograd.grad(loss, arch_parameters)

        # recover w back
        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))
        model_parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        model_params = filter(lambda p: id(p) not in arch_params, model_parameters)
        for p, v in zip(model_params, vector):
            p.data.add_(v, alpha=R)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
