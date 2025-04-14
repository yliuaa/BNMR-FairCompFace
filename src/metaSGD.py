from torch.optim.sgd import SGD


class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if "." in name:
            name_split = name.split(".")
            module_name = name_split[0]
            rest_name = ".".join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        group = self.param_groups[0]
        lr = group["lr"]

        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            parameter.detach_()
            grad_n = grad
            self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))
