"""Read feature processors from config and return them as a dict."""


import sys
import importlib


def setup(config, device, trainer=None, **kwargs):
    components = dict()
    if "cls" in config:
        return init_component(config, device, trainer, **kwargs)
    for name, cfg in config.items():
        if "cls" in cfg:
            components[name] = init_component(cfg, device, trainer, **kwargs)
        else:
            components[name] = cfg
    return components


def init_component(config, device, trainer, **kwargs):
    module_str, cls_str = config.cls.rsplit(".", 1)
    module = importlib.import_module(module_str)
    cls = getattr(module, cls_str)
    if trainer is not None:
        return cls(config, device, trainer, **kwargs)
    else:
        return cls(config, device, **kwargs)
