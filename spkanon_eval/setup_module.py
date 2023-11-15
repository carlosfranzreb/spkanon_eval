"""Read feature processors from config and return them as a dict."""


import importlib

from omegaconf import DictConfig

from spkanon_eval.component_definitions import Component


def setup(config: DictConfig, device: str, **kwargs) -> dict[str, Component]:
    components = dict()
    if "cls" in config:
        return init_component(config, device, **kwargs)
    for name, cfg in config.items():
        if "cls" in cfg:
            components[name] = init_component(cfg, device, **kwargs)
        else:
            components[name] = cfg
    return components


def init_component(config: DictConfig, device: str, **kwargs) -> Component:
    module_str, cls_str = config.cls.rsplit(".", 1)
    module = importlib.import_module(module_str)
    cls = getattr(module, cls_str)
    return cls(config, device, **kwargs)
