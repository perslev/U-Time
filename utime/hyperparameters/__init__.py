from utime import Defaults
from yamlhparams import YAMLHParams as _YAMLHParams


def check_deprecated_params(hparams):
    pass


class YAMLHParams(_YAMLHParams):
    """
    Wrapper around the yamlhparams.YAMLHParams object to pass the utime package name for VC.
    Also allows to disable VC with no_version_control parameter.
    """
    def __init__(self, yaml_path, no_version_control=False):
        vc = Defaults.PACKAGE_NAME if not no_version_control else None
        super(YAMLHParams, self).__init__(yaml_path,
                                          version_control_package_name=vc,
                                          check_deprecated_params_func=check_deprecated_params)
