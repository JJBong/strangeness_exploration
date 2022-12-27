import pprint
import inspect


def check_parameters_overlapped(*param_classes):
    param_classes_list = [param_class for param_class in param_classes]
    param_dict = {}
    param_overlapped_dict = {}
    for param_class in param_classes_list:
        param_class_name = param_class.__name__.split('.')[-1]
        for param in dir(param_class):
            if not param.startswith("__"):
                if param in param_dict.keys():
                    param_overlapped_dict[param] = '[{} <-- {}]'.format(param_class_name, param_dict[param])
                    param_dict[param] = param_class_name
                else:
                    param_dict[param] = param_class_name
    return param_overlapped_dict


def print_parameters(param_cls_instance, param_overlapped_dict):
    all_members_of_param_cls = inspect.getmembers(param_cls_instance)

    params = {}
    for name, value in all_members_of_param_cls:
        if not (name.startswith("__") or name == 'param_overlapped_dict'):
            if name in param_overlapped_dict.keys():
                params[name] = (value, type(value), param_overlapped_dict[name])
            else:
                params[name] = (value, type(value))

    print()
    print("HYPER-PARAMETERS : ")
    pp = pprint.PrettyPrinter(width=200, indent=2)
    pp.pprint(params)
    print()
