import os
import importlib.util


def load_all_modules_from_directory(directory):
    modules = {}
    for filename in os.listdir(directory):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Strip .py from filename to get module name
            file_path = os.path.join(directory, filename)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[module_name] = module
    return modules
