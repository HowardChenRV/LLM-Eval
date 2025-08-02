from typing import Any, Type


class Registry:
    """A simple registry for storing and retrieving classes by name."""

    def __init__(self):
        self._registry = {}

    def register(self, name: str, cls: Type) -> Type:
        """
        Register a class with a given name.

        Args:
            name (str): The name to associate with the class.
            cls (Type): The class to register.

        Returns:
            Type: The registered class.
        """
        if name in self._registry:
            print(f"Warning: '{name}' is already registered. Overwriting the existing class.")
        self._registry[name] = cls
        return cls

    def get_class(self, name: str) -> Type:
        """
        Retrieve a registered class by name.

        Args:
            name (str): The name of the class to retrieve.

        Returns:
            Type: The class associated with the given name.

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in self._registry:
            raise KeyError(f"Class '{name}' is not registered in the registry.")
        return self._registry[name]

    def all_classes(self) -> list[str]:
        """Return a list of all registered class names."""
        return list(self._registry.keys())
    
    def __call__(self, name: str) -> Any:
        """Allow the registry instance to be called to get a class by name."""
        return self.get_class(name)


# Create registries for different purposes
dataset_registry = Registry()
api_registry = Registry()


def register_dataset(name: str):
    """
    Decorator to register a dataset class.

    Args:
        name (str): The name to associate with the dataset class.
    """
    def class_decorator(cls: Type) -> Type:
        dataset_registry.register(name, cls)
        return cls
    return class_decorator


def register_api(name: str):
    """
    Decorator to register an api class.

    Args:
        name (str): The name to associate with the api class.
    """
    def class_decorator(cls: Type) -> Type:
        api_registry.register(name, cls)
        return cls
    return class_decorator

