from dataclasses import dataclass, fields
from typing import Type


def link_dataclass(**dataclass_kwargs) -> Type:
    """A decorator to create dataclasses with optional custom __repr__ method."""
    repr_enabled = dataclass_kwargs.pop("repr", True)

    def decorator(cls: Type) -> Type:
        """Decorator function to add custom __repr__ method."""
        cls = dataclass(
            cls, repr=False, **dataclass_kwargs
        )  # Repr is False, if not set or True, it will get overwritten by the custom method anyway

        if repr_enabled:

            def __repr__(self) -> str:
                """Custom __repr__ method for the class."""
                cls_fields = fields(self)
                attrs = [
                    (f.name, getattr(self, f.name))
                    for f in cls_fields
                    if not f.name.startswith("_")
                ]
                attr_str = ", ".join(f"{name}={value!r}" for name, value in attrs)
                return f"{self.__class__.__name__}({attr_str})"

            cls.__repr__ = __repr__

        return cls

    return decorator
