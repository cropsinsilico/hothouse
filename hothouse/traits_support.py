import traitlets


def check_shape(*dimensions):
    r"""Factory to return a valiator to check the shape of an ndarray
    trait.

    Args:
        *dimensions: Arguments are treated as sizes in each dimension.

    Returns:
        callable: Validator that takes a traitlets.TraitType and its
            value as input.

    """

    def validator(trait, value):
        if value is None:
            return value
        if len(value.shape) != len(dimensions):
            raise traitlets.TraitError(
                f"{trait.name}: expected rank {len(dimensions)} but got "
                f"rank {len(value.shape)}"
            )
        for a, b in zip(value.shape, dimensions):
            if b is not None and a != b:
                raise traitlets.TraitError(
                    f"{trait.name}: expected shape of {dimensions} but got "
                    f"shape of {value.shape}"
                )
        return value

    return validator


def check_dtype(dtype):
    r"""Factory to return a valiator to check the shape of an ndarray
    trait.

    Args:
        dtype (np.dtype): Datatype that the validator should check for.

    Returns:
        callable: Validator that takes a traitlets.TraitType and its
            value as input.

    """

    def validator(trait, value):
        if value is None:
            return value
        if value.dtype != dtype:
            raise traitlets.TraitError(
                f"{trait.name}: expected dtype {dtype} but got "
                f"{value.dtype}"
            )
        return value

    return validator
