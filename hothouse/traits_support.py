import traitlets
import warnings


class DependentTraitMixin:
    r"""Mixin for dependent traits.

    Args:
        depends_on (list): Name(s) of traits that the updated trait
            depends on to calculate the default.
        *args: Additional arguments will be passed to the parent class.
        strict (bool, list, optional): Set of dependencies that should
            force setting the trait to the default even when it was
            previously explicitly set. If True, a change to any of the
            dependencies will force the update.
        change_type (str, list, optional): Change(s) for which the
            named trait should be updated.
        **kwargs: Additional keyword arguments are passed to the parent
            class.

    """

    def __init__(self, depends_on, *args, strict=False,
                 change_type=["change", "clear"], **kwargs):
        self.depends_on = depends_on
        self.change_type = (
            change_type if isinstance(change_type, list) else change_type
        )
        if strict is True:
            strict = depends_on
        elif strict is False:
            strict = []
        self.strict = strict
        super().__init__(*args, **kwargs)

    def _get_circular_root(self, change):
        if change['name'] == self.trait_name:
            return change
        if 'trigger' in change:
            return self._get_circular_root(change['trigger'])
        return None

    def recalculate(self, change):
        r"""Callback method to recalculate the trait value after an
        update to one of the traits it depends on.

        Args:
            change (dict): Metadata about the change being made.

        """
        inst = change['owner']
        if not inst.trait_has_value(self.trait_name):
            # TODO: Force calculation if there are validators?
            # No need to recalculate
            return False
        default_set = inst.trait_metadata(self.trait_name, "default_set")
        if default_set is None:
            # None indicates the value was set explicitly, but the
            #   set_default callback has not yet been called (due to
            #   order of traits set during initialization)
            return False
        change_root = self._get_circular_root(change)
        if default_set is False:
            # False indicates the value was set by the user and should
            #   not be recalculated
            if change['name'] not in self.strict:
                # if change['type'] == 'change':
                #     warnings.warn(
                #         f"Update to \"{change['name']}\" "
                #         f"will not update the existing value "
                #         f"for \"{self.trait_name}\" as it was "
                #         f"explicitly set")
                return False
        if ((change_root is not None
             and change_root['type'] == 'change')):
            return False
        old_value = inst._trait_values.pop(self.trait_name)
        inst.notify_change(
            traitlets.Bunch(
                name=self.trait_name,
                old=old_value,
                owner=inst,
                type="clear",
                trigger=change,
            )
        )
        return True

    def set_default(self, change):
        r"""Callback method to set a metadata flag for the trait
        indicating that the trait was set to it's calculated default
        value.

        Args:
            change (dict): Metadata about the change being made.

        """
        type2value = {
            "default": True,
            "change": False,
            "clear": None,
        }
        if change['type'] not in type2value:  # pragma: no cover
            warnings.warn(f"Traitlets change type \"{change['type']}\" "
                          f"not explicitly accounted for.")
        value = type2value.get(change['type'], False)
        inst = change['owner']
        metadata_name = "_" + self.trait_name + "_metadata"
        if not hasattr(inst, metadata_name):
            setattr(inst, metadata_name, {})
        getattr(inst, metadata_name)['default_set'] = value

    def instance_init(self, inst):
        r"""Callback method for initializing observation handlers to
        handle updates to this trait based on changes to dependencies.

        Args:
            inst (traitlets.HasTraits): Instance that the trait belongs
                to.

        """
        inst.observe(self.set_default, [self.trait_name],
                     type=traitlets.All)
        for x in self.change_type:
            inst.observe(self.recalculate, self.depends_on, type=x)


class DependentDefaultHandler(DependentTraitMixin,
                              traitlets.DefaultHandler):
    r"""Version of traitlets.DefaultHandler that uses the same method
    that dynamically returns a default value to update a trait when
    any of the traits that it depends on are updated.

    Args:
        name (str): Name of the trait that the handler will update.
        depends_on (list): Name(s) of traits that the updated trait
            depends on to calculate the default.
        **kwargs: Additional keyword arguments are passed to the parent
            class.

    """

    def __init__(self, name, depends_on, **kwargs):
        super(DependentDefaultHandler, self).__init__(
            depends_on, name, **kwargs)


class DependentProperty(DependentTraitMixin, traitlets.TraitType,
                        traitlets.EventHandler):
    r"""Event handler that allows methods to be treated as cached
    properties that are updated when the traits it depends on are
    updated.

    Args:
        *depends_on: Name(s) of traits that the property depends on.
        read_only (bool, optional): If True, the property will be
            read-only. Defaults to True.
        **kwargs: Additional keyword arguments are passed to the
            traitlets.TraitType constructor.

    """

    def __init__(self, *depends_on, read_only=True, **kwargs):
        super(DependentProperty, self).__init__(
            depends_on, read_only=read_only, **kwargs)

    def _init_call(self, func):
        self.trait_name = func.__name__
        return super(DependentProperty, self)._init_call(func)

    def default(self, inst):
        r"""Evaluate the method to calculate the property value as the
        default.

        Args:
            inst (traitlets.HasTraits): Instance that the property
                belongs to.

        Returns:
            object: Calculated property value.

        """
        return self(inst)


def dependent_default(name, depends_on, **kwargs):
    r"""Decorator to turn a method into a trait default generator that
    will also cause the trait value to be updated when one of the
    properties it depends on is updated and the trait has not already
    been explicitly set.

    Args:
        name (str): Trait name.
        depends_on (list): Name(s) of traits that the updated trait
            depends on to calculate the default.
        **kwargs: Additional keyword arguments are passed to the
            DependentDefaultHandler constructor.

    Returns:
        DependentDefaultHandler: Handler that can be called on a method
            to add the appropriate trait callbacks.

    """
    return DependentDefaultHandler(name, depends_on, **kwargs)


def dependent_property(*depends_on, **kwargs):
    r"""Decorator to turn a method into a property that will be updated
    when the traits it depends on is updated.

    Args:
        *depends_on: Name(s) of traits that the property depends on.
        **kwargs: Additional keyword arguments are passed to the
            DependentProperty constructor.

    Returns:
        DependentProperty: Handler that can be called on a method to
            add the appropriate trait callbacks.

    """
    return DependentProperty(*depends_on, **kwargs)


def check_shape(*dimensions, ignore_trailing=False):
    r"""Factory to return a valiator to check the shape of an ndarray
    trait.

    Args:
        *dimensions: Arguments are treated as sizes in each dimension.
        ignore_trailing (bool, optional): If True, the value must only
            match the provided dimensions and any trailing dimensions
            will be ignored.

    Returns:
        callable: Validator that takes a traitlets.TraitType and its
            value as input.

    """

    def validator(trait, value):
        if value is None:
            return value
        if ignore_trailing:
            if len(value.shape) < len(dimensions):
                raise traitlets.TraitError(
                    f"{trait.name}: expected rank >={len(dimensions)} "
                    f"but got rank {len(value.shape)}"
                )
        elif len(value.shape) != len(dimensions):
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
