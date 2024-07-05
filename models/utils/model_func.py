#!/usr/bin/env python3
"""This module contains helper functions for classes"""


def add_data(caller, attr, cls, value: str):
    """Adds a value to object's attribute list.
    Args:
        caller: The caller object requesting update
        attr: The attribute to add the value to
        cls: The class of the value to add
        value: The value to add (id of other object)
    """
    from models.base import storage

    if value is None or value == "NULL" or \
        value == "None" or value == "":
        return
    val = storage.get_by(cls, id=value)
    if not val:
        return
    try:
        values = getattr(caller, attr)
        if value not in values:
            values.append(value)
    except AttributeError:
        values = [value]
    return values


def getter(cls, attr, **kwargs)->dict:
    """Gets an object from the database
    Args:
        cls: The class of the object to get
        attr: The attribute to get
        kwargs: The key-value pairs to search for
    return: The value of the attribute
    """
    from models.base import storage
    x = storage.get_by(cls, **kwargs)
    if not x:
        return []
    return x.to_dict().get(attr, [])


def update_value(caller, cls, attr, value, **kwargs)->dict:
    """Sets an object in the database
    Args:
        cls: The class of the object to set
        attr: The attribute to set
        value: The value to set (id of other object)
        kwargs: The key-value pairs to search for
    """
    from models.base import storage
    # get exixting value
    x = storage.get_by(cls, **kwargs)
    if not x:
        caller.__dict__[attr] = value
        return caller.__dict__[attr]
    x_vals = x.to_dict()[attr]
    caller.__dict__[attr] = list(set([*x_vals, *value]))
    return caller.__dict__[attr]

