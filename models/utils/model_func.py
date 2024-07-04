#!/usr/bin/env python3
"""This module contains helper functions for classes"""


def add_data(cls, value, attr) -> int:
    """Adds a value to object's attribute list.
    Args:
        cls: The object to add the value to
        value: The value to add
        attr: The attribute to add the value to
    """
    if value is None or value == "NULL" or \
        value == "None" or value == "":
        return 0
    prev = cls.__dict__[attr]
    if value in prev:
        print(f"Value {value} already in {attr}")
        return 0
    if len(prev) == 0:
        cls.__dict__[attr] = [value]
    else:
        cls.__dict__[attr].append(value)
    return 1

def get_data(cls, attr) -> list:
    """Gets the data from an object's attribute.
    Args:
        cls: The object to get the data from
        attr: The attribute to get the data from
    """
    from models.base import storage
    data = storage.all(cls)
    data_list = []
    for obj in data.values():
        data_list.append(obj.__dict__[attr])