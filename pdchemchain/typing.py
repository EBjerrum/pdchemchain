class InColumnName(str):
    """Special string type, that should be used to typehint properties that are in_columns.
    This information will be used to assert that the input columns exists in processed dataframes"""

    pass


class OutColumnName(str):
    """Special string type, that should be used to typehint properties that are out_columns.
    This information may be used to warn users that the columns already exists and will be overwritten"""

    pass
