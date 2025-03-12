def classify_water_level(level: int, danger_level: int, max_level: int, x: int) -> int:
    """
    Classify the water level based on the given parameters.

    Params
    ------
    - level (int): The water level to classify.
    - danger_level (int): The danger level of the water.
    - max_level (int): The maximum level of the water.
    - x (int): The number of classes to classify the water level into.
    """
    if level < danger_level:
        return 0
    else:
        catogry = (level - danger_level) / (max_level - danger_level) * x
        return int(catogry)
