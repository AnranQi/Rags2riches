from typing import List, Tuple, Union
def update_seams_if_reversed(seams: List[List[Union[int, Tuple[int, int]]]],
    pid: int,
    path: List[int],
    is_reversed: bool
) -> List[List[Union[int, Tuple[int, int]]]]:
    """
    Reverses edge indices for a specific panel if the stitch path is reversed.

    Args:
        seams: List of seams [p1, e1, p2, e2].
        pid: Panel index to check.
        path: Stitch path list.
        is_reversed: Whether the path is reversed.

    Returns:
        Updated seam list with adjusted edge indices.
    """
    if not is_reversed:
        return seams

    n = len(path)
    for i, (p1, e1, p2, e2) in enumerate(seams):
        if p1 == pid and isinstance(e1, int):
            seams[i][1] = n - 1 - e1
        if p2 == pid and isinstance(e2, int):
            seams[i][3] = n - 1 - e2

    return seams



def replace_edges_with_tp(seams: List[List[Union[int, Tuple[int, int]]]],
    pid: int,
    tp_edges: List[Tuple[int, int]]
) -> List[List[Union[int, Tuple[int, int]]]]:
    """
    Replaces edge indices with (start, end) TP indices for a given panel.

    Args:
        seams: List of seams [p1, e1, p2, e2].
        pid: Target panel index.
        tp_edges: Edge index to (start, end) TP mapping.

    Returns:
        Seam list with edges replaced by TP tuples.
    """
    for i, (p1, e1, p2, e2) in enumerate(seams):
        if p1 == pid and isinstance(e1, int):
            seams[i][1] = tp_edges[e1]
        if p2 == pid and isinstance(e2, int):
            seams[i][3] = tp_edges[e2]

    return seams

