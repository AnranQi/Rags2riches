def bbox_intersects(bbox1, bbox2):
    # Convert (min_x, min_y, width, height) to (x_min, y_min, x_max, y_max)
    x1_min, y1_min, w1, h1 = bbox1
    x1_max = x1_min + w1
    y1_max = y1_min + h1

    x2_min, y2_min, w2, h2 = bbox2
    x2_max = x2_min + w2
    y2_max = y2_min + h2

    # Check for intersection
    return not (x1_max <= x2_min or  # bbox1 is to the left of bbox2
                x1_min >= x2_max or  # bbox1 is to the right of bbox2
                y1_max <= y2_min or  # bbox1 is above bbox2
                y1_min >= y2_max)    # bbox1 is below bbox2

def check_intersections(list1, list2):
    for bbox1 in list1:
        for bbox2 in list2:
            if bbox_intersects(bbox1, bbox2):
                return True  # Found an intersection
    return False  # No intersections found

def translated_quads_overlap(source_quad, target_quad):
    """ Check if two quads overlap by comparing their positions """
    return (source_quad.get_center_x() == target_quad.get_translated_center_x() and source_quad.get_center_y() == target_quad.get_translated_center_y())



