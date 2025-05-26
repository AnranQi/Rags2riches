from typing import List, Tuple
import igl
import numpy as np
from elements import Vertex, Quad
from elements import find_object_by_three_attributes, find_object_by_five_attributes
from holders import source_quads, target_quads, source_vertices, target_vertices, source_edges, target_edges


def build_quads_from_polygon(panel):
    s_or_t = panel.get_s_or_t()
    panel_name = panel.get_name()
    polygon = panel.get_boundary_integer_points()
    x_min, y_min, x_max, y_max = panel.get_bounding_box()
    panel_quads = []
    # explain: for asap deformation, it can not use the index of source/target vertices, need to compute seperatly
    triangle_faces = []
    triangle_vertices = []

    asap_fixed_indices = []  # fixed index in the quad mesh
    asap_fixed_values = []  # the coresponding postions in the pattern
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            p1 = (i, j + 1)
            p2 = (i, j)
            p3 = (i + 1, j)
            p4 = (i + 1, j + 1)
            quad = [p1, p2, p3, p4]
            if is_quad_inside_polygon(quad, polygon):
                per_triangle_vertices_index = []
                quad_vertices_index = []
                for p in quad:
                    if s_or_t == "source":
                        v_obj = find_object_by_three_attributes(source_vertices, "quad_x", p[0], "quad_y", p[1],
                                                                "panel_name", panel_name)
                    else:
                        v_obj = find_object_by_three_attributes(target_vertices, "quad_x", p[0], "quad_y", p[1],
                                                                "panel_name", panel_name)
                    if v_obj == None:
                        if s_or_t == "source":
                            v_index = len(source_vertices)
                        else:
                            v_index = len(target_vertices)
                        v_obj = Vertex(v_index, s_or_t)
                        v_obj.set_panel(panel_name)
                        v_obj.set_type("newcut")
                        v_obj.set_quad_x(p[0])
                        v_obj.set_quad_y(p[1])
                        if s_or_t == "source":
                            source_vertices.append(v_obj)
                        else:
                            target_vertices.append(v_obj)
                    v_index = v_obj.get_index()
                    quad_vertices_index.append(v_index)
                    # explain: process triangle used in arap
                    triangle_vertex = (v_obj.get_quad_x(), v_obj.get_quad_y())
                    if triangle_vertex not in triangle_vertices:
                        triangle_vertices.append((v_obj.get_quad_x(), v_obj.get_quad_y()))
                    triangle_vertex_index = triangle_vertices.index(triangle_vertex)
                    per_triangle_vertices_index.append(triangle_vertex_index)
                    # find the fixed vertices index for asap
                    if p in polygon and triangle_vertex_index not in asap_fixed_indices:
                        asap_fixed_indices.append(triangle_vertex_index)
                        coresponding_panel_points = (v_obj.get_curve_x(), v_obj.get_curve_y())
                        asap_fixed_values.append(coresponding_panel_points)
                    # end to process triangle

                q = Quad(quad_vertices_index[0], quad_vertices_index[1], quad_vertices_index[2], quad_vertices_index[3],
                         s_or_t)
                q.set_position_x(i)
                q.set_position_y(j)

                if s_or_t == "source":
                    q.set_index(len(source_quads))
                    source_quads.append(q)
                else:
                    q.set_index(len(target_quads))
                    target_quads.append(q)
                panel_quads.append(q)

                # build the triangle for arap
                triangle_faces.append(
                    [per_triangle_vertices_index[0], per_triangle_vertices_index[1], per_triangle_vertices_index[2]])
                triangle_faces.append(
                    [per_triangle_vertices_index[0], per_triangle_vertices_index[2], per_triangle_vertices_index[3]])

    # explain: set quads for the panel
    panel.set_quads(panel_quads)
    # Convert the list of triangle_faces to a numpy array
    triangle_faces = np.array(triangle_faces)
    triangle_vertices = np.array(triangle_vertices)

    asap_fixed_indices = np.array(asap_fixed_indices)
    asap_fixed_values = np.array(asap_fixed_values)

    arap = igl.ARAP(triangle_vertices, triangle_faces, 2, asap_fixed_indices)

    triangle_vertices_deformed = arap.solve(asap_fixed_values, triangle_vertices)
    return triangle_faces, triangle_vertices, triangle_vertices_deformed


def is_array_in_list_np(array, array_list):
    for arr in array_list:
        if np.array_equal(arr, array):
            return True
    return False


def compute_deformation_gradients_edge(panel, triangle_vertices, triangle_vertices_deformed, triangle_edge_indices):
    """
    given the triangle information (not in class format of vertex, edges), to compute the deformation gradient per edge
    :param panel: panel class object
    :param triangle_vertices: triangle_vertices
    :type triangle_vertices: list of list (not list of class)
    :param triangle_vertices_deformed:
    :type triangle_vertices_deformed: list of list (not list of class)
    :param triangle_edge_indices:
    :type triangle_edge_indices: list of list

    :return:
    edge_centers:list, the center coordinates of all edges
    gradients: list of 3 dims: number_edges * 2 * 2, which stores the deformation gradient per edge
    boundary_centers:list, the center coordinates of boundary edges
    boundary_gradients: deformation gradient of the boundary edges
    :rtype:
    """

    type = panel.get_s_or_t()  # type: "source" or "target"
    panel_name = panel.get_name()
    boundary_integer_points = panel.get_boundary_integer_points()
    # Extract all x coordinates
    quad_x_coords = [point[0] for point in boundary_integer_points]  # the integer coordinates of x for the boundary
    quad_y_coords = [point[1] for point in boundary_integer_points]

    # compute the deformation gradient based on per edge
    gradients = []
    boundary_gradients = []
    boundary_centers = []
    edge_centers = []
    boundary_coors = np.vstack((quad_x_coords, quad_y_coords)).T
    for edge in triangle_edge_indices:
        v0_index, v1_index = edge
        original_edge = triangle_vertices[v1_index] - triangle_vertices[v0_index]

        if type == "source":
            # here we can have 1 edgee containing those two vertices in order
            edge = find_object_by_five_attributes(source_edges, "start_vertex_quad_x", triangle_vertices[v0_index][0],
                                                  "start_vertex_quad_y", triangle_vertices[v0_index][1],
                                                  "end_vertex_quad_x", triangle_vertices[v1_index][0],
                                                  "end_vertex_quad_y",
                                                  triangle_vertices[v1_index][1], "panel_name", panel_name)
        else:
            # here we can have 1 edgee containing those two vertices in order
            edge = find_object_by_five_attributes(target_edges, "start_vertex_quad_x", triangle_vertices[v0_index][0],
                                                  "start_vertex_quad_y", triangle_vertices[v0_index][1],
                                                  "end_vertex_quad_x", triangle_vertices[v1_index][0],
                                                  "end_vertex_quad_y",
                                                  triangle_vertices[v1_index][1], "panel_name", panel_name)

        deformed_edge = triangle_vertices_deformed[v1_index] - triangle_vertices_deformed[v0_index]

        # Compute deformation gradient F for this edge
        # Assuming edges are non-zero, normalize the edge vectors
        original_edge_normalized = original_edge / np.linalg.norm(original_edge)
        deformed_edge_normalized = deformed_edge / np.linalg.norm(original_edge)

        # Deformation gradient for the edge
        F = np.outer(deformed_edge_normalized, original_edge_normalized)

        edge.set_deformation_gradient(F)
        if edge is None:
            print("this should not happen")

        gradients.append(F)
        edge_centers.append(0.5 * (triangle_vertices[v1_index] + triangle_vertices[v0_index]))
        if is_array_in_list_np(triangle_vertices[v0_index], boundary_coors) and is_array_in_list_np(
                triangle_vertices[v1_index], boundary_coors):
            boundary_gradients.append(F)
            boundary_centers.append(0.5 * (triangle_vertices[v1_index] + triangle_vertices[v0_index]))

    return edge_centers, gradients, boundary_centers, boundary_gradients


def is_point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    # is the point inside the polygon
    # ray tracing method
    num = len(polygon)
    j = num - 1
    odd_nodes = False

    for i in range(num):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if yi < y <= yj or yj < y <= yi:
            if xi + (y - yi) / (yj - yi) * (xj - xi) < x:
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes


def is_point_on_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    # is the point on the edge of polygon
    num = len(polygon)
    j = num - 1
    for i in range(num):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if is_on_segment((xi, yi), (xj, yj), (x, y)):
            return True
        j = i
    return False

def is_on_segment(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
    # is the point on the segment
    if min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]):
        return True
    return False


def is_quad_inside_polygon(quad: List[Tuple[float, float]], polygon: List[Tuple[float, float]]) -> bool:
    # including both fully inside or on the boundary of the polygon
    for vertex in quad:
        if not (is_point_in_polygon(vertex[0], vertex[1], polygon) or is_point_on_polygon(vertex[0], vertex[1], polygon)):
            return False
    return True


def compute_edges(faces, vertices):
    edges = []
    for face in faces:
        for i in range(3):
            edge = [face[i], face[(i + 1) % 3]]
            v0, v1 = edge
            if (vertices[v1][0] == vertices[v0][0]) or (vertices[v1][1] == vertices[v0][1]):
                edges.append(edge)
    return np.array(edges)
