import ast
import copy
import os
import numpy as np
from holders import source_edges, source_vertices, target_vertices, target_edges
from pattern.wrappers import signed_area
from svgpathtools import paths2svg, svg2paths, Path

class Vertex:
    def __init__(self, index, s_or_t, type=None, x=None, y=None, quad_x=None, quad_y=None, segment_index=None,
                 segment_t=None, panel_path_number=None):
        self.index = index  # explain: the index of the vertex
        self.s_or_t = s_or_t  # explain: source or target
        self.type = type  # explain: vertex only has two types: boundary or newcut; "seam" is only available for edge.
        self.x = x
        self.y = y
        self.quad_x = quad_x
        self.quad_y = quad_y
        self.segment_index = segment_index
        self.segment_t = segment_t
        self.panel_path_number = panel_path_number  # only boundary/seam vertex have this attribute

    def set_type(self, type):
        self.type = type

    def set_curve_x(self, x):
        self.x = x

    def set_curve_y(self, y):
        self.y = y

    def set_quad_x(self, quad_x):
        self.quad_x = quad_x

    def set_quad_y(self, quad_y):
        self.quad_y = quad_y

    def set_panel(self, panel_name):
        self.panel_name = panel_name

    def set_segment_index(self, segment_index):
        self.segment_index = segment_index

    def set_panel_path_number(self, panel_path_number):
        # explain: set the number of path on the current panel
        self.panel_path_number = panel_path_number

    def set_segment_t(self, segment_t):
        self.segment_t = segment_t

    def set_global_t(self, global_t):
        self.global_t = global_t

    def get_index(self):
        return self.index

    def get_type(self):
        return self.type

    def get_curve_x(self):
        return self.x

    def get_curve_y(self):
        return self.y

    def get_quad_x(self):
        return self.quad_x

    def get_quad_y(self):
        return self.quad_y

    def get_panel(self):
        return self.panel_name

    def get_segment_index(self):
        return self.segment_index

    def get_segment_t(self):
        return self.segment_t

    def get_global_t(self):
        return self.global_t

    def get_s_or_t(self):
        return self.s_or_t

    def get_panel_path_number(self):
        # explain: set the number of path on the current panel
        return self.panel_path_number


class Edge:
    def __init__(self, start_vertex_index, end_vertex_index, s_or_t):
        self.start_vertex_index = start_vertex_index
        self.end_vertex_index = end_vertex_index
        self.s_or_t = s_or_t  # explain: this need to be set outside, as we only have the start_vertex_index, still need to know if it is from source/target, then search
        self.path_index = None
        self.deformation_gradient = None
        self.cycle_indices = []
        self.set_vertices()
        self.set_type()
        self.set_panel_name()
        self.pair_index = None

    def set_vertices(self):
        if self.s_or_t == "source":
            v0 = find_object_by_attribute(source_vertices, "index", self.start_vertex_index)
            v1 = find_object_by_attribute(source_vertices, "index", self.end_vertex_index)
        else:
            v0 = find_object_by_attribute(target_vertices, "index", self.start_vertex_index)
            v1 = find_object_by_attribute(target_vertices, "index", self.end_vertex_index)
        self.start_vertex = v0
        self.end_vertex = v1

        self.start_vertex_quad_x = self.start_vertex.get_quad_x()
        self.start_vertex_quad_y = self.start_vertex.get_quad_y()
        self.end_vertex_quad_x = self.end_vertex.get_quad_x()
        self.end_vertex_quad_y = self.end_vertex.get_quad_y()

        self.start_vertex_x = self.start_vertex.get_curve_x()
        self.start_vertex_y = self.start_vertex.get_curve_y()
        self.end_vertex_x = self.end_vertex.get_curve_x()
        self.end_vertex_y = self.end_vertex.get_curve_y()

    def set_type(self):
        # explain: this is the first time set, not consider "seam" yet
        if self.start_vertex.get_type() == "boundary" and self.end_vertex.get_type() == "boundary":
            if self.start_vertex.get_segment_index() == self.end_vertex.get_segment_index():
                self.type = "boundary"
            elif (self.start_vertex.get_segment_index() + 1) % self.start_vertex.get_panel_path_number() == self.end_vertex.get_segment_index() and self.end_vertex.get_segment_t() < 0.005:  # explain: should be 0?
                self.type = "boundary"
            else:
                self.type = "newcut"
        else:
            self.type = "newcut"

    def set_seam_type(self):
        # explain: this is for setting the "seam" type only
        self.type = "seam"

    def set_panel_name(self):
        assert (self.start_vertex.get_panel() == self.end_vertex.get_panel())
        self.panel_name = self.start_vertex.get_panel()

    def set_deformation_gradient(self, deformation_gradient):
        self.deformation_gradient = deformation_gradient

    def set_index(self, index):
        self.index = index

    def set_path_index(self, path_index):
        self.path_index = path_index

    def set_path_t(self, t):
        # this t is the index, sorted by starting position (roughly) on the segment path. Use to find paired seam on another panel. Only the seam (at the boundary) has this attribute
        self.segment_t = t

    def set_max_path_index(self, l):
        # explain: this the max segment number, use to get the seam pair index on the other panel
        self.max_path_index = l

    def find_set_pair_index(self):
        if self.s_or_t == "source":
            e_obj = find_object_by_two_attributes(source_edges, "start_vertex_index", self.end_vertex_index,
                                                  "end_vertex_index", self.start_vertex_index)
        else:
            e_obj = find_object_by_two_attributes(target_edges, "start_vertex_index", self.end_vertex_index,
                                                  "end_vertex_index", self.start_vertex_index)
        if e_obj is not None:
            self.pair_index = e_obj.get_index()
            e_obj.set_pair_index(self.index)

    def set_pair_index(self, index):
        self.pair_index = index

    def set_cycle_indices(self, cycle_index):
        self.cycle_indices.append(cycle_index)

    def get_index(self):
        return self.index

    def get_pair_index(self):
        return self.pair_index

    def get_cycle_indices(self):
        return self.cycle_indices

    def get_type(self):
        return self.type

    def get_panel(self):
        return self.panel_name

    def get_path_index(self):
        return self.path_index

    def get_start_vertex(self):
        return self.start_vertex

    def get_end_vertex(self):
        return self.end_vertex

    def get_deformation_gradient(self):
        return self.deformation_gradient

    def get_path_t(self):
        return self.segment_t

    def get_s_or_t(self):
        return self.s_or_t

    def get_max_path_index(self):
        return self.max_path_index

    def get_start_vertex_quad_x(self):
        return self.start_vertex_quad_x

    def get_start_vertex_quad_y(self):
        return self.start_vertex_quad_y

    def get_end_vertex_quad_x(self):
        return self.end_vertex_quad_x

    def get_end_vertex_quad_y(self):
        return self.end_vertex_quad_y

    def get_start_vertex_x(self):
        return self.start_vertex_x

    def get_start_vertex_y(self):
        return self.start_vertex_y

    def get_end_vertex_x(self):
        return self.end_vertex_x

    def get_end_vertex_y(self):
        return self.end_vertex_y


class Quad:
    def __init__(self, v0_index, v1_index, v2_index, v3_index, s_or_t):
        self.v0_index = v0_index
        self.v1_index = v1_index
        self.v2_index = v2_index
        self.v3_index = v3_index
        self.s_or_t = s_or_t
        self.edges = []

        e0 = Edge(self.v0_index, self.v1_index, self.s_or_t)
        e1 = Edge(self.v1_index, self.v2_index, self.s_or_t)
        e2 = Edge(self.v2_index, self.v3_index, self.s_or_t)
        e3 = Edge(self.v3_index, self.v0_index, self.s_or_t)

        self.edges.append(e0)
        self.edges.append(e1)
        self.edges.append(e2)
        self.edges.append(e3)
        # explain: every edge is new
        if self.s_or_t == "source":
            e0.set_index(len(source_edges))
            e0.find_set_pair_index()
            source_edges.append(e0)

            e1.set_index(len(source_edges))
            e1.find_set_pair_index()
            source_edges.append(e1)

            e2.set_index(len(source_edges))
            e2.find_set_pair_index()
            source_edges.append(e2)

            e3.set_index(len(source_edges))
            e3.find_set_pair_index()
            source_edges.append(e3)
        else:
            e0.set_index(len(target_edges))
            e0.find_set_pair_index()
            target_edges.append(e0)

            e1.set_index(len(target_edges))
            e1.find_set_pair_index()
            target_edges.append(e1)

            e2.set_index(len(target_edges))
            e2.find_set_pair_index()
            target_edges.append(e2)

            e3.set_index(len(target_edges))
            e3.find_set_pair_index()
            target_edges.append(e3)

        self.vertices = []
        for e in self.edges:
            self.vertices.append(e.get_start_vertex())
        self.calculate_center()

    def set_position_x(self, position_x):
        self.position_x = position_x

    def set_position_y(self, position_y):
        self.position_y = position_y

    def set_index(self, index):
        self.index = index

    def set_translated_center_x(self, translation_x):
        self.translated_center_x = self.center_x + translation_x

    def set_translated_center_y(self, translation_y):
        self.translated_center_y = self.center_y + translation_y

    def calculate_overlap(self, quad):
        """ Check if the current quad overlap of another quad(para) by comparing their positions """
        return (self.center_x == quad.get_center_x() and quad.get_cent == quad.get_center_y())

    def calculate_center(self):
        x = 0.0
        y = 0.0
        for v in self.vertices:
            x = x + v.get_quad_x()
            y = y + v.get_quad_y()
        self.center_x = 0.25 * x
        self.center_y = 0.25 * y

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_edges(self):
        return self.edges

    def get_index(self):
        return self.index

    def get_center_x(self):
        return self.center_x

    def get_center_y(self):
        return self.center_y

    def get_translated_center_x(self):
        return self.translated_center_x

    def get_translated_center_y(self):
        return self.translated_center_y


class Panel:
    def __init__(self, index, path, name, s_or_t, pattern):
        self.pattern = pattern
        self.index = index
        self.path = path
        self.name = name
        self.s_or_t = s_or_t  # "source" or "target"
        self.quads = []
        path_bd = paths2svg.big_bounding_box(self.path)
        self.panel_path_center = np.array([0.5 * (path_bd[0] + path_bd[1]), 0.5 * (path_bd[2] + path_bd[3])])
        self.number_of_border_edges = 0
        self.triangle_faces = []
        self.triangle_vertices = []

    def add_quad(self, quad):
        self.quads.append(quad)

    def set_quads(self, quads):
        self.quads = quads
        self.build_panel_index_mask()

    def set_boundary_integer_points(self, boundary_integer_points):
        self.boundary_integer_points = boundary_integer_points
        self.min_x, self.min_y, self.max_x, self.max_y = get_bounding_box(self.boundary_integer_points)
        self.panel_quad_center = np.array([0.5 * (self.min_x + self.max_x), 0.5 * (self.min_y + self.max_y)])

    def set_boundary_curve_points(self, boundary_curve_points):
        self.boundary_curve_points = boundary_curve_points

    def set_translation_curve_to_quad_panel(self, translation_curve_to_quad_panel):
        self.translation_curve_to_quad_panel = translation_curve_to_quad_panel

    def set_seams(self):
        for i in range(len(self.path)):
            current_path_index = i
            pair_panel, pair_path_index = self.pattern._get_globle_stitch_info(self.name, current_path_index)
            if pair_panel != None:
                seam_edges = []
                if self.s_or_t == "source":
                    edges = source_edges
                elif self.s_or_t == "target":
                    edges = target_edges
                else:
                    print("the s_or_t of this panel is not set yet, check please")
                for edge in edges:
                    if edge.get_type() == "boundary" and edge.get_panel() == self.name:
                        self.number_of_border_edges = self.number_of_border_edges + 1
                        start_vertex = edge.get_start_vertex()
                        if start_vertex.get_segment_index() == current_path_index:
                            edge.set_path_t(start_vertex.get_segment_t())
                            seam_edges.append(edge)

                sorted_seam_edges = sorted(seam_edges, key=lambda obj: obj.get_path_t())
                for index, seam in enumerate(sorted_seam_edges):
                    seam.set_max_path_index(len(sorted_seam_edges) - 1)
                    seam.set_path_t(index)
                    seam.set_path_index(current_path_index)
                    seam.set_seam_type()

    def get_path(self):
        return self.path

    def get_name(self):
        return self.name

    def get_s_or_t(self):
        return self.s_or_t

    def get_quads(self):
        return self.quads

    def get_edges(self):
        self.edges = []
        for quad in self.quads:
            for e in quad.get_edges():
                self.edges.append(e)
        return self.edges

    def get_index(self):
        return self.index

    def get_quads_number(self):
        return len(self.quads)

    def get_boundary_integer_points(self):
        return self.boundary_integer_points

    def get_boundary_curve_points(self):
        return self.boundary_curve_points

    def get_boundary_vertices(self):
        # explain: this is the vertex (class) on the panel's boundary
        return self.boundary_vertices

    def get_bounding_box(self):
        return self.min_x, self.min_y, self.max_x, self.max_y

    def get_bounding_box_center(self):
        return 0.5 * (self.max_x - self.min_x), 0.5 * (self.max_y - self.min_y)

    def build_panel_index_mask(self, pad_width=0):
        # pad: pad the array in both x, y direction
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        panel_index_mask = np.zeros((width, height))  # the array of the index, if not occupied, then 0.
        panel_index_mask_shift1 = np.zeros((width, height))  # quad index start from 1 not 0
        for quad in self.quads:
            quad_center_x = quad.get_center_x()
            quad_center_y = quad.get_center_y()
            # Translate the center of the quad relative to the bounding box
            relative_x = quad_center_x - self.min_x
            relative_y = quad_center_y - self.min_y
            grid_index_x = int(np.floor(relative_x))
            grid_index_y = int(np.floor(relative_y))
            panel_index_mask[grid_index_x, grid_index_y] = quad.get_index()
            # explain: quad index starts from 0, make sure the quad index 0 is 1, to distinguish with padded 0
            panel_index_mask_shift1[grid_index_x, grid_index_y] = quad.get_index() + 1

        self.panel_index_mask = np.pad(panel_index_mask, pad_width=pad_width, mode='constant', constant_values=0)
        self.panel_index_mask_shift1 = np.pad(panel_index_mask_shift1, pad_width=pad_width, mode='constant',
                                              constant_values=0)

    def get_panel_index_mask_shift1(self):
        # explain: this is because one the panel there might be one quad whose index is 0, which will casue some problem when computing the binary mask
        return self.panel_index_mask_shift1

    def get_panel_quad_bounding_box_center(self):
        return self.panel_quad_center[0], self.panel_quad_center[1]

    def get_number_of_border_edges(self):
        # explain: should be after the self.set_seam() function,
        return self.number_of_border_edges

    def get_translation_curve_to_quad_panel(self):
        # explain: this is the translation, that curve needs to do (curve+ self.translation_curve_to_quad_panel to best align with the quad panel in a least square sense (centroid)
        return self.translation_curve_to_quad_panel



class Cycle:
    def __init__(self, index):
        self.index = index
    def set_source_panel_index(self, source_panel_index):
        self.source_panel_index = source_panel_index

    def set_source_panel_name(self, name):
        self.source_panel_name = name

    def set_target_panel_index(self, target_panel_index):
        # which target panel this cycle is corresponding with
        self.target_panel_index = target_panel_index

    def set_target_panel_name(self, name):
        self.target_panel_name = name

    def set_polyomino(self, polyomino):
        self.polyomino = polyomino
        self.compute_greedy_cover_polyomino()

    def set_bounding_box(self, min_x, min_y, weight, height):
        self.min_x = min_x
        self.min_y = min_y
        self.weight = weight
        self.height = height

    def set_quad_iou(self, iou):
        self.quad_iou = iou

    def set_curve_iou(self, iou):
        self.curve_iou = iou

    def set_boundary_polygon(self, polygon):
        # this boundary_polgyon (Shapely, Polygon) is the all boundary points' positions in order that on the source panel. this is used to compute intersection in the curve space and extract the final results
        self.boundary_polygon = polygon

    def set_st_edge_pairs(self, st_edge_pairs):
        # explain:record all the source-target edge pairs; then set the possible seam based on information on the target
        self.st_edge_pairs = st_edge_pairs
        self.st_edge_pairs_dict = dict(self.st_edge_pairs)
        self.set_st_edges_based_tseam()
        self.set_st_edges_based_stseam()

    def set_decided_vertices_in_tbv(self, decided_vertices_in_tbv):
        # explain: set fixed vertices for boundary vertices
        self.decided_vertices_in_tbv = decided_vertices_in_tbv

    def set_new_reused_vertex_xys(self, new_reused_vertex_xys):
        # explain: set new_reused_vertex_xys
        self.new_reused_vertex_xys = new_reused_vertex_xys

    def get_decided_vertices_in_tbv(self):
        return self.decided_vertices_in_tbv

    def get_new_reused_vertex_xys(self):
        return self.new_reused_vertex_xys

    def get_source_panel_index(self):
        # which source panel this cycle is come from
        return self.source_panel_index

    def get_target_panel_index(self):
        # which target panel this cycle is corresponding with
        return self.target_panel_index

    def get_bounding_box(self):
        return self.min_x, self.min_y, self.weight, self.height

    def get_bounding_box_center(self):
        return self.min_x + self.weight * 0.5, self.min_y + self.height * 0.5

    def get_quad_iou(self):
        return self.quad_iou

    def get_st_edge_pairs(self):
        # record all the source-target edge pairs
        return self.st_edge_pairs

    def set_st_edges_based_tseam(self):
        source_seam_edges = []
        target_seam_edges = []
        for (s_edge, t_edge) in self.st_edge_pairs:
            if t_edge.get_type() == "seam":
                source_seam_edges.append(s_edge)
                target_seam_edges.append(t_edge)
        self.source_edges_based_tseam = source_seam_edges
        self.target_edges_based_tseam = target_seam_edges

    def get_st_edges_based_tseam(self):
        return self.source_edges_based_tseam, self.target_edges_based_tseam

    def set_st_edges_based_stseam(self):
        source_seam_edges = []
        target_seam_edges = []
        for (s_edge, t_edge) in self.st_edge_pairs:
            if s_edge.get_type() == "seam" and t_edge.get_type() == "seam":
                source_seam_edges.append(s_edge)
                target_seam_edges.append(t_edge)
        self.source_edges_based_stseam = source_seam_edges
        self.target_edges_based_stseam = target_seam_edges

    def get_st_edges_based_stseam(self):
        return self.source_edges_based_stseam, self.target_edges_based_stseam

    def get_source_panel_name(self):
        return self.source_panel_name

    def get_target_panel_name(self):
        return self.target_panel_name

    def set_translation(self, translation_vector):
        # the x, y translation the current sliding window
        self.translation_vector = translation_vector

    def get_translation(self):
        # the x, y translation the current sliding window
        return self.translation_vector

    def compute_greedy_cover_polyomino(self):
        """
        return a list of bounding box in a format of (minx, miny, width, height)
        :param polyomino: a list of 0/1 (width, height), which 1 indicates occupied of the current panel
        :type polyomino:
        :return: [(minx, miny, width, height), (),..]
        :rtype:
        """
        rectangles = []
        n, m = len(self.polyomino), len(self.polyomino[0])
        # Create a deep copy of the original list
        polyomino = copy.deepcopy(self.polyomino)
        for i in range(n):
            for j in range(m):
                if polyomino[i][j] == 1:
                    # Find the largest rectangle starting from (i, j)
                    x1, y1, x2, y2 = find_max_rectangle(polyomino, i, j)
                    # Cover this rectangle in the grid
                    polyomino = cover_rectangle(polyomino, x1, y1, x2, y2)
                    # Add the rectangle to the list of rectangles
                    rectangles.append((x1, y1, x2 - x1 + 1, y2 - y1 + 1))
        self.divided_bounding_boxes = rectangles  # explain: this is based on per panel, no global translation

    def get_divided_bounding_boxes(self):
        return self.divided_bounding_boxes

    def get_index(self):
        return self.index


class Pattern:
    def __init__(self, panel_dir, panel_names):

        self.panel_names = panel_names
        self.panel_dir = panel_dir
        self.pattern = {}
        self.stitch_file = os.path.join(self.panel_dir, 'stitch.txt')
        with open(self.stitch_file, 'r') as file:
            # Read content and parse it as a Python literal using `ast.literal_eval`
            stitch = ast.literal_eval(file.read())
        self.pattern['stitches'] = stitch
        self.pattern['panels'] = {}
        for panel_name in self.panel_names:
            self.pattern['panels'][panel_name] = {}

    def _read_a_panel(self, panel_name):
        paths, attributes = svg2paths(os.path.join(self.panel_dir, panel_name + '.svg'))
        # check the length of the last segment
        if paths[0][-1].length() < 0.1:
            paths[0].pop(-1)  # Remove the last segment
        return paths[0]

    # explain: this needs to be called for each panel: Function to check orientation and return reversed path with index mapping
    def ensure_counterclockwise(self, panel_name, path):
        area = signed_area(path)
        if area > 0:
            # Reverse the path to make it counterclockwise
            reversed_path = Path(*reversed([seg.reversed() for seg in path]))
            # explain: update seam information
            for ind, stitch in enumerate(self.pattern['stitches']):
                if stitch[0]['panel'] == panel_name:
                    self.pattern['stitches'][ind][0]['edge'] = len(path) - 1 - stitch[0]['edge']
                if stitch[1]['panel'] == panel_name:
                    self.pattern['stitches'][ind][1]['edge'] = len(path) - 1 - stitch[1]['edge']
            return reversed_path, True
        else:
            return path, False

    def get_stitch(self):
        stitches = self.pattern['stitches']
        return stitches

    def _get_globle_stitch_panel(self, panelname):
        corresponding_panels = []
        stitches = self.pattern['stitches']
        for stitch_pair in stitches:
            stitch_one = stitch_pair[0]
            stitch_one_panel = stitch_one["panel"]
            stitch_two = stitch_pair[1]
            stitch_two_panel = stitch_two["panel"]
            if stitch_one_panel == panelname:
                corresponding_panels.append(stitch_two_panel)
            elif stitch_two_panel == panelname:
                corresponding_panels.append(stitch_one_panel)
        return corresponding_panels

    def _get_globle_stitch_info(self, panelname, original_edge_index_on_panel):
        """
            #get the another global pair information given the current information
        :param panelname: the name of the current panel
        :type panelname: str
        :param original_edge_index_on_panel: the original edge index on the panel of the current small edge
        :type original_edge_index_on_panel:  int

        """
        stitches = self.pattern['stitches']
        for stitch_pair in stitches:
            stitch_one = stitch_pair[0]
            stitch_one_panel = stitch_one["panel"]
            stitch_one_edge_index = stitch_one["edge"]
            stitch_two = stitch_pair[1]
            stitch_two_panel = stitch_two["panel"]
            stitch_two_edge_index = stitch_two["edge"]

            if stitch_one_panel == panelname and stitch_one_edge_index == original_edge_index_on_panel:
                return stitch_two_panel, stitch_two_edge_index
            elif stitch_two_panel == panelname and stitch_two_edge_index == original_edge_index_on_panel:
                return stitch_one_panel, stitch_one_edge_index
        return None, None

######################################

def find_object_by_attribute(obj_list, attribute, value):
    for obj in obj_list:
        if getattr(obj, attribute) == value:
            return obj
    return None


# Function to find an object by two attributes
def find_object_by_two_attributes(obj_list, attr1, value1, attr2, value2):
    for obj in obj_list:
        if getattr(obj, attr1) == value1 and getattr(obj, attr2) == value2:
            return obj
    return None


def find_object_by_three_attributes(obj_list, attr1, value1, attr2, value2, attr3, value3):
    for obj in obj_list:
        if getattr(obj, attr1) == value1 and getattr(obj, attr2) == value2 and getattr(obj, attr3) == value3:
            return obj
    return None


def find_object_by_five_attributes(obj_list, attr1, value1, attr2, value2, attr3, value3, attr4, value4, attr5, value5):
    for obj in obj_list:
        if getattr(obj, attr1) == value1 and getattr(obj, attr2) == value2 and getattr(obj,attr3) == value3 and getattr(obj, attr4) == value4 and getattr(obj, attr5) == value5:
            return obj
    return None


def get_bounding_box(points):
    """
    To calculate the bounding box of a list of points with coordinates (x,y)(x, y)(x,y)
    :param points:
    :type points: [(x,y), (x, y), (x,y), ...]
    :return: (min_x, min_y, max_x, max_y)
    :rtype:
    """
    if not points:
        return None  # Return None if the list is empty

    # Separate the x and y coordinates
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # Find the minimum and maximum values for x and y
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # Return the bounding box as a tuple (min_x, min_y, max_x, max_y)
    return min_x, min_y, max_x, max_y


def find_max_rectangle(grid, start_x, start_y):
    """Find the largest rectangle starting at (start_x, start_y) that can be fully covered."""
    max_x, max_y = len(grid), len(grid[0])
    end_x, end_y = start_x, start_y

    while end_x < max_x and grid[end_x][start_y] == 1:
        temp_y = start_y
        while temp_y < max_y and grid[end_x][temp_y] == 1:
            temp_y += 1
        end_y = min(end_y, temp_y - 1) if end_y != start_y else temp_y - 1
        end_x += 1

    # Return the coordinates of the largest rectangle found
    return start_x, start_y, end_x - 1, end_y


def cover_rectangle(grid, x1, y1, x2, y2):
    """Cover the specified rectangle area by setting it to 0."""
    for i in range(x1, x2 + 1):
        for j in range(y1, y2 + 1):
            grid[i][j] = 0
    return grid


def set_seam_pair_index(edges, pattern):
    """
    Assigns a pair index to each seam edge based on a corresponding seam
    on the opposite panel. This should be called after the clockwise order
    is verified and data preparation is complete.
    """
    for edge in edges:
        if edge.get_type() != 'seam':
            continue
        panel_name = edge.get_panel()
        path_index = edge.get_path_index()
        path_segment_index = edge.get_path_t()
        max_path_segment_index = edge.get_max_path_index()

        # Get the corresponding panel and path index from the pattern
        corr_panel, corr_path_index = pattern._get_globle_stitch_info(panel_name, path_index)

        for candidate_edge in edges:
            if candidate_edge.get_type() != 'seam':
                continue
            if (candidate_edge.get_panel() == corr_panel and
                    candidate_edge.get_path_index() == corr_path_index):
                expected_path_t = max_path_segment_index - path_segment_index
                if candidate_edge.get_path_t() == expected_path_t:
                    edge.set_pair_index(candidate_edge.get_index())
                    break