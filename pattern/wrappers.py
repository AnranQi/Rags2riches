"""
    To be used in Python 3.6+ due to dependencies
"""

import os
import numpy as np
import svgwrite
from svglib import svglib
from reportlab.graphics import renderPM
from svgpathtools import parse_path, Path, disvg, QuadraticBezier, Arc, Line, CubicBezier
from pattern import core


class VisPattern(core.ParametrizedPattern):
    """
        "Visualizible" pattern wrapper of pattern specification in custom JSON format.
        Input:
            * Pattern template in custom JSON format
        Output representations: 
            * Pattern instance in custom JSON format 
                * In the current state
            * SVG (stitching info is lost)
            * PNG for visualization
        
        Not implemented: 
            * Support for patterns with darts
    """

    # ------------ Interface -------------

    def __init__(self, pattern_file=None, view_ids=True):
        super().__init__(pattern_file)

        # tnx to this all patterns produced from the same template will have the same 
        # visualization scale
        # and that's why I need a class object fot 
        self.scaling_for_drawing = self._verts_to_px_scaling_factor()
        self.view_ids = view_ids  # whatever to render vertices & endes indices
        #self.is_counterclockwise()

    # Function to compute signed area of a path


    # explain: this needs to be called for each panel: Function to check orientation and return reversed path with index mapping
    def ensure_counterclockwise(self, panel_name, path):
        area = signed_area(path)
        if area > 0:
            # Reverse the path to make it counterclockwise

            reversed_path = Path(*reversed([seg.reversed() for seg in path]))


            #explain: update seam information, Create index map: (old_index, new_index)
            for ind, stitch in enumerate(self.pattern['stitches']):
                if stitch[0]['panel'] == panel_name:
                    self.pattern['stitches'][ind][0]['edge'] = len(path) - 1 - stitch[0]['edge']
                if stitch[1]['panel'] == panel_name:
                    self.pattern['stitches'][ind][1]['edge'] = len(path) - 1 - stitch[1]['edge']

            #explain: update vertex information
            panel = self.pattern['panels'][panel_name]
            updated_vertices = []
            for ind, v in enumerate(panel['vertices']):
                updated_vertices.append(panel['vertices'][len(path) - 1 - ind]) #len(path) == len(vertices)
            panel['vertices'] = updated_vertices

            # explain: update edge information
            for ind, e in enumerate(panel['edges']):
                e['endpoints'][0] = len(path) - 1 - e['endpoints'][0]
                e['endpoints'][1] = len(path) - 1 - e['endpoints'][1]

            return reversed_path, True
        else:
            return path, False



    def serialize(self, path, to_subfolder=True, tag=''):

        log_dir = super().serialize(path, to_subfolder, tag=tag)
        svg_file = os.path.join(log_dir, (self.name + tag + '_pattern.svg'))
        png_file = os.path.join(log_dir, (self.name + tag + '_pattern.png'))

        # save visualtisation
        self._save_as_image(svg_file, png_file)

        return log_dir

    def list_to_c(self, num):
        """Convert 2D list or list of 2D lists into complex number/list of complex numbers"""
        if isinstance(num[0], list) or isinstance(num[0], np.ndarray):
            return [complex(n[0], n[1]) for n in num]
        else:
            return complex(num[0], num[1])

    # -------- Drawing ---------

    def _verts_to_px_scaling_factor(self):
        """
        Estimates multiplicative factor to convert vertex units to pixel coordinates
        Heuritic approach, s.t. all the patterns from the same template are displayed similarly
        """
        if len(self.pattern['panels']) == 0:  # empty pattern
            return None
        
        avg_box_x = []
        for panel in self.pattern['panels'].values():
            vertices = np.asarray(panel['vertices'])
            box_size = np.max(vertices, axis=0) - np.min(vertices, axis=0) 
            avg_box_x.append(box_size[0])
        avg_box_x = sum(avg_box_x) / len(avg_box_x)

        if avg_box_x < 2:      # meters
            scaling_to_px = 300
        elif avg_box_x < 200:  # sentimeters
            scaling_to_px = 3
        else:                    # pixels
            scaling_to_px = 1  

        return scaling_to_px

    def _verts_to_px_coords(self, vertices):
        """Convert given to px coordinate frame & units"""
        # Flip Y coordinate (in SVG Y looks down)
        vertices[:, 1] *= -1
        # Put upper left corner of the bounding box at zero
        offset = np.min(vertices, axis=0)
        vertices = vertices - offset
        # Update units scaling
        vertices *= self.scaling_for_drawing
        return vertices

    def _flip_y(self, point):
        """
            To get to image coordinates one might need to flip Y axis
        """
        flipped_point = list(point)  # top-level copy
        flipped_point[1] *= -1
        return flipped_point

    def _draw_a_panel(self, drawing, panel_name, offset=[0, 0]):
        """
        Adds a requested panel to the svg drawing with given offset and scaling
        Assumes (!!) 
            that edges are correctly oriented to form a closed loop
        Returns 
            the lower-right vertex coordinate for the convenice of future offsetting.
        """
        panel = self.pattern['panels'][panel_name]
        vertices = np.asarray(panel['vertices'])
        vertices = self._verts_to_px_coords(vertices)
        # Shift vertices for visibility
        vertices = vertices + offset

        # draw edges
        start = vertices[panel['edges'][0]['endpoints'][0]]
        path = drawing.path(['M', start[0], start[1]],
                            stroke='black', fill='rgb(255,217,194)')
        for edge in panel['edges']:
            start = vertices[edge['endpoints'][0]]
            end = vertices[edge['endpoints'][1]]
            if ('curvature' in edge):
                control_scale = self._flip_y(edge['curvature'])
                control_point = self._control_to_abs_coord(
                    start, end, control_scale)
                path.push(
                    ['Q', control_point[0], control_point[1], end[0], end[1]])
            else:
                path.push(['L', end[0], end[1]])
        path.push('z')  # path finished
        drawing.add(path)

        # name the panel
        panel_center = np.mean(vertices, axis=0)
        text_insert = panel_center + np.array([-25, 3])
        drawing.add(drawing.text(panel_name, insert=text_insert, 
                    fill='rgb(9,33,173)', font_size='25'))
        text_max_x = text_insert[0] + 10 * len(panel_name)

        panel_center = np.mean(vertices, axis=0)
        if self.view_ids:
            # name vertices 
            for idx in range(vertices.shape[0]):
                shift = vertices[idx] - panel_center
                # last element moves pivot to digit center
                shift = 5 * shift / np.linalg.norm(shift) + np.array([-5, 5])
                drawing.add(
                    drawing.text(str(idx), insert=vertices[idx] + shift, 
                                 fill='rgb(245,96,66)', font_size='25'))
            # name edges
            for idx, edge in enumerate(panel['edges']):
                middle = np.mean(
                    vertices[[edge['endpoints'][0], edge['endpoints'][1]]], axis=0)
                shift = middle - panel_center
                shift = 5 * shift / np.linalg.norm(shift) + np.array([-5, 5])
                # name
                drawing.add(
                    drawing.text(idx, insert=middle + shift, 
                                 fill='rgb(50,179,101)', font_size='20'))

        return max(np.max(vertices[:, 0]), text_max_x), np.max(vertices[:, 1])

    def _save_as_image(self, svg_filename, png_filename):
        """
            Saves current pattern in svg and png format for visualization
        """
        if self.scaling_for_drawing is None:  # re-evaluate if not ready
            self.scaling_for_drawing = self._verts_to_px_scaling_factor()

        dwg = svgwrite.Drawing(svg_filename, profile='full')
        base_offset = [60, 60]
        panel_offset_x = 0
        heights = [0]  # s.t. it has some value if pattern is empty -- no panels

        panel_order = self.panel_order()
        for panel in panel_order:
            if panel is not None:
                panel_offset_x, height = self._draw_a_panel(
                    dwg, panel,
                    offset=[panel_offset_x + base_offset[0], base_offset[1]]
                )
                heights.append(height)

        # final sizing & save
        dwg['width'] = str(panel_offset_x + base_offset[0]) + 'px'  # using latest offset -- the most right
        dwg['height'] = str(max(heights) + base_offset[1]) + 'px'
        dwg.save(pretty=True)

        # to png
        svg_pattern = svglib.svg2rlg(svg_filename)
        renderPM.drawToFile(svg_pattern, png_filename, fmt='PNG')


    def get_stitch(self):
        """
        structure: list; each element has two dicts (a pair of seam), each dict is {'edge': 4, 'panel': 'Rfront'}
        #stitches: [[{'edge': 4, 'panel': 'Rfront'}, {'edge': 1, 'panel': 'Rback'}], [{'edge': 1, 'panel': 'Rfront'}, {'edge': 4, 'panel': 'Rback'}],
        # [{'edge': 1, 'panel': 'Lfront'}, {'edge': 4, 'panel': 'Lback'}]]
        """
        stitches = self.pattern['stitches']
        return stitches



        #get the another global pair information given the current information

    def _get_globle_stitch_panel(self, panelname):
        corresponding_panels = []
        stitches = self.pattern['stitches']
        for stitch_pair in stitches:
            stitch_one = stitch_pair[0]
            stitch_one_panel = stitch_one["panel"]

            stitch_two = stitch_pair[1]
            stitch_two_panel = stitch_two["panel"]

            if stitch_one_panel == panelname :
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

    def _get_globle_stitch_edge_index(self, panelname, original_edge_index_on_panel):
        """
       #get the position in the list of the current original_edge_index_on_panel, 2 edges of a seam should return the same value
       :param panelname: the name of the current panel
       :type panelname: str
       :param original_edge_index_on_panel: the original edge index on the panel of the current small edge
       :type original_edge_index_on_panel:  int

       """
        stitches = self.pattern['stitches']
        for index, stitch_pair in enumerate(stitches):
            stitch_one = stitch_pair[0]
            stitch_one_panel = stitch_one["panel"]
            stitch_one_edge_index = stitch_one["edge"]

            stitch_two = stitch_pair[1]
            stitch_two_panel = stitch_two["panel"]
            stitch_two_edge_index = stitch_two["edge"]
            if stitch_one_panel == panelname and stitch_one_edge_index == original_edge_index_on_panel:
                return index
            if stitch_two_panel == panelname and stitch_two_edge_index == original_edge_index_on_panel:
                return index
        return None

    def _verts_to_px_coords_garmentcode(self, vertices, translation_2d):
        """no flip: Convert given vertices and panel (2D) translation to px coordinate frame & units"""
        # Put upper left corner of the bounding box at zero
        offset = np.min(vertices, axis=0)
        vertices = vertices - offset
        translation_2d = translation_2d + offset
        return vertices, translation_2d


    def _read_a_panel(self, panel_name):
        """
        this function is for pattern got by garmentcode, and no flip the coordinate for AI
        Assumes (!!)
            that edges are correctly oriented to form a closed loop
        Returns
            the lower-right vertex coordinate for the convenice of future offsetting.
        """


        panel = self.pattern['panels'][panel_name]
        vertices = np.asarray(panel['vertices'])
        vertices, translation = self._verts_to_px_coords_garmentcode(vertices, np.array(panel['translation'][:2]))  # Only XY

        # draw edges
        start = vertices[panel['edges'][0]['endpoints'][0]]
        segs = []
        for edge in panel['edges']:
            start = vertices[edge['endpoints'][0]]
            end = vertices[edge['endpoints'][1]]
            if ('curvature' in edge):
                if isinstance(edge['curvature'], list) or edge['curvature']['type'] == 'quadratic':
                    control_scale = edge['curvature'] if isinstance(edge['curvature'], list) else edge['curvature']['params'][0]
                    control_point = self._control_to_abs_coord(start, end, control_scale)
                    segs.append(QuadraticBezier(*self.list_to_c([start, control_point, end])))
                elif edge['curvature']['type'] == 'circle':  # Assuming circle
                    # https://svgwrite.readthedocs.io/en/latest/classes/path.html#svgwrite.path.Path.push_arc
                    radius, large_arc, right = edge['curvature']['params']
                    segs.append(Arc(self.list_to_c(start), radius + 1j * radius,
                        rotation=0, large_arc=large_arc, sweep=right, end=self.list_to_c(end)))
                    # TODO Support full circle separately (?)
                elif edge['curvature']['type'] == 'cubic':
                    cps = []
                    for p in edge['curvature']['params']:
                        control_scale = p
                        control_point = self._control_to_abs_coord(start, end, control_scale)
                        cps.append(control_point)

                    segs.append(CubicBezier(*self.list_to_c([start, *cps, end])))

                else:
                    raise NotImplementedError(
                        f'{self.__class__.__name__}::Unknown curvature type {edge["curvature"]["type"]}')

            else:
                segs.append(Line(*self.list_to_c([start, end])))

        # Placement and rotation according to the 3D location
        # But flatterened on 2D
        path = Path(*segs)
        return path


def signed_area(path):
    area = 0
    for seg in path:
        if isinstance(seg, Line):
            x0, y0 = seg.start.real, seg.start.imag
            x1, y1 = seg.end.real, seg.end.imag
            area += (x1 - x0) * (y1 + y0) / 2
        elif isinstance(seg, CubicBezier):
            # Approximate the curve by dividing it into small line segments
            curve_points = [seg.start] + [seg.control1, seg.control2, seg.end]
            for i in range(len(curve_points) - 1):
                x0, y0 = curve_points[i].real, curve_points[i].imag
                x1, y1 = curve_points[i + 1].real, curve_points[i + 1].imag
                area += (x1 - x0) * (y1 + y0) / 2
        elif isinstance(seg, Arc):
            mid_point = (seg.start + seg.end) / 2
            x0, y0 = seg.start.real, seg.start.imag
            x1, y1 = seg.end.real, seg.end.imag
            area += (x1 - x0) * (y1 + y0) / 2
    return area

