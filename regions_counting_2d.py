
import numpy as np

#-------------------------------------------------------------------------------------------------------------------#
#---------------------------------- Implementation borrows from Hanin and Rolnick ----------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

"""
This file has all the tools needed for counting regions and transitions over a 2D plane.
"""


class LinearRegion2D:
    def __init__(self, fn_weight, fn_bias, vertices, next_layer_off):
        self._fn_weight = fn_weight
        self._fn_bias = fn_bias
        self._vertices = vertices
        self._num_vertices = len(vertices)
        self._next_layer_off = next_layer_off

    def get_new_regions(self, new_weight_n, new_bias_n, n):
        weight_n = np.dot(self._fn_weight, new_weight_n)
        bias_n = np.dot(self._fn_bias, new_weight_n) + new_bias_n
        vertex_images = np.dot(self._vertices, weight_n) + bias_n
        is_pos = (vertex_images > 0)
        is_neg = np.logical_not(is_pos)  # assumes that distribution of bias_n has no atoms
        if np.all(is_pos):
            return [self]
        elif np.all(is_neg):
            self._next_layer_off.append(n)
            return [self]
        else:
            pos_vertices = []
            neg_vertices = []
            for i in range(self._num_vertices):
                j = np.mod(i + 1, self._num_vertices)
                vertex_i = self.vertices[i, :]
                vertex_j = self.vertices[j, :]
                if is_pos[i]:
                    pos_vertices.append(vertex_i)
                else:
                    neg_vertices.append(vertex_i)
                if is_pos[i] == ~is_pos[j]:
                    intersection = intersect_lines_2d(weight_n, bias_n, vertex_i, vertex_j)
                    pos_vertices.append(intersection)
                    neg_vertices.append(intersection)
            pos_vertices = np.array(pos_vertices)
            neg_vertices = np.array(neg_vertices)
            next_layer_off0 = list(np.copy(self._next_layer_off))
            next_layer_off1 = list(np.copy(self._next_layer_off))
            next_layer_off0.append(n)
            region0 = LinearRegion2D(self._fn_weight, self._fn_bias, neg_vertices, next_layer_off0)
            region1 = LinearRegion2D(self._fn_weight, self._fn_bias, pos_vertices, next_layer_off1)
            return [region0, region1]

    def next_layer(self, new_weight, new_bias):
        self._fn_weight = np.dot(self._fn_weight, new_weight.T)
        self._fn_bias = np.dot(self._fn_bias, new_weight.T) + new_bias
        self._fn_weight[:, self._next_layer_off] = 0
        self._fn_bias[self._next_layer_off] = 0
        self._next_layer_off = []

    @property
    def vertices(self):
        return self._vertices

    @property
    def fn_weight(self):
        return self._fn_weight

    @property
    def fn_bias(self):
        return self._fn_bias

    @property
    def dead(self):
        return np.all(np.equal(self._fn_weight, 0))


def get_sample_plane(X):
    indices = np.random.choice(len(X), 3, replace=False)
    vert0 = X[indices[0], :]
    vert1 = X[indices[1], :]
    vert2 = X[indices[2], :]
    side0 = vert1 - vert2
    side1 = vert2 - vert0
    side2 = vert0 - vert1
    cos0 = -np.dot(side1, side2) / (np.linalg.norm(side1) * np.linalg.norm(side2))
    cos1 = -np.dot(side2, side0) / (np.linalg.norm(side2) * np.linalg.norm(side0))
    cos2 = -np.dot(side0, side1) / (np.linalg.norm(side0) * np.linalg.norm(side1))
    sin0 = np.sqrt(1 - cos0 ** 2)
    """ Every triangle can be inscribed in some circle + inscribed angle theorem """
    circumradius = 0.5 * np.linalg.norm(side0) / sin0  # law of sines
    # Parallelogram with radius 0 as diagonal, sides along sides 1 and 2
    proj1 = (cos2 / sin0) * circumradius * (side1 / np.linalg.norm(side1))
    proj2 = (cos1 / sin0) * circumradius * (side2 / np.linalg.norm(side2)) ###HERE
    circumcenter = vert0 + proj1 - proj2
    square_center = circumcenter
    # Scale so that the square is slightly bigger than the circumcircle
    square_vec_1 = (proj2 - proj1) * 1.25
    unit_vec_1 = square_vec_1 / np.linalg.norm(square_vec_1)
    square_vec_2 = (side2 - np.dot(side2, unit_vec_1) * unit_vec_1)
    square_vec_2 = square_vec_2 * (circumradius * 1.25 / np.linalg.norm(square_vec_2))
    sq_norm1 = np.linalg.norm(square_vec_1) ** 2
    sq_norm2 = np.linalg.norm(square_vec_2) ** 2
    x0 = np.dot(vert0 - circumcenter, square_vec_1) / sq_norm1
    y0 = np.dot(vert0 - circumcenter, square_vec_2) / sq_norm2
    x1 = np.dot(vert1 - circumcenter, square_vec_1) / sq_norm1
    y1 = np.dot(vert1 - circumcenter, square_vec_2) / sq_norm2
    x2 = np.dot(vert2 - circumcenter, square_vec_1) / sq_norm1
    y2 = np.dot(vert2 - circumcenter, square_vec_2) / sq_norm2
    return np.array([square_vec_1, square_vec_2]), circumcenter, [x0, x1, x2], [y0, y1, y2]


def count_regions_2d(the_weights, the_biases, input_fn_weight, input_fn_bias,
                     input_vertices=np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]),
                     return_regions=False, eval_bounded=False, consolidate_dead_regions=False):
    # Note: eval_bounded requires that input_vertices is a rectangle.
    regions = [LinearRegion2D(input_fn_weight, input_fn_bias, input_vertices, [])]
    depth = len(the_weights)
    for k in range(depth):
        for n in range(the_biases[k].shape[0]):
            new_regions = []
            for region in regions:
                new_regions = new_regions + region.get_new_regions(the_weights[k][n, :], the_biases[k][n], n)
            regions = new_regions
        for region in regions:
            region.next_layer(the_weights[k], the_biases[k])
    if consolidate_dead_regions:
        raise NotImplementedError
    if eval_bounded:
        bounded_regions = []
        unbounded_regions = []
        mins = np.min(input_vertices, axis=0)
        maxs = np.max(input_vertices, axis=0)
        for region in regions:
            verts = region.vertices
            if ((mins[0] == np.min(verts, axis=0)[0]) or (mins[1] == np.min(verts, axis=0)[1])
                or (maxs[0] == np.max(verts, axis=0)[0]) or (maxs[1] == np.max(verts, axis=0)[1])):
                unbounded_regions.append(region)
            else:
                bounded_regions.append(region)
    if return_regions:
        if eval_bounded:
            return bounded_regions, unbounded_regions
        else:
            return regions
    else:
        if eval_bounded:
            return len(bounded_regions), len(unbounded_regions)
        else:
            return len(regions)

def intersect_lines_2d(line_weight, line_bias, pt1, pt2):
    t = (np.dot(pt1, line_weight) + line_bias) / np.dot(pt1 - pt2, line_weight)
    return pt1 + t * (pt2 - pt1)