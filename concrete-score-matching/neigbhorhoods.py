# in this, different neighborhood structures will be implemented

"""
Each class accepts vertices which should be integers from 0 to max value.
Using vertices, neighborhoods are built. Each vertex will have same number of elements in the neighborhood.
For example
- In cycle neighborhood 0 is connected 1, 1 is connected to 2 and so on.
- In adjacent neighborhood 0 is connected to 1 and n-1, 1 is connected to 0 and 2 and so on.
"""


class Neighborhood:
    """
        Neighborhood base class. This class contains the common methods each of the neighborhood
        will use (if any)
    """
    def build_inverse_neighborhood(self):
        """
        Docstring for build_inverse_neighborhood

        In this we will build inverse neighborhood for each vertex.
        As of now this is not optimized and it is brute force.
        """
        for xp in self.vertices:
            inv_neigbhs = []
            for x, neigbhs in self.neighborhood.items():
                if xp in neigbhs:
                    inv_neigbhs.append(x)
            self.inv_neighborhood[xp] = inv_neigbhs


class CycleNeighborhood(Neighborhood):
    def __init__(self, vertices):
        self.vertices = vertices
        self.num_neighbors = 1
        self.neighborhood = {}
        self.inv_neighborhood = {}

        # using the vertices, build the neighborhood dictionary
        self.build_neighborhood()
        # using the vertices, build the inverse neighborhood dictionary
        self.build_inverse_neighborhood()

    def build_neighborhood(self):
        # in this x is connected (x + 1) % n where n is number of vertices
        for x in self.vertices:
            neigbhs = [(x + 1) % len(self.vertices)]
            self.neighborhood[x] = neigbhs


class AdjacentNeighborhood(Neighborhood):
    def __init__(self, vertices):
        self.vertices = vertices
        self.num_neighbors = 2
        self.neighborhood = {}
        self.inv_neighborhood = {}

        # using the vertices, build the neighborhood dictionary
        self.build_neighborhood()
        # using the vertices, build the inverse neighborhood dictionary
        self.build_inverse_neighborhood()

    def build_neighborhood(self):
        num_vertices = len(self.vertices)
        for x in self.vertices:
            # x is connected to (x - 1) and (x + 1).
            neighbs = [(x - 1) % num_vertices, (x + 1) % num_vertices]
            self.neighborhood[x] = neighbs
