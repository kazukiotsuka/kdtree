#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# core/tasks/spatial_index/kd_tree.py
#
#   Nearest neighbor search in computational geometry
#

import numpy as np
from nose.tools import eq_, ok_, with_setup
import json
from bson import ObjectId
import sys
from modelbase import ModelBase



class Object(ModelBase):
    __structure__ = {
        '_id': ObjectId,
        'index': int,
        'name': str,
        'coordinate': tuple,
        'radius': int
    }
    __required_fields__ = ['_id', 'index', 'coordinate', 'radius']
    __default_values__ = {}
    __validators__ = {}


def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


class KDTree():

    def __init__(self):
        return

    def coordinates(self, objects) -> np.array:
        """Object list to coordinate array.

        args:
            - objects (list of Object)
        returns:
            - coordinates (np.array) : size of (length of objects, 2)

            e.g.
            [[(8.80,35.22), (80.11,54.24), (66.70,64.69), (141.17,77.73), (25.09,87.78), (101.10,91.35), (48.85,26.03), (134.72,23.11)]
        """
        coordinates = np.zeros((len(objects), 2))
        for obj in objects:
            coordinates[obj.index-1] = obj.coordinate
        print(f'\nlist of objects\'s coordinate ->\n{coordinates}')
        return coordinates

    def create_tree(self, objects) -> (np.array, np.array):
        """Create kd Tree structured json data.

        args:
            - objects (list of Object instance) : list of object

        returns:
            - tree (np.array) : kd tree structured json
            - coordinates (np.array) : list of nodes's (x, y)
        """

        # create_coordinate
        print(f'\n===============================================================')
        coordinates = self.coordinates(objects)
        print(f'(x, y) of objects =>\n{coordinates}')

        # initialize tree
        tree = np.zeros(10000)  # 10000 zeros == 80Kb
        print(f'\n===============================================================')
        print(f'create tree in maximum {len(tree)} size array')
        # generate tree
        self.recursive_split(objects, 'x', 1, tree, verbose=True)

        def remove_last_zeros(tree):
            c = 0
            for i, elem in enumerate(tree[::-1]):
                if elem != 0:
                    break
                else: c += 1
            return tree[1:-c]

        tree = remove_last_zeros(tree)
        #tree = tree[tree !=0]
        print(f'\n\nkd Tree created => \n{tree}\n\n')
        return tree, coordinates

    def recursive_split(self, group: list, axis: str = 'x', pos: int = 1, tree: np.array = None, verbose=False):
        """Split group of object by x or y axis recursively.

        args:
            - group (list of Object) :
            - axis ('x' or 'y') : split axis
            - index (int) : index in binary tree
            - tree (np.array) : binary tree like [1 | 2  3 | 4 5 6 7 | ...]

               x1     y2    y3     x4    x5    x6      x7      N1  N7  N5  N3  N2  N8  N6  N4
            [ 75.8 | 53.4  61.6 | 28.8  45.8  107.4  121.1  |  1.  7.  5.  3.  2.  8.  6.  4.]

                 1     2     3      4      5     6      7        8     9    10    11  12  13    14  15   16  17  18   19  20  21   22   23    24  25  26  27   28
                x1    y2    y3     x4     x5    x6     x7       y8    y9   y10   y11  N9  N8   y14  N4   N2  N1  N7  N10  N3  N5  N14  N11   x23  N6     N12  N13
            [ 62.9 | 54.3  58.6 | 20.7  28.7  119.3  111.0  | 23.0  33.6  68.4  92.3  9.  8.  73.3  4. | 2.  1.  7.  10.  3.  5.  14.  11.  96.8  6.  0  12.  13.]


            binary tree position  *rule: (left->x2, right->x2+1)
                         1
                    2         3
                 4    5     6    7
               8  9 10 11 12 13 14 15

        """
        if verbose: print(f'---------------------------------------------------------------')
        if verbose: print(f'pos: {pos}, split axis: {axis}')
        if verbose: print(f'{len(group)} objects in group found')

        if axis not in ('x', 'y'):
            raise RuntimeError(f'{axis} is invalid axis')

        x_or_y = 0 if axis == 'x' else 1
        next_axis = 'y' if axis == 'x' else 'x'
        if len(group) == 0:
            if verbose: print(f'[({pos})|{tree[pos]}]')
        elif len(group) == 1:
            tree[pos] = group[0].index
            if verbose: print(f'write to {pos}, {pos+1}')
            if verbose: print(f'[({pos})|{tree[pos]}] | [({pos + 1})|{tree[pos+1]}]')
        elif len(group) == 2:
            val_split = np.array([object.coordinate[x_or_y] for object in group]).mean()
            n1 = [object for object in group if object.coordinate[x_or_y] <= val_split][0]
            n2 = [object for object in group if object.coordinate[x_or_y] > val_split][0]
            tree[pos] = val_split
            tree[pos * 2] = n1.index
            tree[pos * 2 + 1] = n2.index
            if verbose: print(f'write to {pos}, {pos*2}, {pos*2+1}')
            if verbose: print(f'[({axis}{pos})|{tree[pos]}]')
            if verbose: print(f'[({pos*2})|{tree[pos*2]}] | [({pos*2+1})|{tree[pos*2+1]}]')
        else:
            val_split = np.array([object.coordinate[x_or_y] for object in group]).mean()
            tree[pos] = val_split
            if verbose: print(f'write to {pos}')
            if verbose: print(f'[({axis}{pos})|{tree[pos]}]')
            group_A = [object for object in group if object.coordinate[x_or_y] <= val_split]
            group_B = [object for object in group if object.coordinate[x_or_y] > val_split]
            if verbose: print(f'{len(group_A)} nodes <= {axis}{pos}:{val_split:.1f} < {len(group_B)} nodes')
            if verbose: print(f'{[obj.index for obj in group_A]} | {[obj.index for obj in group_B]}')
            return self.recursive_split(group_A, next_axis, pos * 2, tree, verbose), \
                   self.recursive_split(group_B, next_axis, pos * 2 + 1, tree, verbose)


    def split(self, group: list, axis='x'):
        """Split group of object by x or y axis.

        args:
            - group (list of Object) :
            - axis ('x' or 'y') : split axis
        """
        if axis == 'x':
            x = np.array([object.coordinate[0] for object in group]).mean()
            group_A = [object for object in group if object.coordinate[0] <= x]
            group_B = [object for object in group if object.coordinate[0] > x]
            return group_A, group_B, x
        elif axis == 'y':
            y = np.array([object.coordinate[0] for object in group]).mean()
            group_A = [object for object in group if object.coordinate[1] <= y]
            group_B = [object for object in group if object.coordinate[1] > y]
            return group_A, group_B, y
        else:
            raise RuntimeError(f'axis {axis} is invalid')

    def search_in_range(self, tree, coordinates, position, r=10):
        """Search nodes in range from kd-Tree.

            N1
            ○_______________
        ry1 | N3 ○  |       |
            |  ○ N2 r       |
            |       |       |
            |<- r ->x<- r ->|  N4
            |       |       | ○
            |       r       |
        ry1 |_______|_______|
           rx1             rx2

        args:
            - tree (np.array) : kd tree structured array
            - coordinates (np.array) : list of nodes's (x, y)
            - position (tuple) : (x, y) for current position
            - r (int) : search range distance [m]

        returns:
            - nearest_node (dict): object
        """

        # extends tree e.g. [1 | 2, 3 | 0] -> [1 | 2, 3 | 0, 0, 0, 0]
        tree = np.append(tree, np.zeros(1000))

        # current position and search range
        current_x = position[0]
        current_y = position[1]
        rx1 = current_x - r
        rx2 = current_x + r
        ry1 = current_y - r
        ry2 = current_y + r

        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(f'\n\nsearch nearest nodes\n\n')
        print(f'    tree {tree}')
        print(f'    coordinates {coordinates}')
        print(f'    current (x, y)|({current_x}, {current_y})\n')
        print(f'rx1:{rx1:.1f} --------- rx2:{rx2:.1f}\n')
        print(f'ry2:{ry2:.1f}')
        print(f'|\n|\n|\n|\n|\n|')
        print(f'ry1:{ry1:.1f}')

        results = []
        self.binary_search('x', 1, rx1, rx2, ry1, ry2, tree, coordinates, results, verbose=True)
        print(f'\n\n  binary search result => {results}\n\n')

        return results

    def binary_search(
            self, axis='x', pos=1, rx1=0.0, rx2=0.0, ry1=0.0, ry2=0.0,
            tree: np.array = None, coordinates: np.array = None, results=[], verbose=False):
        """

        args:
            - pos (int) : position in binary tree. starts from 1.

            binary tree position  *rule: (left->x2, right->x2+1)
                         1
                    2         3
                 4    5     6    7
               8  9 10 11 12 13 14 15

            - tree (np.array) : kd tree structured array

           pos   1     2     3      4      5     6      7        8     9    10    11  12  13    14  15   16  17  18   19  20  21   22   23    24  25  26  27   28
                x1    y2    y3     x4     x5    x6     x7       y8    y9   y10   y11  N9  N8   y14  N4   N2  N1  N7  N10  N3  N5  N14  N11   x23  N6     N12  N13
            [ 62.9 | 54.3  58.6 | 20.7  28.7  119.3  111.0  | 23.0  33.6  68.4  92.3  9.  8.  73.3  4. | 2.  1.  7.  10.  3.  5.  14.  11.  96.8  6.  0  12.  13.]
               0      1     2      3    ... (pos-1) ...

           *make sure that "position" and "index of array" is different.

        returns:
            - results (list) : list of [index of node, (x, y)]
                e.g.
                [[4., (35.12, 44.9)], [2., (18.25, 89.12)],...]
        """
        if verbose: print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if verbose: print(f'pos: {pos}, axis: {axis}')

        if axis not in ('x', 'y'):
            raise RuntimeError(f'{axis} is invalid axis')

        # if no child, check if the node is in range.
        if tree[pos*2-1] == 0:
            node_index = tree[pos-1]
            print(f'({pos})|{node_index} has no child. (leaf)')
            if node_index == 0:
                raise RuntimeError(f'({pos})|{node_index} must not be 0.')
            if not tree[pos-1].is_integer():
                # check if the value is interger num (e.g. 4.0)
                raise RuntimeError(f'({pos})|{node_index} must be interger number.')
            node_x = coordinates[int(node_index)-1][0]
            node_y = coordinates[int(node_index)-1][1]
            if rx1 < node_x and node_x < rx2 and ry1 < node_y and node_y < ry2:
                print(f'node {int(node_index)} ({node_x}, {node_y}) is in range.')
                results.append([node_index, (node_x, node_y)])
                print(f'\nnode {int(node_index)} {[node_index, (node_x, node_y)]} is saved!!\n')
            else:
                print(f'node {int(node_index)} ({node_x}, {node_y}) is out of range.')

        # if children, it's x or y split line.
        else:
            next_axis = 'y' if axis == 'x' else 'x'
            split_x_or_y = tree[pos-1]
            print(f'({pos})|{axis}:{split_x_or_y:.1f} has children. (split line)')
            r1 = rx1 if axis is 'x' else ry1
            r2 = rx2 if axis is 'x' else ry2
            print(f'r1:{r1:.1f} |---- {split_x_or_y:.1f} ---| r2:{r2:.1f}')
            if r1 < split_x_or_y:
                # go to the left
                print(f'pos({pos}) r{axis}1|{r1:.1f} < {axis}|{split_x_or_y:.1f}. go to left child -> pos ({pos*2}).')
            if split_x_or_y < r2:
                # go to the right
                print(f'pos({pos}) {axis}|{split_x_or_y:.1f} < r{axis}2|{rx2}. go to right child. -> pos ({pos*2+1})')
            # to arrange stdout properly..
            if r1 < split_x_or_y:
                self.binary_search(next_axis, pos * 2, rx1, rx2, ry1, ry2, tree, coordinates, results, verbose)
            if split_x_or_y < r2:
                self.binary_search(next_axis, pos*2+1, rx1, rx2, ry1, ry2, tree, coordinates, results, verbose)
            #else:
            #    print(f'{axis}|{split_x_or_y:.1f} is not in range of [r{axis}1|{r1:.1f}, r{axis}2|{r2:.1f}]. ignore children.')

    def proximity_detection(self, position: tuple, search_results: list, objects: list) -> Object:
        """Proximity detection.

        1. calculate distances for each objects
        2. to evaluate proximity of the nearest object by the radius
        3. if not in range, evaluate the second nearest object.
        4. continue until it gets proximate object, or check all.

        args:
            - position (tuple) : (x, y) for current position
            - search_results (list) : e.g. [[10.0, (28.87, 49.33)], [3.0, (28.18, 61.13)]]
            - objects (list of Object) :

        returns:
            - object (Object) : proximate object
        """
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('\nproximity detection\n')

        def euclidean_distance(p1:list, p2:list):
            return np.linalg.norm(np.array(p1)-np.array(p2))

        print(f'\n  center {position}\n')

        # calculate distances
        node_indices = []
        distances = []
        for i, result in enumerate(search_results):
            node_index = result[0]
            node_center = result[1]
            print(f'\n{i}: node{int(node_index)} {node_center}')
            d = euclidean_distance(position, node_center)
            node_indices.append(int(node_index))
            distances.append(d)
            print(f'distance => {d}')

        # evaluate proximity
        print('----------------------------------------------------')
        for i in np.argsort(distances):
            distance = distances[i]
            node_index = int(node_indices[i])
            obj = [obj for obj in objects if obj.index == node_index][0]
            if distance < obj.radius:
                print(f'\nobject {node_index}: distance {distance} < radius {obj.radius}')
                print(f'******** proximity detected ********\n')
                break
            else:
                print(f'\nobject {node_index}: distance {distance} > radius {obj.radius}')
                print(f'not in range. pass\n')






#### unittest

def setUp():
    print('unittest is now setup')

def tearDown():
    print('unittest is now teardown')

class TestFixture():
    # in 100m x 160m
    coordinates_A = [(8.80,35.22), (80.11,54.24), (66.70,64.69), (141.17,77.73), (25.09,87.78), (101.10,91.35), (48.85,26.03), (134.72,23.11)]
    coordinates_B = [(2.97,37.81), (9.22,8.28), (28.18,61.13), (149.59,79.67), (13.11,75.80), (101.10,91.35), (42.05,17.94), (145.40,22.14), (93.28,30.04), (28.87,49.33), (31.16,95.73), (83.36,60.55), (110.31,68.24), (42.48,89.01)]
    radiuses_A = [3, 3, 3, 3, 3, 3, 24, 3]
    radiuses_B = [3, 3, 6, 3, 3, 3, 3, 3, 3, 10, 3, 3, 3, 3]

    @classmethod
    def create_objects(cls, coordinates, radiuses):
        objects = []
        print('create object set 1')
        for i in range(len(coordinates)):
            objects.append(
                Object({'_id': ObjectId(), 'index': i+1, 'name': f'n{i+1}', 'coordinate': coordinates[i], 'radius': radiuses[i]})
            )
            print(objects[-1])
        return objects


if __name__ == "__main__":
    # create objects
    objects_A = TestFixture.create_objects(TestFixture.coordinates_A, TestFixture.radiuses_A)
    objects_B = TestFixture.create_objects(TestFixture.coordinates_B, TestFixture.radiuses_B)

    kd_tree = KDTree()

    # problem A
    current_position = (25, 25)
    search_range = 30
    tree_A, coordinates_A = kd_tree.create_tree(objects_A)
    results = kd_tree.search_in_range(tree_A, coordinates_A, current_position, r=search_range)
    kd_tree.proximity_detection(current_position, results, objects_A)

    # problem B
    #current_position = (48, 43)
    current_position = (33, 55)
    search_range = 20
    tree_B, coordinates_B = kd_tree.create_tree(objects_B)
    results = kd_tree.search_in_range(tree_B, coordinates_B, current_position, r=search_range)
    kd_tree.proximity_detection(current_position, results, objects_B)
