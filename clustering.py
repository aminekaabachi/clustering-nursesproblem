from __future__ import division
import pandas as pd
import numpy as np
from collections import namedtuple
from random import random, choice
from copy import copy
from sklearn import manifold
from functools import reduce 
import math

class City:
    __slots__ = ["x", "y", "id", "group"]
    def __init__(self, x=0.0, y=0.0, id=0, group=0):
        self.x, self.y, self.id, self.group = x, y, id, group

class Visit:
    __slots__ = ["id", "citizen_id", "locked_nurse", "locked", "competence", "early_start", "duration", "latest_end", "city", "preferred", "group"]
    def __init__(self, id, citizen_id, locked_nurse, locked, competence, early_start, duration, latest_end, city, preferred, group):
        self.id, self.citizen_id, self.locked_nurse, self.locked, self.competence, self.early_start, self.duration, self.latest_end, self.city, self.preferred, self.group = id, citizen_id, locked_nurse, locked, competence, early_start, duration, latest_end, city, preferred, group
   
class Nurse:
    __slots__ = ["id", "start_location", "excluded_citizens", "competence", "start_duty", "end_duty", "timespan"]
    def __init__(self, id=0, start_location=0, excluded_citizens=[], competence=[], start_duty=0, end_duty=0):
        self.id = id
        self.start_location = start_location
        self.excluded_citizens = excluded_citizens
        self.competence = competence
        self.start_duty = start_duty
        self.end_duty = end_duty
        self.timespan = end_duty-start_duty

def distribution():
    """building the similarity matrix from hh_dist.dat"""
    raw = []
    with open('hh_dist.dat','r') as f:
        for line in f:
                    l = line.strip().split(',')
                    del l[-2]
                    raw.append(l)

    data = pd.DataFrame(raw,columns = ['row','column','value'])
    data_ind = data.set_index(['row','column']).unstack('column')
    matrix = np.array(data_ind.values,dtype=float)
    where_are_NaNs = np.isnan(matrix)
    matrix[where_are_NaNs] = 0

    """Applying Multidimensional scaling on the similarity matrix"""
    def symmetrize(a):
        return a + a.T - np.diag(a.diagonal())

    mds = manifold.MDS(n_jobs=-1, random_state=1337, dissimilarity='precomputed')
    pos_e = mds.fit(symmetrize(matrix)).embedding_
    return (pos_e, data_ind)
 
FLOAT_MAX = 1e100
 

def generate_cities():
    (pos_e, data_ind) = distribution()
    indexes = data_ind.index.values
    cities = [City() for _ in xrange(len(indexes)) ]

    i = 0
    for p in cities:
        p.id = int(indexes[i])
        p.x = pos_e[i][0]
        p.y = pos_e[i][1]
        i += 1 
    return cities
 
 
def nearest_cluster_center(City, cluster_centers):
    """Distance and index of the closest cluster center"""
    def sqr_distance_2D(a, b):
        return (a.x - b.x) ** 2  +  (a.y - b.y) ** 2
 
    min_index = City.group
    min_dist = FLOAT_MAX
 
    for i, cc in enumerate(cluster_centers):
        d = sqr_distance_2D(cc, City)
        if min_dist > d:
            min_dist = d
            min_index = i
 
    return (min_index, min_dist)
 
 
def kpp(cities, cluster_centers):
    cluster_centers[0] = copy(choice(cities))
    d = [0.0 for _ in xrange(len(cities))]
 
    for i in xrange(1, len(cluster_centers)):
        sum = 0
        for j, p in enumerate(cities):
            d[j] = nearest_cluster_center(p, cluster_centers[:i])[1]
            sum += d[j]
 
        sum *= random()
 
        for j, di in enumerate(d):
            sum -= di
            if sum > 0:
                continue
            cluster_centers[i] = copy(cities[j])
            break
 
    for p in cities:
        p.group = nearest_cluster_center(p, cluster_centers)[0]
 
 
def lloyd(cities, nclusters):
    cluster_centers = [City() for _ in xrange(nclusters)]
 
    # call k++ init
    kpp(cities, cluster_centers)
 
    lenpts10 = len(cities) >> 10
 
    changed = 0
    while True:
        # group element for centroids are used as counters
        for cc in cluster_centers:
            cc.x = 0
            cc.y = 0
            cc.group = 0
 
        for p in cities:
            cluster_centers[p.group].group += 1
            cluster_centers[p.group].x += p.x
            cluster_centers[p.group].y += p.y
 
        for cc in cluster_centers:
            cc.x /= cc.group
            cc.y /= cc.group
 
        # find closest centroid of each CityPtr
        changed = 0
        for p in cities:
            min_i = nearest_cluster_center(p, cluster_centers)[0]
            if min_i != p.group:
                changed += 1
                p.group = min_i
 
        # stop when 99.9% of cities are good
        if changed <= lenpts10:
            break
 
    for i, cc in enumerate(cluster_centers):
        cc.group = i
 
    return cluster_centers

     
def make_clusters(cities, cluster_centers):
    visits = []
    with open("hh_visit.dat") as f:
        for line in f:
            l = line.strip().split(',')
            id = int(l[0])
            citizen_id = int(l[2])
            locked_nurse = int(l[1])
            locked = int(l[3])
            competence = int(l[5])
            early_start = int(l[7])
            duration = int(l[9])
            latest_end = int(l[8])
            city = int(l[14])
            preferred = [int(l[15]), int(l[16]), int(l[17])]
            group = next(p.group for p in cities if (p.id == city))
            visits.append(Visit(id, citizen_id, locked_nurse, locked, competence, early_start, duration, latest_end, city, preferred, group))

    clusters = []
    for i, cc in enumerate(cluster_centers):
        clusters.append([])
        for v in visits:
            if v.group != i:
                continue
            clusters[i].append(v)

    return visits,clusters


def fill_nurses():
    nurses = []
    with open("hh_nurse.dat") as f:
        for line in f:
            l = line.strip().split(',')
            id = int(l[0])
            start_location = int(l[5])
            excluded_citizens = [int(l[6]),int(l[7])]
            competence = [int(l[i]) for i in range(8,17)]
            start_duty = int(l[17])
            end_duty = int(l[18])
            if start_duty == 0 and end_duty == 0:
                continue
            nurse = Nurse(id, start_location, excluded_citizens, competence, start_duty, end_duty)
            nurses.append(nurse)
    return nurses

assigned_nurses = []

def assignements(nurses, cluster):
    matrix = np.zeros((len(cluster), len(nurses)))
    global assigned_nurses
    for i,v in enumerate(cluster):
        for j,n in enumerate(nurses):
            if n.id in assigned_nurses:
                matrix[i][j] = 0 
            elif v.locked:
                if v.locked_nurse == n.id:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0
            elif v.citizen_id in n.excluded_citizens:
                matrix[i][j] = 0
            elif n.competence[v.competence-1] == 0:
                matrix[i][j] = 0
            else:
                matrix[i][j] = n.competence[v.competence-1]
                if n.id in v.preferred:
                    matrix[i][j] += 1
		m = max(matrix[i])
		assigned_nurses.extend([nurses[i].id for i, j in enumerate(matrix[i]) if j == m])
        assigned_nurses = list(set(assigned_nurses))
    print matrix    


    
def main():
    k = 5
    cities = generate_cities()
    cluster_centers = lloyd(cities, k)
    (visits, clusters) = make_clusters(cities, cluster_centers)
    nurses = fill_nurses()

    for c in clusters:
        assignements(nurses, c)
        
    
main()
