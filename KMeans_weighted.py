import random as rand
import math as math
import numpy as np

class KMeans_weighted:
    def __init__(self, k_clusters=8, n_init=10, maxtol=0.001, debug_=False, debug_iter_n_=200000000):
        self.k = k_clusters
        self.labels_ = []  #cluster index of each data entry
        self.centers_ = []     #means of clusters, shape(k, m), index is cluster index
        self.debug = debug_  #debug flag
        self.debug_iter_n = debug_iter_n_
        self.inertia_ = None
        self.tol = maxtol
        self.n_init_ = n_init

    
    #this method computes the initial means
    def comp_init_centers(self):
        #pick the first node at random
        tmp = np.random.rand(self.k, self.m)
        return tmp * (self.feat_bd[1] - self.feat_bd[0]) + self.feat_bd[0]


    def compute_centers_inertia(self, centers, clusters):
        means = np.zeros((self.k, self.m))
        denum = np.zeros((self.k, self.m))
        inertia = 0

        for i in range(self.n):
            cid = clusters[i]
            means[cid] += self.x[i] * self.w[i]
            denum[cid] += self.w[i]
            inertia +=  self.distance_sqr(self.x[i], centers[cid]) * self.w[i]

        new_centers = means / denum

        #examine non valid centers 
        has_regen = False
        for i in range(self.k):
                if np.isnan(new_centers[i]).any():
                    # null center, random a new center
                    new_centers[i] = np.random.rand(self.m) * (self.feat_bd[1] - self.feat_bd[0]) + self.feat_bd[0]
                    has_regen = True
                    if self.debug:
                        print "center -", i, "is null, regenerate to be:", new_centers[i]


        return new_centers, inertia, has_regen


    #this method assign nodes to the cluster with the smallest mean
    def assign_points(self, centers):
        clusters = np.zeros(self.n).astype(np.int)

        for j in range(self.n):
            #find the best cluster for this node
            min_dist = None
            min_clst = None

            for i in range(self.k):
                cur_dist = self.distance_sqr(self.x[j], centers[i])
                if min_dist == None or min_dist > cur_dist:
                    min_dist = cur_dist
                    min_clst = i

            clusters[j] = min_clst

        return clusters

    def distance_sqr(self, a, b):
        return np.square(a - b).sum()

    #k_means algorithm
    def fit(self, X, weights=None):
        self.x = X
        self.n, self.m = X.shape[0], X.shape[1]
        if weights is None:
            self.w = np.ones(self.n)
        else:
            self.w = weights
        
        self.find_feature_boundary()
        centers = None
        inertia = None
        clusters = None

        for tt in range(0, self.n_init_):
            centers1 = centers = self.comp_init_centers()
            iter_n = 0
            stop = False
            inertia = None

            while not stop:

                #assignment step: assign each node to the cluster with the closest mean
                clusters = self.assign_points(centers1)
                centers = centers1

                centers1, inertia1, has_regen = self.compute_centers_inertia(centers, clusters)

                stop = inertia and abs((inertia1 - inertia) / inertia) < self.tol and not has_regen
                inertia = inertia1

                iter_n += 1
                if self.debug:
                    print tt, ".", iter_n, "-------------------------RESULTS:"
                    #self.print_centers(centers)
                    #self.print_clusters(clusters)
                    print "inertia =", inertia
                    print "stop =", stop, "     has_regen =", has_regen
                    if iter_n > self.debug_iter_n:
                        break

            if (not self.inertia_) or inertia < self.inertia_:
                self.inertia_ = inertia
                self.centers_ = centers
                self.labels_ = clusters

        if self.debug:
            print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print "clustering finish, output results:"
            print "final inertia =", self.inertia_
            print self.print_clusters(self.labels_)
            print self.print_centers(self.centers_)
            print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"


    def find_feature_boundary(self):
        # first row is min of each feature, second is max.
        self.feat_bd = np.empty((2, self.m)) 
        self.feat_bd[0] = np.amin(self.x, axis=0)
        self.feat_bd[1] = np.amax(self.x, axis=0)


    #debug function: print cluster points
    def print_clusters(self, clusters):
        for i in range(self.n):
            print self.x[i], "--clst-->", clusters[i]

    #print centers
    def print_centers(self, centers):
        for i in range(self.k):
            print "cluster ", i, ":", centers[i]