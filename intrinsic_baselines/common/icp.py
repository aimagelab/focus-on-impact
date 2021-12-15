import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(a, b, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    B stays fixed and A is transformed
    """

    # assert A.shape == B.shape

    # get number of dimensions
    m = a[0].shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = [np.ones((m + 1, pc.shape[0])) for pc in a]
    dst = [np.ones((m + 1, pc.shape[0])) for pc in b]
    for i in range(len(a)):
        src[i][:m, :] = np.copy(a[i].T)
        dst[i][:m, :] = np.copy(b[i].T)

    # apply the initial pose estimation
    if init_pose is not None:
        for i in range(len(a)):
            src[i] = np.dot(init_pose[i], src[i])

    prev_errors = np.zeros(len(a))
    continue_masks = np.zeros(len(a)).astype(np.bool)

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = [], []
        T = np.zeros((len(a), m + 1, m + 1))

        for idx in range(len(a)):

            if continue_masks[idx]:  # if tolerance is satisfied skip transformation
                distances.append(np.zeros(1))
                continue

            dist, ind = nearest_neighbor(src[idx][:m, :].T, dst[idx][:m, :].T)
            distances.append(dist)

            # compute the transformation between the current source and nearest destination points
            trans, _, _ = best_fit_transform(src[idx][:m, :].T, dst[idx][:m, ind].T)
            T[idx] = trans

            # update the current source
            src[idx] = np.dot(trans, src[idx])

        # check error
        mean_errors = np.array([np.mean(dist) for dist in distances])
        new_continue_masks = np.abs(prev_errors - mean_errors) < tolerance
        continue_masks = new_continue_masks + continue_masks

        if np.all(continue_masks):
            break

        prev_errors = mean_errors

    # calculate final transformations
    T = np.zeros((len(a), m + 1, m + 1))
    for idx in range(len(a)):
        trans, _, _ = best_fit_transform(a[idx], src[idx][:m, :].T)
        T[idx] = trans

    return T, distances, i
