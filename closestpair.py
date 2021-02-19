import random
import math
import functools


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tic = time.time()
        for _ in range(100):
            results = func(*args, **kwargs)
        toc = time.time()
        print(f"time for {func.__name__} for 100 times is {toc - tic}")
        return results
    return wrapper

class Point:
    def __init__(self, x = None, y = None):
        assert isinstance(x, (int, float, type(None))) and isinstance(y, (int, float, type(None)))
        self.x = random.randint(0, 100) if x is None else x
        self.y = random.randint(0, 100) if y is None else y

    def dist(self, pt2):
        return math.sqrt((self.x - pt2.x) ** 2 + (self.y - pt2.y) ** 2)

    def dist_x(self, pt2):
        return abs(self.x - pt2.x)
    
    def dist_y(self, pt2):
        return abs(self.y - pt2.y)

    def __str__(self):
        return f"({self.x}, {self.y})"


@timeit
def get_closest_pair(pts):
    pts_by_x = sorted(pts, key=lambda pt: pt.x)
    pts_by_y = sorted(pts, key=lambda pt: pt.y)
    pair, _ = _get_closest_pair(pts_by_x, 0, len(pts_by_x) -1,  pts_by_y)
    return pair


def _brute_force_closest(pts):
    assert len(pts) >= 2
    min_dist = 1e15
    pair = None
    for i in range(len(pts) - 1):
        for j in range(i + 1, len(pts)):
            dist = pts[i].dist(pts[j])
            if dist < min_dist:
                pair = (pts[i], pts[j])
                min_dist = dist

    assert pair is not None
    return pair, min_dist

@timeit
def brute_force_closest(pts):
    pair, _ = _brute_force_closest(pts)
    return pair

def _get_closest_pair(pts_by_x, x_beg, x_end, pts_by_y):
    if (x_end - x_beg + 1) <= 4:
        return _brute_force_closest(pts_by_x[x_beg: x_end + 1])

    x_mid = x_beg + (x_end - x_beg) // 2
    x_set = set(pts_by_x[x_beg: x_end + 1])
    pts_by_y = [pt for pt in pts_by_y if pt in x_set]

    pair_left, sigma_left = _get_closest_pair(
        pts_by_x, x_beg, x_mid, pts_by_y)
    pair_right, sigma_right = _get_closest_pair(
        pts_by_x, x_mid + 1, x_end, pts_by_y)

    if sigma_left < sigma_right:
        pair = pair_left
        sigma = sigma_left
    else:
        pair = pair_right
        sigma = sigma_right

    pt_c = pts_by_x[x_mid]
    pts_set = {pt_c}

    i = x_mid - 1
    while i >= x_beg and pts_by_x[i].dist_x(pt_c) <= sigma:
        pts_set.add(pts_by_x[i])
        i -= 1
    i = x_mid + 1
    while i <= x_end and pts_by_x[i].dist_x(pt_c) < sigma:
        pts_set.add(pts_by_x[i])
        i += 1
    
    pts_by_y = [pt for pt in pts_by_y if pt in pts_set]
    for i in range(len(pts_by_y) - 1):
        for j in range(i + 1, min(len(pts_by_y), i + 8)):
            dist = pts_by_y[i].dist(pts_by_y[j])
            if dist < sigma:
                sigma = dist
                pair = (pts_by_y[i], pts_by_y[j])

    return pair, sigma



if __name__ == "__main__":
    import time
    length = 500
    pts = [Point() for _ in range(length)]

    tic = time.time()
    brute_pair = brute_force_closest(pts)
    x, y = brute_pair
    print(f"({x}, {y}), dist: {x.dist(y)}")

    pair = get_closest_pair(pts)
    x, y = pair
    print(f"({x}, {y}), dist: {x.dist(y)}")
