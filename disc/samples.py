import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import copy

def get_1a_sample(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # Class is the shown shape,
    # 0: triangle, 1: square, 2: disk
    # [triangles, squares, disks]
    rejected = True
    while rejected:
        classes = [[], [], []]
        rand_specs = get_random_spec(h,w, intensity=intensity, noise_strength=noise_strength)
        classes[c].append(rand_specs)
        sample, areas = get_sample(h,w,*classes, canvas_noise, canvas_sigma)
        
        if np.sum(sample>0) > 9./10. * areas[0]:
            rejected = False
        
    return sample

def get_1b_sample(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # Class is the majority count of shape
    # Classes as in 1a
    class_ids = [0,1,2]
    class_ids_all = [0,1,2]
    
    class_ids.remove(c)
    c0 = class_ids[0]
    class_ids.remove(c0)
    c1 = class_ids[0]
    
    n_c = np.random.randint(1,4)
    n_0 = np.random.randint(0,n_c)
    n_1 = np.random.randint(0,n_c)
    
    n = {c: n_c,
         c0: n_0,
         c1: n_1}
    
    rejected = True
    while rejected:
        classes = [[], [], []]
        for class_id in class_ids_all:
            n_in_class = n[class_id]
            for i in range(n_in_class):
                classes[class_id].append(get_random_spec(h,w, intensity=intensity, noise_strength=noise_strength))
                
        sample, areas = get_sample(h,w,*classes, canvas_noise, canvas_sigma)
        
        if np.sum(sample>0) > 9./10. * np.sum(areas):
            rejected = False
        
    return sample
        
def get_1c_sample(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # Each class is a combination of two shapes (Zebra Socks)
    # 0: triangle, square
    # 1: triangle, disk
    # 2: square disk
    
    if c == 0:
        class_ids = [0, 1]
    elif c == 1:
        class_ids = [0, 2]
    elif c == 2:
        class_ids = [1, 2]
    else:
        raise ValueError("Class not supported")
    
    
    rejected = True
    while rejected:
        classes = [[], [], []]
        
        for c in class_ids:
            rand_specs = get_random_spec(h,w, intensity=intensity, noise_strength=noise_strength)
            classes[c].append(rand_specs)
            
        sample, areas = get_sample(h,w,*classes, canvas_noise)
        
        if np.sum(sample>0) > 9./10. * np.sum(areas):
            rejected = False
        
    return sample

def get_1d_sample(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # Each class is a combination of two shapes (Zebra Socks)
    # 0: even number of triangles
    # 1: odd number of triangles

    rand_idx = np.random.randint(0,3)
    if c == 0:
        n = [2,4,6][rand_idx]
    elif c == 1:
        n = [1,3,5][rand_idx]
    else:
        raise ValueError("Class not supported")
    
    rejected = True
    while rejected:
        classes = [[], [], []]
        
        for i in range(n):
            rand_specs = get_random_spec(h,w, intensity=intensity, noise_strength=noise_strength)
            classes[0].append(rand_specs)
            
        sample, areas = get_sample(h,w,*classes, canvas_noise)
        
        if np.sum(sample>0) > 9/10. * np.sum(areas):
            rejected = False
        
    return sample

def get_2a_sample(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # 0: Triangle XOR Disk (Triangle, Disk)
    # 1: !(Triangle XOR Disk) (Triangle & Disk)
    
    class_ids = []
    if c == 0:
        class_ids.append([0,2][np.random.randint(2)])
    if c == 1:
        class_ids.append(0)
        class_ids.append(2)
        
    rejected = True
    while rejected:
        classes = [[], [], []]
        
        for c in class_ids:
            rand_specs = get_random_spec(h,w, intensity=intensity, noise_strength=noise_strength)
            classes[c].append(rand_specs)
            
        sample, areas = get_sample(h,w,*classes, canvas_noise)
        
        if np.sum(sample>0) > 9./10. * np.sum(areas):
            rejected = False
        
    return sample

def get_2b_sample(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # 0: Even number of excess triangles
    # 1: Odd number of excess triangles
    # excess = triangles - disks
    
    class_ids = [0,2]
    if c == 0:
        n_disks = np.random.randint(4)
        excess_triangles = [2,4][np.random.randint(2)]
        n_triangles = n_disks + excess_triangles
        
    if c == 1:
        n_disks = np.random.randint(4)
        excess_triangles = [1,3][np.random.randint(2)]
        n_triangles = n_disks + excess_triangles
    
    n_samples = [n_triangles, n_disks]

        
    rejected = True
    while rejected:
        classes = [[], [], []]
        
        j = 0
        for c in class_ids:
            for n in range(n_samples[j]):
                rand_specs = get_random_spec(h,w, intensity=intensity, noise_strength=noise_strength)
                classes[c].append(rand_specs)
            j += 1
            
        sample, areas = get_sample(h,w,*classes, canvas_noise)
        
        if np.sum(sample>0) > 9./10. * np.sum(areas):
            rejected = False
        
    return sample

def get_2c_sample(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # 0: Even number of excess triangles
    # 1: Odd number of excess triangles
    # excess = triangles - (disks + squares)
    
    class_ids = [0,1,2]
    if c == 0:
        n_disks = np.random.randint(2)
        n_squares = np.random.randint(2)
        excess_triangles = [2,4][np.random.randint(2)]
        n_triangles = n_disks + n_squares + excess_triangles
        
    if c == 1:
        n_disks = np.random.randint(2)
        n_squares = np.random.randint(2)
        excess_triangles = [1,3][np.random.randint(2)]
        n_triangles = n_disks + n_squares + excess_triangles
    
    n_samples = [n_triangles, n_squares, n_disks]

        
    rejected = True
    while rejected:
        classes = [[], [], []]
        
        j = 0
        for c in class_ids:
            for n in range(n_samples[j]):
                rand_specs = get_random_spec(h,w, intensity=intensity, noise_strength=noise_strength)
                classes[c].append(rand_specs)
            j += 1
            
        sample, areas = get_sample(h,w,*classes, canvas_noise)
        
        if np.sum(sample > 0) > 9./10. * np.sum(areas):
            rejected = False
        
    return sample


def get_2d_sample(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # 0: Even number of excess triangles
    # 1: Odd number of excess triangles
    # excess = triangles - disks
    # plus random number of squares as distractors
    
    class_ids = [0,1,2]
    if c == 0:
        n_disks = np.random.randint(3)
        excess_triangles = [2,4][np.random.randint(2)]
        n_triangles = n_disks + excess_triangles
        
    if c == 1:
        n_disks = np.random.randint(3)
        excess_triangles = [1,3][np.random.randint(2)]
        n_triangles = n_disks + excess_triangles
    
    n_squares = [0,1,2][np.random.randint(3)]
    n_samples = [n_triangles, n_squares, n_disks]
        
    rejected = True
    while rejected:
        classes = [[], [], []]
        
        j = 0
        for c in class_ids:
            for n in range(n_samples[j]):
                rand_specs = get_random_spec(h,w, intensity=intensity, noise_strength=noise_strength)
                classes[c].append(rand_specs)
            j += 1
            
        sample, areas = get_sample(h,w,*classes, canvas_noise)
        
        if np.sum(sample>0) > 9./10. * np.sum(areas):
            rejected = False
        
    return sample

def get_2b_sample_simple(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # 0: Even number of excess triangles
    # 1: Odd number of excess triangles
    # excess = triangles - disks
    
    class_ids = [0,2]
    if c == 0:
        n_disks = np.random.randint(4)
        excess_triangles = [2,4][np.random.randint(2)]
        n_triangles = n_disks + excess_triangles
        
    if c == 1:
        n_disks = np.random.randint(4)
        excess_triangles = [1,3][np.random.randint(2)]
        n_triangles = n_disks + excess_triangles
    
    n_samples = [n_triangles, n_disks]

        
    rejected = True
    while rejected:
        classes = [[], [], []]
        
        j = 0
        for c in class_ids:
            for n in range(n_samples[j]):
                rand_specs = get_random_spec(h,w, intensity=intensity, 
                                             noise_strength=noise_strength, rotation=0, size=20)
                classes[c].append(rand_specs)
            j += 1
            
        sample, areas = get_sample(h,w,*classes, canvas_noise)
        
        if np.sum(sample>0)/np.sum(areas)>0.999:
        #if np.sum(sample>0) >= np.sum(areas):
            rejected = False
        
    return sample

def get_2c_sample_simple(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # 0: Even number of excess triangles
    # 1: Odd number of excess triangles
    # excess = triangles - (disks + squares)
    
    class_ids = [0,1,2]
    if c == 0:
        n_disks = np.random.randint(2)
        n_squares = np.random.randint(2)
        excess_triangles = [2,4][np.random.randint(2)]
        n_triangles = n_disks + n_squares + excess_triangles
        
    if c == 1:
        n_disks = np.random.randint(2)
        n_squares = np.random.randint(2)
        excess_triangles = [1,3][np.random.randint(2)]
        n_triangles = n_disks + n_squares + excess_triangles
    
    n_samples = [n_triangles, n_squares, n_disks]

        
    rejected = True
    while rejected:
        classes = [[], [], []]
        
        j = 0
        for c in class_ids:
            for n in range(n_samples[j]):
                rand_specs = get_random_spec(h,w, intensity=intensity, noise_strength=noise_strength, 
                                             rotation=0, size=20)
                
                classes[c].append(rand_specs)
            j += 1
            
        sample, areas = get_sample(h,w,*classes, canvas_noise)
        
        if np.sum(sample>0)/np.sum(areas)>0.999:
            rejected = False
        
    return sample

def get_2d_sample_simple(c, h=128, w=128, canvas_noise=0, canvas_sigma=0, intensity=None, noise_strength=None):
    # 0: Even number of excess triangles
    # 1: Odd number of excess triangles
    # excess = triangles - disks
    # plus random number of squares as distractors
    
    class_ids = [0,1,2]
    if c == 0:
        n_disks = np.random.randint(3)
        excess_triangles = [2,4][np.random.randint(2)]
        n_triangles = n_disks + excess_triangles
        
    if c == 1:
        n_disks = np.random.randint(3)
        excess_triangles = [1,3][np.random.randint(2)]
        n_triangles = n_disks + excess_triangles
    
    n_squares = [0,1,2][np.random.randint(3)]
    n_samples = [n_triangles, n_squares, n_disks]

        
    rejected = True
    while rejected:
        classes = [[], [], []]
        
        j = 0
        for c in class_ids:
            for n in range(n_samples[j]):
                rand_specs = get_random_spec(h,w, intensity=intensity, noise_strength=noise_strength, 
                                             rotation=0, size=20)
                classes[c].append(rand_specs)
            j += 1
            
        sample, areas = get_sample(h,w,*classes, canvas_noise)
        
        if np.sum(sample>0)/np.sum(areas)>0.999:
            rejected = False
        
    return sample


def get_sample(h, w, triangles, squares, disks, canvas_noise=0, canvas_sigma=0):
    canvas = get_canvas(h,w, canvas_noise, canvas_sigma)
    areas = []
    for d in disks:
        canvas,area = add_disk(canvas, *d)
        areas.append(area)
    for s in squares:
        canvas,area = add_square(canvas, *s)
        areas.append(area)
    for t in triangles:
        canvas,area = add_triangle(canvas, *t)
        areas.append(area)
    return canvas, areas

def get_random_spec(h, w, intensity=None, noise_strength=None, rotation=None, size=None):
    min_side = np.minimum(h,w)
    if size is None:
        size_low = int(min_side * 0.2)
        size_high = int(min_side * 0.4)
        size = np.random.randint(size_low, size_high)
    
    x = np.random.randint(0, w)
    y = np.random.randint(0, h)
    if intensity is None:
        intensity = np.random.randint(120,200)
    if noise_strength is None:
        noise_strength = np.random.randint(20,120)
    if rotation is None:
        rotation = np.random.randint(0,359)
    sigma=2
    
    return (x,y,size,intensity,noise_strength,sigma,rotation)

def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def get_canvas(h,w, noise_strength=0, sigma=0):
    canvas = np.random.randn(h,w) * noise_strength
    canvas = scipy.ndimage.filters.gaussian_filter(canvas, sigma, mode='constant')
    canvas[canvas > 255] = 255
    canvas[canvas < 0] = 0
    return canvas.astype(np.int16)

def add_disk(canvas, x, y, diameter, intensity=100, noise_strength=20, sigma=2, rot=0):
    radius = diameter/2
    h,w = np.shape(canvas)
    xx, yy = np.mgrid[:h, :w]
    circle = (xx - (h - y)) ** 2 + (yy - x) ** 2 <= radius**2
    
    noise_canvas = np.random.randn(*np.shape(canvas)) * noise_strength
    canvas[circle==1] = canvas[circle==1] + noise_canvas[circle==1] + intensity
    canvas[circle==1] = scipy.ndimage.filters.gaussian_filter(canvas, sigma, mode='constant')[circle==1]
    canvas[canvas > 255] = 255
    canvas[canvas < 0] = 0
    return canvas, np.pi * (diameter/2)**2 - np.sum(canvas[circle==1] == 0)

def point_in_triangle(canvas, px, py, ax, ay, bx, by, cx, cy):
    # Check if a point p is inside a triangle defined by 
    # the three corners a,b,c 
    pax = px - ax
    pay = py - ay
    
    pab = (bx - ax) * pay - (by - ay) * pax > 0    
    
    canvas = canvas == 0
    canvas[((cx - ax) * pay - (cy - ay) * pax > 0) == pab] = 0
    canvas[((cx - bx) * (py-by) - (cy - by) * (px - bx) > 0) != pab] = 0
    return canvas

def add_triangle(canvas, x, y, length, intensity=100, noise_strength=20, sigma=2, rot=80):
    # Add rotation
    h = np.sqrt(3)/2 * length 
    ax = np.shape(canvas)[0] - y
    ay = x
    
    bx = ax - length/2
    by = ay - h
    
    cx = ax + length/2
    cy = ay - h
    
    center = (int(ax), int(ay - h/2))
    points = [(ax,ay), (bx, by), (cx, cy)]
    points = rotate(points, origin=center, degrees=rot)
    a = points[0]
    b = points[1]
    c = points[2]
    ax = a[0]
    ay = a[1]
    bx = b[0]
    by = b[1]
    cx = c[0]
    cy = c[1]
    
    h,w = np.shape(canvas)
    xx, yy = np.mgrid[:h, :w]
    triangle = point_in_triangle(canvas, xx, yy, ax, ay, bx, by, cx, cy)
    
    noise_canvas = np.random.randn(*np.shape(canvas)) * noise_strength
    canvas[triangle==1] = canvas[triangle==1] + noise_canvas[triangle==1] + intensity
    canvas[triangle==1] = scipy.ndimage.filters.gaussian_filter(canvas, sigma, mode='constant')[triangle==1]
    canvas[canvas > 255] = 255
    canvas[canvas < 0] = 0
    return canvas, np.sqrt(3)/4. * length**2 - np.sum(canvas[triangle==1] == 0)

def point_in_square(canvas, px, py, ax, ay, bx, by, cx, cy):
    p = (px, py)
    
    ab = np.array([bx - ax, by - ay])
    ap = np.array([px - ax, py - ay])
    bc = np.array([cx - bx, cy - by])
    bp = np.array([px - bx, py - by])
    
    check_0 = 0 <= np.sum(ab*ap.T,axis=2) 
    check_1 = np.sum(ab*ap.T,axis=2) <= np.dot(ab,ab)
    
    check_2 = 0 <= np.sum(bc*bp.T,axis=2) 
    check_3 = np.sum(bc*bp.T,axis=2) <= np.dot(bc,bc)
    
    check_all = check_0 * check_1 * check_2 * check_3
    canvas[check_all] = True
    return canvas

def add_square(canvas, x, y, length, intensity=100, noise_strength=20, sigma=2, rot=45):
    h,w = np.shape(canvas)
    
    ax = x
    ay = y
    
    bx = x + length
    by = y
    
    cx = x + length
    cy = y + length
    
    points = [(ax, ay), (bx, by), (cx, cy)]
    center = (int(x+length/2), int(y + length/2))
    points = rotate(points, origin=center, degrees=rot)
    
    ax = points[0][0]
    ay = points[0][1]    
    bx = points[1][0]
    by = points[1][1]
    cx = points[2][0]
    cy = points[2][1]
    
    xx, yy = np.mgrid[:h, :w]
    square = point_in_square(canvas, xx, yy, ax, ay, bx, by, cx, cy)
    square_copy = copy.deepcopy(square)
    noise_canvas = np.random.randn(*np.shape(canvas)) * noise_strength
    canvas[square==1] = canvas[square==1] + noise_canvas[square==1] + intensity
    canvas[square_copy==1] = scipy.ndimage.filters.gaussian_filter(canvas, sigma, mode='constant')[square_copy==1]
    canvas[canvas > 255] = 255
    canvas[canvas < 0] = 0
    return canvas, length**2 - np.sum(canvas[square==1] == 0)



