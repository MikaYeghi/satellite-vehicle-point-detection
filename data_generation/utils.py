import math
from random import uniform

def sample_random_elev_azimuth(x_min, y_min, x_max, y_max, distance):
    """
    This function samples x and y coordinates on a plane, and converts them to elevation and azimuth angles.
    
    It was found that x_min = y_min = -1.287 and x_max = y_max = 1.287 result in the best angles, where elevation ranges roughly from 70 to 90, and azimuth goes from 0 to 360.
    """
    x = uniform(x_min, x_max)
    y = uniform(y_min, y_max)
    
    if x == 0 and y == 0:
        elevation = 90.0
        azimuth = 0.0
    elif x == 0:
        elevation = math.atan(distance / math.sqrt(x * x + y * y)) * 180.0 / math.pi
        azimuth = 0.0
    else:
        elevation = math.atan(distance / math.sqrt(x * x + y * y)) * 180.0 / math.pi
        azimuth = math.atan(y / x) * 180.0 / math.pi
        if x < 0:
            if y > 0:
                azimuth += 180
            else:
                azimuth -= 180
    
    return (elevation, azimuth)