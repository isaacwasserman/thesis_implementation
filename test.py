def get_window_indices(image_size, window_size):
    indices = []
    for x in range(image_size - window_size + 1):
        for y in range(image_size - window_size + 1):
            indices.append((x, y))
    return indices

def get_window_index(x, y, window_size):
    return (x // window_size, y // window_size)

def get_region_indices(image_size, window_size):
    indices = []
    for x in range(window_size):
        for y in range(window_size):
            indices.append((x, y))
    return indices

def get_regions_containing_pixel(pixel_index, image_size, window_size):
    window_indices = get_window_indices(image_size, window_size)
    region_indices = get_region_indices(window_size, window_size)
    regions = []
    for window_index in window_indices:
        window_x, window_y = window_index
        for region_index in region_indices:
            region_x, region_y = region_index
            x = window_x * window_size + region_x
            y = window_y * window_size + region_y
            if (x, y) == pixel_index:
                regions.append((window_x, window_y))
    return regions

# Example usage:
image_size = 512
window_size = 32
pixel_index = (150, 212)

# Find regions containing pixel at pixel_index
regions = get_regions_containing_pixel(pixel_index, image_size, window_size)

# Print window indices containing pixel
for region in regions:
    print("Window index: ({}, {})".format(region[0], region[1]))