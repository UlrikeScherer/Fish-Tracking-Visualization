from .tank_area_config import BLOCK, BLOCK1, BLOCK2, nov02, nov21, get_areas, compute_calibrations

if __name__ == "__main__":
    if BLOCK == BLOCK1:
        get_areas()
        compute_calibrations()
    elif BLOCK == BLOCK2:
        get_areas(nov02)
        get_areas(nov21)
        compute_calibrations(nov02)
        compute_calibrations(nov21)