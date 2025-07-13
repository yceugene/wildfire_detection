import os
import psutil
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import box
from pyproj import CRS
import numpy as np
import geodatasets 
from pykrige.ok import OrdinaryKriging

def plot_kriging_with_perimeter_and_basegrid(
    z_pred, height, width, basegrid_tif_path, perimeter_shp_path, title="Kriging Interpolation Result"
):
    """
    Visualize kriging result overlaid with fire perimeter (shapefile) and basegrid extent (tif).
    Main functions:
    - Show kriging prediction area as image
    - Overlay fire perimeter (blue) and basegrid extent (red)

    Parameters:
        z_pred: 1D numpy array, kriging interpolation result (length = height * width)
        height, width: raster grid shape (from tif)
        basegrid_tif_path: str, path to basegrid (e.g. 'basegrid_180m.tif')
        perimeter_shp_path: str, path to fire perimeter shapefile
        title: str, plot title
    """
    # Load basegrid bounds and CRS
    with rasterio.open(basegrid_tif_path) as src:
        bounds = src.bounds
        raster_crs = src.crs

    # Load fire perimeter
    perimeter = gpd.read_file(perimeter_shp_path).to_crs(raster_crs)

    # Build extent rectangle for basegrid
    from shapely.geometry import box
    basegrid_box = gpd.GeoDataFrame(
        {'geometry':[box(bounds.left, bounds.bottom, bounds.right, bounds.top)]},
        crs=raster_crs
    )

    # Plot everything
    fig, ax = plt.subplots(figsize=(10, 10))
    # 1. Show kriging result as image (background)
    im = ax.imshow(
        z_pred.reshape(height, width),
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin='lower', cmap='hot', alpha=0.85
    )
    plt.colorbar(im, ax=ax, label='Fire Arrival (JDAYDEC)')
    # 2. Draw basegrid extent (red line)
    basegrid_box.boundary.plot(ax=ax, color='red', linewidth=2, label="Basegrid extent")
    # 3. Draw fire perimeter (blue)
    perimeter.boundary.plot(ax=ax, color='blue', linewidth=1, label="Fire perimeter")
    # Finish
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.tight_layout()
    plt.show()

# --- Example usage ---
# plot_kriging_with_perimeter_and_basegrid(z_pred, height, width, "basegrid_180m.tif", "perimeter.shp")

def blockwise_kriging(hotspot_x, hotspot_y, hotspot_z, grid_points, block_size=10000, variogram_model='exponential'):
    """
    Perform kriging in blocks to avoid memory errors.
    Returns concatenated prediction values.
    """
    n_points = grid_points.shape[0]
    n_blocks = int(np.ceil(n_points / block_size))
    z_pred = np.empty(n_points, dtype=np.float32)
    
    print(f"Total grid points: {n_points}, processing {n_blocks} blocks of {block_size} points each.")
    
    # Create an Ordinary Kriging Opject
    OK = OrdinaryKriging(hotspot_x, hotspot_y, hotspot_z,
                         variogram_model=variogram_model,
                         verbose=False, enable_plotting=False)
    
    for i in range(n_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, n_points)
        grid_block = grid_points[start:end]
        z_block, _ = OK.execute('points', grid_block[:, 0], grid_block[:, 1])
        z_pred[start:end] = z_block
        print(f"Block {i+1}/{n_blocks} done. Points {start} ~ {end}")
    return z_pred

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / 1024**2:.2f} MB")



def plot_perimeter_and_basegrid(perimeter_path, basegrid_tif_path):
    """
    Visualize the overlap of a fire perimeter (shapefile) and a raster grid (GeoTIFF), 
    and display their respective areas in km².
    
    Parameters:
        perimeter_path (str): Path to the fire perimeter shapefile.
        basegrid_tif_path (str): Path to the basegrid GeoTIFF.
    """
    # Load perimeter shapefile
    perimeter = gpd.read_file(perimeter_path)
    
    # Load raster and get its bounding box
    with rasterio.open(basegrid_tif_path) as src:
        bounds = src.bounds
        raster_crs = src.crs
    
    # Build raster bounding box polygon
    basegrid_box = gpd.GeoDataFrame({'geometry': [box(bounds.left, bounds.bottom, bounds.right, bounds.top)]}, crs=raster_crs)
    
    # Project both to an equal area CRS for accurate area calculation
    equal_area_crs = CRS.from_epsg(6933)  # World Cylindrical Equal Area
    perimeter_proj = perimeter.to_crs(equal_area_crs)
    basegrid_box_proj = basegrid_box.to_crs(equal_area_crs)
    
    # Calculate area in km²
    perimeter_area_km2 = perimeter_proj.geometry.area.sum() / 1e6
    basegrid_area_km2 = basegrid_box_proj.geometry.area.sum() / 1e6
    
    # Project both to WGS84 for visualization
    perimeter_wgs84 = perimeter.to_crs("EPSG:4326")
    basegrid_box_wgs84 = basegrid_box.to_crs("EPSG:4326")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    basegrid_box_wgs84.boundary.plot(ax=ax, color='red', linewidth=2, label=f'Basegrid extent\n({basegrid_area_km2:.1f} km²)')
    perimeter_wgs84.boundary.plot(ax=ax, color='blue', linewidth=2, label=f'Fire perimeter\n({perimeter_area_km2:.1f} km²)')
    
    ax.set_title("Overlay of basegrid and perimeter boundary", fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.show()
    
    # Print area info
    print(f"Basegrid area: {basegrid_area_km2:.2f} km²")
    print(f"Perimeter area: {perimeter_area_km2:.2f} km²")

# Example usage:
# plot_perimeter_and_basegrid("perimeter.shp", "basegrid_180m.tif")

def show_shapefile_info(shp_path, head_n=5):
    """
    Print basic info and show a preview for a shapefile (.shp).
    
    Parameters
    ----------
    shp_path : str
        Path to the shapefile (.shp)
    head_n : int
        Number of rows to display (default: 5)
    """
    gdf = gpd.read_file(shp_path)
    print("=== Shapefile Info ===")
    print(f"File: {shp_path}")
    print(f"Number of records: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"Geometry type(s): {gdf.geom_type.unique()}")
    print(f"Columns: {list(gdf.columns)}")
    print("\nSample rows:")
    print(gdf.head(head_n))
    
    # Quick plot of the geometry
    gdf.plot(figsize=(6,6), edgecolor='black')
    plt.title("Shapefile geometry preview")
    plt.xlabel("Longitude (or X)")
    plt.ylabel("Latitude (or Y)")
    plt.show()

def show_tif_info(tif_path):
    """
    Display basic information, real-world size (km), and a preview plot for a GeoTIFF file.

    Parameters:
    -----------
    tif_path : str
        File path to the .tif or .tiff file.
    """
    with rasterio.open(tif_path) as src:
        print("=== GeoTIFF Info ===")
        print(f"File: {tif_path}")
        print(f"Driver: {src.driver}")
        print(f"Size: {src.width} x {src.height}")
        print(f"Number of bands: {src.count}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Pixel size: {src.res}")
        print(f"NoData value: {src.nodata}")
        print(f"Transform: {src.transform}")

        # Calculate real-world coverage in km
        width_m = abs(src.bounds.right - src.bounds.left)
        height_m = abs(src.bounds.top - src.bounds.bottom)
        print(f"\n=== Real-world coverage ===")
        print(f"Width:  {width_m/1000:.2f} km")
        print(f"Height: {height_m/1000:.2f} km")

        # Read data and show statistics for each band
        for i in range(1, src.count + 1):
            data = src.read(i)
            print(f"\n--- Band {i} statistics ---")
            print(f"Min: {np.nanmin(data)}")
            print(f"Max: {np.nanmax(data)}")
            print(f"Mean: {np.nanmean(data)}")
            unique_vals = np.unique(data[~np.isnan(data)])
            print(f"Unique values (sample): {unique_vals[:10]}")
            if len(unique_vals) > 10:
                print(f"... ({len(unique_vals)} total unique values)")

        # Display preview image
        plt.figure(figsize=(8,6))
        if src.count == 1:
            plt.imshow(src.read(1), cmap='viridis')
            plt.title("Single-band preview")
            plt.colorbar()
        elif src.count >= 3:
            # Show as RGB if possible
            rgb = np.stack([src.read(i) for i in [1,2,3]], axis=-1)
            # Normalize for display
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            plt.imshow(rgb)
            plt.title("RGB preview (Bands 1,2,3)")
        else:
            plt.imshow(src.read(1), cmap='gray')
            plt.title("First band preview")
            plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        


def plot_basegrid_cover_on_canada(tif_path):
    """
    Visualize which part of Canada is covered by a given raster grid (GeoTIFF).
    The function will draw: Canada outline, basegrid extent (red), 
    and highlight the portion of Canada land that falls inside the basegrid (blue fill).

    Parameters:
        tif_path (str): Path to the GeoTIFF raster file.
    """
    # 1. Read raster bounds and CRS
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        raster_crs = src.crs

    # 2. Load Canada boundary as GeoDataFrame
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    # Crop to North America region (for better speed & focus)
    canada = world.cx[-141:-52, 41:84]
    canada = canada.to_crs(raster_crs)

    # 3. Create raster extent as polygon GeoDataFrame
    raster_box = gpd.GeoDataFrame({'geometry': [box(bounds.left, bounds.bottom, bounds.right, bounds.top)]}, crs=raster_crs)
    raster_box_wgs = raster_box.to_crs("EPSG:4326")
    canada_wgs = canada.to_crs("EPSG:4326")

    # 4. Intersection: which part of Canada is inside the raster?
    covered_canada = gpd.overlay(canada, raster_box, how='intersection')
    covered_canada_wgs = covered_canada.to_crs("EPSG:4326")

    # 5. Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    # Draw all Canada border
    canada_wgs.boundary.plot(ax=ax, color='black', linewidth=1, label="Canada border")

    # Draw the covered part of Canada inside the raster (fill)
    if not covered_canada_wgs.empty:
        covered_canada_wgs.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.7, label="Basegrid-covered Canada")
    else:
        print("Warning: No intersection found between raster and Canada region.")

    # Draw the raster extent as a red box
    raster_box_wgs.boundary.plot(ax=ax, color='red', linewidth=2, label=f"Raster extent ({tif_path})")

    ax.set_title(f"{tif_path} coverage on Canada: red box = raster, blue = Canadian land inside raster")
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_basegrid_cover_on_canada("basegrid_180m.tif")

def plot_perimeter_and_basegrid_v2(perimeter_path, basegrid_tif_path, grid_points=None, z_pred=None):
    """
    Visualize the overlap of a fire perimeter (shapefile), a raster grid (GeoTIFF),
    and optionally display kriging predictions as a scatter plot.
    """

    # Load perimeter shapefile
    perimeter = gpd.read_file(perimeter_path)

    # Load raster and get its bounding box
    with rasterio.open(basegrid_tif_path) as src:
        bounds = src.bounds
        raster_crs = src.crs

    # Build raster bounding box polygon
    basegrid_box = gpd.GeoDataFrame({'geometry': [box(bounds.left, bounds.bottom, bounds.right, bounds.top)]}, crs=raster_crs)

    # Project both to an equal area CRS for accurate area calculation
    equal_area_crs = CRS.from_epsg(6933)  # World Cylindrical Equal Area
    perimeter_proj = perimeter.to_crs(equal_area_crs)
    basegrid_box_proj = basegrid_box.to_crs(equal_area_crs)

    # Calculate area in km²
    perimeter_area_km2 = perimeter_proj.geometry.area.sum() / 1e6
    basegrid_area_km2 = basegrid_box_proj.geometry.area.sum() / 1e6

    # Project both to WGS84 for visualization
    perimeter_wgs84 = perimeter.to_crs("EPSG:4326")
    basegrid_box_wgs84 = basegrid_box.to_crs("EPSG:4326")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    basegrid_box_wgs84.boundary.plot(ax=ax, color='red', linewidth=2, label=f'Basegrid extent\n({basegrid_area_km2:.1f} km²)')
    perimeter_wgs84.boundary.plot(ax=ax, color='blue', linewidth=2, label=f'Fire perimeter\n({perimeter_area_km2:.1f} km²)')

    # kriging results
    if grid_points is not None and z_pred is not None:
        # Check grid_points format (N,2)
        grid_x = grid_points[:, 0]
        grid_y = grid_points[:, 1]
        sc = ax.scatter(grid_x, grid_y, c=z_pred, cmap='viridis', s=4, alpha=0.7, label='Kriging prediction')
        cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
        cbar.set_label('Predicted (Kriging)')

    ax.set_title("Overlay of basegrid, perimeter boundary, and kriging prediction", fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.show()

    print(f"Basegrid area: {basegrid_area_km2:.2f} km²")
    print(f"Perimeter area: {perimeter_area_km2:.2f} km²")

# Example usage:
# plot_perimeter_and_basegrid("perimeter.shp", "basegrid_180m.tif", grid_points=grid_points, z_pred=z_pred)