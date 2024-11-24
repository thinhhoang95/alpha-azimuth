from traffic.data import eurofirs

def get_fir_polygon(fir_name):
    """
    Get FIR polygon from traffic.data eurofirs.
    
    Args:
        fir_name (str): FIR identifier (e.g., 'EGTT')
    
    Returns:
        shapely.geometry.Polygon: Polygon representing the FIR boundary
    """
    # Get FIR data from eurofirs
    fir = eurofirs.query(f"designator == '{fir_name}'")
    return fir.data.iloc[0].geometry

def filter_segments_by_fir(segments_df, fir_name):
    """
    Filter segments that have at least one endpoint within the specified FIR.
    
    Args:
        segments_df (pd.DataFrame): DataFrame containing segments with columns 
            [from_lat, from_lon, to_lat, to_lon]
        fir_name (str): FIR identifier (e.g., 'EGTT')
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing only segments with points inside the FIR
    """
    from shapely.geometry import Point
    from shapely.prepared import prep
    
    # Get FIR polygon from your FIR data source
    fir_polygon = get_fir_polygon(fir_name)  # You'll need to implement this
    prepared_polygon = prep(fir_polygon)  # Optimize for repeated contains checks
    
    def point_in_fir(lat, lon):
        point = Point(lon, lat)  # Note: Shapely uses (x,y) order (lon,lat)
        return prepared_polygon.contains(point)
    
    # Create mask for segments where either endpoint is inside the FIR
    mask = (
        segments_df.apply(lambda row: (
            point_in_fir(row['from_lat'], row['from_lon']) or 
            point_in_fir(row['to_lat'], row['to_lon'])
        ), axis=1)
    )
    
    # Return filtered DataFrame
    return segments_df[mask]