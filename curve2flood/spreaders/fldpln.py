import numpy as np
from numba import njit, prange

@njit(cache=True)
def get_dr_dc_from_flowdir(flowdir_value):
    if flowdir_value == 1:
        return -1, 1
    elif flowdir_value == 2:
        return 0, 1
    elif flowdir_value == 4:
        return 1, 1
    elif flowdir_value == 8:
        return 1, 0
    elif flowdir_value == 16:
        return 1, -1
    elif flowdir_value == 32:
        return 0, -1
    elif flowdir_value == 64:
        return -1, -1
    elif flowdir_value == 128:
        return -1, 0
    return 0, 0

@njit(cache=True)
def get_d8():
    # Whitebox D8 pointer encoding:
    # 1=NE, 2=E, 4=SE, 8=S, 16=SW, 32=W, 64=NW, 128=N
    return (
        (-1, 0, 128),
        ( 1, 0,   8),
        ( 0, 1,   2),
        ( 0,-1,  32),
        (-1,-1, 64),
        ( 1,-1, 16),
        (-1, 1,  1),
        ( 1, 1,  4),
    )

@njit(cache=True)
def get_neighbor_offsets():
    return ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

@njit(cache=True)
def backfill_and_identify_main_channel(streams: np.ndarray, flowdir: np.ndarray, WSE_Initial: np.ndarray, WSE_Out_stream: np.ndarray, E: np.ndarray, stream_id: int):
    main_channel_not_in_stream = set()
    nrows, ncols = streams.shape
    MIN_FLOOD_DEPTH = 0.0

    # Process the seed catalog for this stream.
    for sr in range(nrows):
        for sc in range(ncols):
            if streams[sr, sc] != stream_id:
                continue

            # Go downstream to find the main channel, not represented in the stream raster
            rr = sr
            cc = sc
            while True:
                dr, dc = get_dr_dc_from_flowdir(flowdir[rr, cc])
                if dr == 0 and dc == 0:
                    break
                rr += dr
                cc += dc
                if not (0 <= rr < nrows and 0 <= cc < ncols):
                    break
                if streams[rr, cc] == stream_id:
                    break
                if (rr, cc) in main_channel_not_in_stream:
                    break
                main_channel_not_in_stream.add((rr, cc))

            wse = WSE_Initial[sr, sc]
            if np.isnan(wse) or wse <= -9998.0 or wse <= E[sr, sc] + MIN_FLOOD_DEPTH:
                continue

            if np.isnan(WSE_Out_stream[sr, sc]) or wse > WSE_Out_stream[sr, sc]:
                WSE_Out_stream[sr, sc] = wse

            # Start backfilling from this water elevation source cell
            queue = [(sr, sc)]

            # Depth-first search (DFS) over reverse-flow neighbors.
            while queue:
                r, c = queue.pop()

                for dr, dc, dir_value in get_d8():
                    nr = r - dr
                    nc = c - dc

                    # Is the new cell a cell which flows into this one?
                    if 0 <= nr < nrows and 0 <= nc < ncols and flowdir[nr, nc] == dir_value:
                        # Is the new cell inundated by this stream cell?
                        if E[nr, nc] > -9998.0 and E[nr, nc] + MIN_FLOOD_DEPTH <= wse:
                            if np.isnan(WSE_Out_stream[nr, nc]) or wse >= WSE_Out_stream[nr, nc]:
                                queue.append((nr, nc))
                                WSE_Out_stream[nr, nc] = wse

    return main_channel_not_in_stream

@njit(cache=True)
def sort_interior_boundary(lst: list):
    # This is a numba-compatible sort, since numba cannot cache list.sort(key=lambda x: x[2]) or sorted(lst, key=lambda x: x[2])
    def key_func(x):
        return x[2]  # Sort by elevation (the third element of the tuple)

    for i in range(1, len(lst)):
        current_item = lst[i]
        current_key = key_func(current_item)
        j = i - 1
        
        while j >= 0 and key_func(lst[j]) > current_key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = current_item
        
    return lst

@njit(cache=True)
def sort_exterior_boundary(lst: list):
    # This is a numba-compatible sort, since numba cannot cache list.sort(key=lambda x: x[2], x[3]) or sorted(lst, key=lambda x: x[2], x[3])
    def key_func(x):
        return x[2], x[3]  # Sort by spillover, than elevation

    for i in range(1, len(lst)):
        current_item = lst[i]
        current_key = key_func(current_item)
        j = i - 1
        
        while j >= 0 and key_func(lst[j]) < current_key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = current_item
        
    return lst

@njit(cache=True, nogil=True, parallel=True)
# @profile
def fldpln(WSE_Initial: np.ndarray, E: np.ndarray, flowdir: np.ndarray, streams: np.ndarray, unique_stream_ids: np.ndarray):
    """
    This function is meant to mimic the FLDPLN model developed at the University of Kansas, translated iteratively
    using Codex-ChatGPT and this repository: https://github.com/AlabamaWaterInstitute/fldpln and this documentation: 
    https://services.kars.geoplatform.ku.edu/fldpln/AGU_2023_Operational_FIM_in_Kansas.pdf and 
    https://kuscholarworks.ku.edu/server/api/core/bitstreams/df102f13-5968-4e45-ad04-91b4d8086de0/content

    Propagate each seeded WSE upstream along reverse flow direction, then iteratively
    perform boundary spillover and upstream backfill to steady-state.
    A cell is inundated if its elevation + 0.1 m is lower than the seed WSE and it drains to
    that seed cell. If multiple seeds reach a cell, keep the maximum WSE.

    Inputs:
    - WSE_Initial: seeded WSE raster (nan/-9998 where dry); WSE for each stream cell
    - E: ground elevation raster
    - flowdir: D8 flow direction grid
    - streams: stream/segment ids for source labeling
    - nrows/ncols: raster dimensions
    
    Notes:
    - Spillover candidates are dry boundary cells adjacent to wet cells where WSE_Out > E.
    - Candidate depth is selected from wet neighbors using a minimum required depth
      (tie-breaker: highest boundary elevation).
    - Spillover floods the candidate point and then backfills upstream (reverse flowdir)
      to the spill depth until steady-state.

    Returns WSE_Out or the WSE Array for a one set of streamflow inputs
    """
    nrows, ncols = WSE_Initial.shape
    # Initialize outputs: WSE_Out holds max WSE per cell, fsp holds source stream id, dtf holds first inundation stage.
    WSE_Out = np.full((nrows, ncols), np.nan, dtype=np.float32)

    # create an empty array for this segments WSE and that will be blended at the end
    WSE_Out_stream = np.full((nrows, ncols), np.nan, dtype=np.float32)
    MIN_FLOOD_DEPTH = 0.0

    # begin loop over stream ids in order of average WSE (lowest first)
    for stream_id in unique_stream_ids:
        # reset per-stream workspace
        WSE_Out_stream[:] = np.nan

        main_channel_not_in_stream = backfill_and_identify_main_channel(streams, flowdir, WSE_Initial, WSE_Out_stream, E, stream_id)

        ### now perform spillover and backfill for this stream before moving on to the next stream id ###
        # Iterative spillover: boundary spill points -> upsteam and downstream spread using flowdir until steady-state.

        # Build initial boundary queue of wet cells adjacent to dry cells (WSE_Out < E).
        interior_boundary = []
        all_interior_cells = set()
        for r in range(nrows):
            for c in range(ncols):
                if np.isnan(WSE_Out_stream[r, c]):
                    continue

                if E[r, c] < -9998.0 or (WSE_Out_stream[r, c] - E[r, c]) <= MIN_FLOOD_DEPTH:
                    continue

                for dr, dc in get_neighbor_offsets():
                    nr = r + dr
                    nc = c + dc
                    if np.isnan(WSE_Out_stream[nr, nc]) or (WSE_Out_stream[nr, nc] - E[nr, nc]) <= MIN_FLOOD_DEPTH:
                        interior_boundary.append((r, c, E[r, c]))
                        all_interior_cells.add((r, c))
                        break

        if not interior_boundary:
            continue

        # For performance, let's sort the interior boundary by increasing elevation
        interior_boundary = sort_interior_boundary(interior_boundary)
        interior_boundary = [(r, c) for r, c, elev in interior_boundary]  # we only need the coordinates for the spillover loop

        # Plot the interior boundary for debugging
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        # plt.imshow(WSE_Out_stream, cmap='terrain')
        # boundary_rows, boundary_cols = zip(*interior_boundary)
        # plt.scatter(boundary_cols, boundary_rows, color='red', s=1)
        # plt.title('Interior Boundary Cells (Red) on Elevation Map')
        # plt.xlabel('Column Index')
        # plt.ylabel('Row Index')
        # # plt.gca().invert_yaxis()
        # plt.show()

        spill_changed = True
        while spill_changed:
            new_interior_boundary = set()
            spill_changed = False
                        
            if len(interior_boundary) == 0:
                # End the loop because we don't have spillover candidates.
                break

            exterior_boundary = {}

                
            # Identify spillover candidates by minimum depth from wet neighbors.
            while interior_boundary:
                r, c = interior_boundary.pop()
                # Plot a 10x10 grid of the DEM
                # import matplotlib.pyplot as plt
                # plt.imshow(E[max(r-10, 0):min(r+10, nrows), max(c-10, 0):min(c+10, ncols)], cmap='terrain')
                # plt.colorbar(label='Elevation')
                # plt.title(f'10x10 DEM around Interior Boundary Cell ({r}, {c})')
                # plt.xlabel('Column Index')
                # plt.ylabel('Row Index')
                # plt.show()

                # plt.imshow(WSE_Out_stream[max(r-10, 0):min(r+10, nrows), max(c-10, 0):min(c+10, ncols)], cmap='Blues')
                # plt.colorbar(label='WSE_Out_stream')
                # plt.title(f'10x10 WSE_Out_stream around Interior Boundary Cell ({r}, {c})')
                # plt.xlabel('Column Index')
                # plt.ylabel('Row Index')
                # plt.show()
            
                # the wet cell elevation
                wet_cell_elev = E[r, c]
                if wet_cell_elev <= -9998.0:
                    continue
                wet_wse = WSE_Out_stream[r, c]
                wet_depth = wet_wse - wet_cell_elev
                if wet_depth <= MIN_FLOOD_DEPTH:
                    continue

                # Find the candidate dry cell that the wet cell should spill into.
                for dr, dc in get_neighbor_offsets():
                    nr = r + dr
                    nc = c + dc

                    if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                        continue

                    if (nr, nc) in main_channel_not_in_stream:
                        continue

                    # candidate cell's elevation
                    dry_cell_elevation = E[nr, nc]

                    # if we've hit the boundary of the DEM, ignore this cell
                    if dry_cell_elevation <= -9998.0:
                        continue

                    # Preserve the wet-cell depth and reduce it only when the
                    # spill candidate is at a higher elevation.
                    candidate_depth = wet_depth
                    delta_elevation = dry_cell_elevation - wet_cell_elev
                    if delta_elevation > 0.0:
                        candidate_depth = candidate_depth - delta_elevation
                    if candidate_depth <= MIN_FLOOD_DEPTH:
                        continue

                    # if the cell is wet, ignore it
                    if not np.isnan(WSE_Out_stream[nr, nc]) and WSE_Out_stream[nr, nc] >= dry_cell_elevation + candidate_depth:
                        continue

                    # Keep a sparse list of newly touched dry cells and retain
                    # the max passing depth from all wet neighbors.
                    if candidate_depth > exterior_boundary.get((nr, nc), np.float32(-np.inf)):
                        exterior_boundary[(nr, nc)] = candidate_depth

            exterior_boundary_list = [(row, col, depth, E[row, col]) for (row, col), depth in exterior_boundary.items()]

            # Sort spillover locations by decreasing spillover depth (tie-breaker: highest elevation) to prioritize deeper spills and reduce iterations to steady-state.
            exterior_boundary_list = sort_exterior_boundary(exterior_boundary_list)

            # Process spillover candidates and backfill immediately for each new wet cell.
            for r, c, source_depth, exterior_cell_elev in exterior_boundary_list:
                source_wse = exterior_cell_elev + source_depth

                # Write candidate spill only if it actually improves this cell.
                if np.isnan(WSE_Out_stream[r, c]) or (source_wse > WSE_Out_stream[r, c] and source_wse > exterior_cell_elev):
                    WSE_Out_stream[r, c] = source_wse
                else:
                    continue

                # This cell is now wet and can be a spill source in the next iteration, so add to the new interior boundary.
                new_interior_boundary.add((r, c))
                spill_changed = True

                # Seed backfill stack with the current spill source.
                queue = [(r, c)]

                # Flow downstream (using flowdir) until encountering a wet cell, stream location, or dead end.
                rr = r
                cc = c
                # intial depth for spilloever
                depth_use = source_depth
                # initial elevation
                previous_elev = exterior_cell_elev
                while True:
                    fd = flowdir[rr, cc]
                    if fd <= 0:
                        break
                    dr, dc = get_dr_dc_from_flowdir(fd)
                    rr = rr + dr
                    cc = cc + dc
                    if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                        break

                    # if we are at the edge of the DEM stop routing
                    spilled_cell_elev = E[rr, cc]
                    if spilled_cell_elev <= -9998.0:
                        break
                    
                    if (rr, cc) in all_interior_cells and (source_wse - spilled_cell_elev) <= MIN_FLOOD_DEPTH:
                        break

                    # delta_elevation is almost like a rough head loss term
                    delta_elevation = spilled_cell_elev - previous_elev
                    if delta_elevation > 0.0:
                        depth_use = depth_use - delta_elevation
                    if depth_use <= MIN_FLOOD_DEPTH:
                        break

                    new_wse = spilled_cell_elev + depth_use

                    # if the cell is dry or not deep enough, go ahead and flood it
                    if np.isnan(WSE_Out_stream[rr, cc]) or WSE_Out_stream[rr, cc] < new_wse:
                        WSE_Out_stream[rr, cc] = new_wse
                        if (rr, cc) not in new_interior_boundary:
                            new_interior_boundary.add((rr, cc))
                            source_wse = new_wse
                            previous_elev = spilled_cell_elev
                    else:
                        break

                    queue.append((rr, cc))

                    # Backfill upstream (reverse flowdir) immediately from newly flooded spillover cell.
                    # Depth-first search (DFS) over reverse-flow neighbors.
                    while queue:
                        ur, uc = queue.pop()

                        for dr, dc, dir_value in get_d8():
                            nr = ur - dr
                            nc = uc - dc

                            # Is the new cell a cell which flows into this one?
                            if 0 <= nr < nrows and 0 <= nc < ncols and flowdir[nr, nc] == dir_value:
                                # Is the new cell inundated by this stream cell?
                                if new_wse >= E[nr, nc] > -9998.0:
                                    if np.isnan(WSE_Out_stream[nr, nc]) or new_wse > WSE_Out_stream[nr, nc]:
                                        new_interior_boundary.add((nr, nc))
                                        queue.append((nr, nc))
                                        WSE_Out_stream[nr, nc] = new_wse
                                        
                    # if the cell is the main channel, stop routing here
                    if (rr, cc) in main_channel_not_in_stream:
                        break

            # Update boundary queue from newly wet cells using local dry-neighbor
            # count refreshes; only the 3x3 neighborhood around each new wet cell
            # can change frontier status.
            for r, c in new_interior_boundary:
                for dr0, dc0 in get_neighbor_offsets():
                    rr = r + dr0
                    cc = c + dc0
                    if rr < 1 or rr >= nrows - 1 or cc < 1 or cc >= ncols - 1:
                        continue
                    if np.isnan(WSE_Out_stream[rr, cc]) or WSE_Out_stream[rr, cc] <= E[rr, cc]:
                        continue
                    if (rr, cc) in all_interior_cells:
                        continue

                    for dr1, dc1 in get_neighbor_offsets():
                        nr = rr + dr1
                        nc = cc + dc1
                        if np.isnan(WSE_Out_stream[nr, nc]) or WSE_Out_stream[nr, nc] <= E[nr, nc]:
                            interior_boundary.append((rr, cc))
                            all_interior_cells.add((rr, cc))
                            break

            # break

            # Plot the new_interior_boundaryfor debugging
            # import matplotlib.pyplot as plt
            # boundary_r = [tup[0] for tup in new_interior_boundary]
            # boundary_c = [tup[1] for tup in new_interior_boundary]
            # min_r = np.min(boundary_r)
            # max_r = np.max(boundary_r)
            # min_c = np.min(boundary_c)
            # max_c = np.max(boundary_c)
            # plt.imshow(WSE_Out_stream, cmap='viridis')
            # plt.scatter(boundary_c, boundary_r, color='red', label='New Interior Boundary')
            # plt.colorbar(label='WSE_Out_stream')
            # plt.title(f'Stream ID {stream_id} WSE_Out_stream with New Interior Boundary')
            # plt.xlabel('Column Index')
            # plt.ylabel('Row Index')
            # plt.legend()
            # # Filter plot to the bounding box of the new interior boundary plus a buffer
            # plt.xlim(max(min_c - 10, 0), min(max_c + 10, ncols))
            # plt.ylim(max(min_r - 10, 0), min(max_r + 10, nrows))
            # plt.show()

        # Here, let us plot WSEout stream for debugging
        # import matplotlib.pyplot as plt
        # plt.imshow(WSE_Out_stream, cmap='Blues')
        # plt.colorbar(label='WSE_Out_stream')
        # plt.title(f'Stream ID {stream_id} WSE_Out_stream')
        # plt.xlabel('Column Index')
        # plt.ylabel('Row Index')
        # plt.show()

        # Make a raster to and plot the main channel not in stream for debugging
        # main_channel_raster = np.zeros((nrows, ncols), dtype=np.float32)
        # for rr, cc in main_channel_not_in_stream:
        #     main_channel_raster[rr, cc] = 1.0
        # import matplotlib.pyplot as plt
        # plt.imshow(main_channel_raster, cmap='Reds')
        # plt.colorbar(label='Main Channel Not in Stream')
        # plt.title(f'Stream ID {stream_id} Main Channel Not in Stream')
        # plt.xlabel('Column Index')
        # plt.ylabel('Row Index')
        # plt.show()

        # add the WSE_Out_stream to WSE_Out taking the maximum value when WSE_Out values are not NaNs.
        for rr in prange(nrows):
            for cc in range(ncols):
                ws = WSE_Out_stream[rr, cc]
                if np.isnan(ws) or ws <= -9998.0:
                    continue
                if np.isnan(WSE_Out[rr, cc]):
                    WSE_Out[rr, cc] = ws
                elif ws > WSE_Out[rr, cc]:
                    WSE_Out[rr, cc] = ws

    return WSE_Out