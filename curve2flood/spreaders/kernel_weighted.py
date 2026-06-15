import numpy as np
from numba import njit

@njit(cache=True)
def FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use):
    FourMatrix = np.full((3, 3), 4)
    
    # JLG commented this out because of an error but not sure the fix is correct
    # nrows_local = r_max - r_min + 2
    # ncols_local = c_max - c_min + 2
    # FloodLocal = np.zeros((nrows_local, ncols_local))
    nrows_local = np.int32(r_max - r_min + 2)
    ncols_local = np.int32(c_max - c_min + 2)
    FloodLocal = np.zeros((nrows_local,ncols_local), dtype=np.float32)
    
    FloodLocal[1:nrows_local-1,1:ncols_local-1] = np.where(E_Box<=WSE,1,0)
    
    # JLG commented this out because of an error but not sure the fix is correct
    #This is the Stream Cell.  Mark it with a 4
    # FloodLocal[(r_use-r_min+1),(c_use-c_min+1)] = 4 
    r_idx = int(r_use - r_min + 1)
    c_idx = int(c_use - c_min + 1)
    FloodLocal[r_idx, c_idx] = 4

    
    #Go through and mark all the cells that 
    for r in range((r_use-r_min+1),nrows_local-1):
        for c in range((c_use-c_min+1),ncols_local-1):
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix
    for r in range((r_use-r_min+1), 0, -1):
        for c in range((c_use-c_min+1), 0, -1):
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix
    
    for r in range(1, nrows_local-1):
        for c in range(1, ncols_local-1):
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix
    
    return np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>3.0,1.0,0.0)

@njit(cache=True)
def create_kernel_weighted_spread_map(
    RR,
    CC,
    T_Rast,
    W_Rast,
    S_Rast,
    E,
    B,
    nrows,
    ncols,
    sd,
    TW_m,
    dx,
    dy,
    LocalFloodOption,
    COMID_Unique_TW: dict,
    COMID_Unique_Depth: dict,
    WeightBox,
    TW_for_WeightBox_ElipseMask,
    TopWidthPlausibleLimit,
    Set_Depth,
):
    COMID_Averaging_Method = 0

    WSE_Times_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    Slope_Times_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)
    Total_Weight = np.zeros((nrows + 2, ncols + 2), dtype=np.float32)

    #Now go through each cell
    num_nonzero = len(RR)
    for i in range(num_nonzero):
        r = RR[i]
        c = CC[i]
        r_use = r
        c_use = c
        E_Min = E[r,c]
        
        COMID_Value = B[r,c]
        if Set_Depth>0.0:
            WSE = float(E[r_use,c_use] + Set_Depth)
            if S_Rast is not None:
                SLOPE = float(S_Rast[r_use,c_use])
            COMID_TW_m = TopWidthPlausibleLimit
        elif COMID_Averaging_Method!=0:
            #Get COMID, TopWidth, and Depth Information for this cell
            COMID_Value = B[r,c]
            # keys are int32, values are float32
            if COMID_Value in COMID_Unique_TW:
                COMID_TW_m = COMID_Unique_TW[COMID_Value]
            else:
                COMID_TW_m = np.float32(0.0)

            if COMID_Value in COMID_Unique_Depth:
                COMID_D = COMID_Unique_Depth[COMID_Value]
            else:
                COMID_D = np.float32(0.0)
            WSE = float(E[r_use,c_use] + COMID_D)
            if S_Rast is not None:
                SLOPE = float(S_Rast[r_use,c_use])
        else:
            #These are Based on the AutoRoute/ARC Results, not averaged for COMID
            WSE = np.round(W_Rast[r-1,c-1], 2)  #Have to have the '-1' because of the Row and Col being inset on the B raster.
            COMID_TW_m = T_Rast[r-1,c-1]
            if S_Rast is not None:
                SLOPE = S_Rast[r-1,c-1]

        if COMID_TW_m < 0.00001 or (WSE - E[r,c]) < 0.001:
            continue

        # give the TW for the weightbox the median if its smaller than the median.
        if COMID_TW_m > TW_m:
            COMID_TW_m = TW_m

        #This is how many cells we will be looking at surrounding our stream cell
        COMID_TW = int(max(np.round(COMID_TW_m / dx), np.round(COMID_TW_m / dy)))

        
        # Find minimum elevation within the search box
        if sd >= 1:
            for rr in range(max(r - sd, 0), min(r + sd + 1, nrows - 1)):
                for cc in range(max(c - sd, 1), min(c + sd + 1, ncols - 1)):
                    if E[rr,cc] > 0.1 and E[rr,cc] < E_Min:
                        E_Min = E[rr,cc]
                        r_use = rr
                        c_use = cc

        r_min = max(r_use - COMID_TW, 1)
        r_max = min(r_use + COMID_TW + 1, nrows + 1)
        c_min = max(c_use - COMID_TW, 1)
        c_max = min(c_use + COMID_TW + 1, ncols + 1)
        
        # This uses the weighting method from FloodSpreader to create a flood map
        # Here we use TW instead of COMID_TW.  This is because we are trying to find the center of the weight raster, which was set based on TW (not COMID_TW).  
        # COMID_TW mainly applies to the r_min, r_max, c_min, c_max
        w_r_min = TW_for_WeightBox_ElipseMask - (r_use - r_min)
        w_r_max = TW_for_WeightBox_ElipseMask + (r_max - r_use)
        w_c_min = TW_for_WeightBox_ElipseMask - (c_use - c_min)
        w_c_max = TW_for_WeightBox_ElipseMask + (c_max - c_use)

        weight_slice = WeightBox[w_r_min:w_r_max, w_c_min:w_c_max]
        if LocalFloodOption:
            #Find what would flood local
            E_Box = E[r_min:r_max,c_min:c_max]
            FloodLocalMask = FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use)
            WSE_Times_Weight[r_min:r_max, c_min:c_max] += (WSE * weight_slice * FloodLocalMask)
            Total_Weight[r_min:r_max,c_min:c_max] += weight_slice * FloodLocalMask
            if S_Rast is not None:
                Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice * FloodLocalMask)
        else:

            # This puts the weights from each cell into the composite arrays.
            cell_wse_weight = WSE * weight_slice                
            WSE_Times_Weight[r_min:r_max, c_min:c_max] += cell_wse_weight
            Total_Weight[r_min:r_max,c_min:c_max] += weight_slice

            # This makes a weighted slope raster that can be used for velocity estimates
            if S_Rast is not None:
                Slope_Times_Weight[r_min:r_max, c_min:c_max] += (SLOPE * weight_slice)

    # These are the cells that we want to flood based on the weighted WSE being greater than the elevation, and also making sure the elevation is valid and that we have some weight there.
    valid_candidate = (
        (E > -9998.0) &
        (WSE_Times_Weight > E * Total_Weight)  # Keeps the values where WSE is greater than E, which means it would be flooded
    )

    WSE_array = np.where(valid_candidate, WSE_Times_Weight / Total_Weight, np.nan).astype(np.float32)
    return WSE_array, Total_Weight, Slope_Times_Weight