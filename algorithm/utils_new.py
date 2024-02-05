"""
Functions used in multi-time scale Paleoclimate Data Assimilation algorithm

MIT License
Copyright 10.12.2023 Mathurin Choblet

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import xarray as xr
import bisect
import cftime
import pandas as pd
import os
from numba import njit,prange
import tqdm

def model_time_subselect(xarray,avg,check_nan=False):
    """
    Function which slices the model data according to the avg start month, 
    or a specific season that will be reconstructed.
    Used for the annual mean calculation or the precipitation weighting for the d18O.
    
    Avg either an integer (starting month for annual mean), or a list of integers for specific months.
    Check_nan used for model data which has missing time steps. Data is replaced with previous time steps.
    """
    xarray=xarray.copy()
    xarray_time_copy=xarray.time

    #hack for ihadcm3 where data is missing
    if check_nan:
        xarray=xarray.resample(time='MS').mean('time')
        #print('Checking prior for all nans/zeros in a time stepp')
        for i,t in enumerate(xarray.time):
            x=xarray.sel(time=t)
            nans=np.count_nonzero(np.isnan(x))
            if nans>0:
                #print('Only nans in year', t.values, '. Replaced values with previous year')
                xarray.loc[dict(time=t)]=xarray.isel(time=(i-1))
            #additional check, if alls zeros
            all_zeros = not np.any(x)
            if all_zeros:
                #print('Only zeroes in year', t.values, '. Replaced values with previous year')
                xarray.loc[dict(time=t)]=xarray.isel(time=(i-1))
    
    if avg==None or avg=='None':
        #calculate mean
        # Calculate the numerator #here resample so that we correctly get out time dimension
        xarray = xarray.resample(time="YS").mean(dim='time')        
        #check beginning/end, full year?    
        fm=xarray_time_copy.time.dt.month[0] #first month
        lm=xarray_time_copy.time.dt.month[-1] #last month
        if fm!=1 and fm!=2:
            xarray=xarray.isel(time=slice(1,None))
        if lm!=12 and lm!=11:
            xarray=xarray.isel(time=slice(None,-1))

        #add info about mean type in attributes
        xarray.attrs['Yearly mean type']='Standard January to December'
    
    #second case start month given (e.g. 4 for April to March averages)
    elif isinstance(avg,int): 
        #Slice the start
        #get first year/month
        fm=xarray.time.dt.month[0].values
        years=np.unique(xarray.time.dt.year.values)

        #first month in Array smaller than start month: slice from avg-month in first year
        #slicing on indices (assumes that no month is missing), else I would need to create a cftime.Datetime-object where I would also need to know the calendar type.
        #cal=xarray.time.to_index().calendar
        if fm<=avg:
            xarray=xarray.isel(time=slice(int(avg-fm),None))
        else:
            xarray=xarray.isel(time=slice(int(12-fm+avg),None))
        
        #Slice the End
        #last available month, needs to be==avg-1 (special case january=1 -> lm==12)
        lm=xarray.time.dt.month[-1].values
        #slice in last year
        if lm>=avg:
            xarray=xarray.isel(time=slice(None,int(avg-lm-1)))
        else:
            xarray=xarray.isel(time=slice(None,int(avg-lm-12-1)))
        
    elif isinstance(avg,np.ndarray) | isinstance(avg,list):
        """
        calculate average for months given in list (e.g [4,5,6], [11,12,1], ...)
        WARNING: If a month is just not existent, this will go wrong (just using coarsen method)
        """
        assert np.ndim(np.squeeze(avg))==1, "<avg> invalid, can be None, integer (1-12) or 1-d list/nd.array"
        #make sure avg is a list
        avg=list(avg)

        #Time-Slice
        #BEGINNING
        fm=xarray.time.dt.month[0]
        years=np.unique(xarray.time.dt.year.values)
        #start month in avg sequence
        sm=avg[0]
        #is start month included in first year?
        
        #first month in Array smaller than start month: slice from avg-month in first year
        if fm<=sm:
            xarray=xarray.isel(time=slice(int(sm-fm),None))
        #slice in second year
        else:
            xarray=xarray.isel(time=slice(int(12-fm+sm),None))

        #Slice the End
        #last available month, needs to be==avg-1
        lm=xarray.time.dt.month[-1].values
        #month to end with
        em=avg[-1]
        #slice in last year
        if lm>em:
            #if last month in prior larger than end month in avg slice from the end
            xarray=xarray.isel(time=slice(None,int(em-lm)))
        elif lm<em:
            #else go one year further back
            xarray=xarray.isel(time=slice(None,int(em-lm-12)))
            
        if len(xarray.dims)>1:
            xarray=xarray[xarray.time.dt.month.isin(avg),:,:] #(where extremely slow)
        else:
            xarray=xarray[xarray.time.dt.month.isin(avg)]

    return xarray
    
def precip_weighted(c,avg=4):
    """
    computes precipitation weighted d18O for each grid cell
    avg: starting month for annual mean
    """
    path_prec=c.basepath+c.vp['prec']
    path_d18O=c.basepath+c.vp['d18O']
    prec=load_prior(path_prec,'prec',bounds=c.regional_bounds,time_cut=c.prior_cut_time) 
    d18O=load_prior(path_d18O,'d18O',bounds=c.regional_bounds,time_cut=c.prior_cut_time) 
    #if 'hadcm' in path_prec.lower(): check_nan=True
    #else:
    check_nan=False
    
    #check nan loops over time and replace with values of previous year. this is ugly, but the only way I see
    prec_sub=model_time_subselect(prec,avg,check_nan)
    d18O_sub=model_time_subselect(d18O,avg,check_nan)
    
    #mask nans in precip
    cond = prec_sub.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    
    if (type(avg)==int or type(avg)==float) or len(avg)==1:
        l=12
    else:
        l=len(avg)
    
    m=(prec_sub*ones).coarsen(time=l).construct(time=('year','month'))
    summa=m.sum('month')
    weights=m/summa
    weights=weights.stack(time=('year','month')).transpose('time',...)
    #final step. sum instead of mean important
    d18_weighted=(d18O_sub*weights.values).coarsen(time=l).sum('time')
    
    #FALL BACK FOR NO PRECIP VALUES? -> use non weighted d18O
    d18=d18O_sub.coarsen(time=l).mean()
    d18_weighted=xr.where(d18_weighted==0,d18,d18_weighted)
    
    #safety measure: count nans (might destroy everything) DELETE LATER
    s=np.sum(np.isnan(d18_weighted.values))
    if s>0:
        print('NANS encountered in d18_weighted! Check that!')
        
    #NOTE THAT THE TIME AXIS DOES NOT LOOK AS IT SHOULD, BUT THE VALUES ARE CORRECT!
    d18_weighted=d18_weighted.resample(time='YS').mean('time')
    d18_weighted.name='d18O_weighted'
    
    #####TO DO? CHECK FOR ALL zeros and all nans in one year -> (all zeros important in precipitation)
    if check_nan:
        #print('Checking prior for all nans/zeros in a time step')
        for i,t in enumerate(d18_weighted.time):
            x=d18_weighted.sel(time=t)
            nans=np.count_nonzero(np.isnan(x))
            if nans>0:
                #print('Only nans in year', t.values, '. Replaced values with previous year')
                d18_weighted.loc[dict(time=t)]=d18_weighted.isel(time=(i-1))
            #additional check, if alls zeros
            all_zeros = not np.any(x)
            if all_zeros:
                #print('Only zeroes in year', t.values, '. Replaced values with previous year')
                d18_weighted.loc[dict(time=t)]=d18_weighted.isel(time=(i-1))            
    return d18_weighted
    
    
    
def prior_block_mme(prior,bs,idx_list):
    """
    Create a Prior-Block, which extends the randomized yearly prior to <bs> continous years
    along the third dimension. 
    Input: 
        - Prior: List of ( Flattened version (time, #gridboxes/values) <- Length of second axis important.) for each model
        - bs: #blocksize
        - idx: random indices, Array of shape(num_models, reps, nens)
    Output:
        - block (bs,nens, #values)
        
    #UPDATE 21.09.22: adapted to multi-model prior. Also runs when just using one single model.
    #UPDATE 18.06.23: I enforce the ensemble mean over the block size to be zero,
    """
    nens=idx_list.shape[-1]
    num_models=idx_list.shape[0]
    stack_size=prior[0].shape[1]
    
    block=np.empty((bs,nens*num_models,stack_size))
    for n in range(num_models):
        idx=idx_list[n]
        mi=nens*n
        ma=nens*(n+1)
        for i in range(bs):
            block[i,mi:ma]=prior[n][idx+i,:]
        #demean for ensemble mean to be really zero at each grid point
        #introduce axis, because else there is a broadcasting problem
        block[:,mi:ma]=block[:,mi:ma]-np.expand_dims(block[:,mi:ma].mean(axis=1),axis=1)
    return block
    
    
def random_indices(nmem,length,reps=1,seed=None):
    """
    Produces array of random indices (for prior selection in Monte Carlo approach or proxy randomization)
    
    Input:
        nmem: how many members of length-members are randomly selected
        length: length of initial array (e.g. 1000 for full prior, or 163 for sisal database)
        reps: repetitions
        seed: Integer (Number you can set to make reproducable results (works over loops))
    
    Output: [reps,nmem] 2d nparray (loop over first index to get the members)
    """
    rng=np.random.default_rng(seed)
    array=np.zeros((reps,nmem))
    for i in range(reps):
        array[i,:]=rng.choice(length,size=nmem, replace=False)
    #if only one repetition get rid of unwanted extra dim <- No, need it for the wrapper!
    #array=array.squeeze()
    return array.astype(int)
    
    
def load_prior(path,var,conversion=True,bounds=None,time_cut=None):
    """
    Loads prior Data, renames latitude/longitude/time dime.
    Doesn't check if all dims that are needed are actually contained!
    If Longitudes go from -180 to 180 they are changed to 0-360

    Input: Path to nc-File, Variable
    conversion=True only relevant for precipitation (to mm/month)
    Output: DataArray
    """
    #print(path)
    array=xr.open_dataset(path,use_cftime=True)[var]
    dims=array.dims
    if ('latitude' in dims) & ('longitude' in dims):
        array=array.rename({'latitude':'lat','longitude':'lon'})
    if ('t' in dims) & ('time' not in dims):
        array=array.rename({'t':'time'})
    if array.lon.min() < -10:
    	#convert lon from -180-180 to 0-360
    	array.coords['lon'] = array.coords['lon'] % 360
    	array= array.sortby(array.lon)
    
    #if we are dealing with precipitation and conversion is True convert it to mm/month
    #also do the same for evaporation!
    if var in ['pr','prec','precip','precipitation','pratesfc','evap']:
        if conversion:
            array=precipitation_conv(array)
    #remove some unnecearray lev... dimension of shape 1
    array=array.squeeze(drop=True)
    
    #bring into uniform shape
    array=array.transpose('time','lat','lon')
    
    #rename
    array=array.rename(var)
    
    array=regioncut(array,bounds)
    if time_cut !=None:
        try:
            array=array.sel(time=slice(time_cut[0],None))
        except:
            array=array.sel(time=slice(time_cut,None))
    return array

def regioncut(array,bounds):
    """
    Reduce input climate field to a specific subregion.
    """
    if bounds!=None:
        if bounds!=False:
            #latitudes selection
            array=array.where((array.lat>= bounds[0][0] ) & (array.lat <= bounds[0][1]), drop=True)            
            #longitude selection
            #needs special treatment (e.g 330, 30) or (0,50). 
            lons=bounds[1]
            lon=array.lon
            if lons[1]<lons[0]:          
                sel_lon_1= array.where( (lon >= lons[0] ), drop=True)
                sel_lon_2= array.where( (lon <= lons[1]), drop=True)
                sel_lon=np.concatenate([sel_lon_1.values,sel_lon_2.values])
                array=array.where(array.lon==sel_lon,drop=True)
            else:    
                array=array.where((lon >= lons[0] ) & (lon <= lons[1]),drop=True)
    return array



def precipitation_conv(array):
    """
    Converts monthly precipitation data into mm/month.
    Array: xarray-DataArray for precipitation which has unit attribute.
    Function automatically detects calendar type and the month lenghth (relevant for kg/m^2s)
    """
    #get unit and month length
    try:
        unit=array.attrs['units']
    except:
        raise AttributeError('Precipitation does not have a unit attribute. Conversion failed!')
    try:
        month_length =array.time.dt.days_in_month
    except:
        raise AttributeError('Data does not contain how many days there are in one month. Crucial')

    if unit=='mm/month':
        #print('No conversion needed!')
        return array
    
    if unit=='m/s':
        #assuming 30 days per month convert to mm/month
        array=array*1000*60*60*24*month_length
    
    elif unit=='mm/day':
        array=array*month_length
    
    elif unit in ['kg/m^2s','kg/m2s','kg/m^2/s','kg/m2/s','kg m-2 s-1']:
        #kg to mm,broadcasting done correctly by xarray
        array=array*(60*60*24*month_length)
    else:
        raise NameError('Unkown precipitation units! (kg/m^2s, kg/m2s, kg/m^2/s,kg/m2/s, kg m-2 s-1,mm/day or m/s)', unit)
    #change unit attribute   
    array.attrs['units']='mm/month'
    
    return array
    
def prior_preparation(cfg,variabs,dictionary):
    """
    load_priors and take annual mean + all seasonally required means (for PSMs) !
    
    Prior is the data array with the variables as they are used in the end.
    In case 'd18O' is parts of the variables: compute precipitation weighted annual mean.
    
    save all variables in one Dataset and separately save attributes
    also save the raw (monthly) prior, in case it is needed for the PSM
    """
    prior=[]
    prior_raw=[]
    print('Load prior data')
    for v,p in cfg.vp.items():
        #load only variables which have path not set to None
        if p!=None:
            #if v!='spei':
                #print('Load',v,' from ',p)
            p=cfg.basepath+p
            data=load_prior(p,v,bounds=cfg.regional_bounds,time_cut=cfg.prior_cut_time) 
            
            #compute annual mean ac
            #print('computing average for avg:', cfg.avg)         
            if 'd18o' in v.lower():
                #mix of prec_weighting + annual_mean function
                #print('compute precipitation weighted d18O')
                data_m=precip_weighted(cfg,avg=cfg.avg)
            else:
                path_lower=p.lower()
                """
                if 'hadcm' in path_lower:
                    check_nan=True
                    data=data.resample(time='MS').mean('time')
                else:
                """
                check_nan=False        
                data_m=annual_mean(data,avg=cfg.avg,check_nan=check_nan)
            #LEGACY
            #special treatment for spei, which is already precomputed
            """
            else:
                p=cfg.basepath+p
                data=xr.open_dataset(p)
                #print('REMOVE FOLLOWING PRINTS!')
                if cfg.avg==[12,1,2]:
                    #print('Use DJF mean of precomputed spei')
                    data_m=data['spei_djf']
                elif cfg.avg==[6,7,8]:
                    #print('Use JJA mean of precomputed spei')
                    data_m=data['spei_jja']
                else:
                    data_m=data['spei_ann']
                
                data_m=data_m.rename('spei')
                data_m=regioncut(data_m,cfg.regional_bounds)
                if cfg.prior_cut_time !=None:
                    try:
                        data_m=data_m.sel(time=slice(cfg.prior_cut_time[0],None))
                    except:
                        data_m=data_m.sel(time=slice(cfg.prior_cut_time,None))
                #remove years where only nans
                data_m=data_m.dropna('time',how='all')
                
                #eventually adjust time to prior
                try:
                    data_m=data_m.sel(time=slice(prior[0]['time'].values[0],prior[0]['time'].values[-1]))
                except:
                    pass
                
                
                #replace nans by constant zero to avoid nan problem
                mask=np.isnan(data_m)
                data_m.data[mask.values]=0
                
             """
            #DEMEAN HERE!
            data_m=data_m-data_m.sel(time=slice(cfg.anomaly_time[0],cfg.anomaly_time[1])).mean('time')
            
            #hotfix for jja means when proxy means start from april, I need to cut t the last one away
            try: 
                if cfg.avg[0]==6:
                    data_m=data_m.transpose('time',...)[:-1] 
            except:
                 pass
            #print(data_m['time'].values[0],data_m['time'].values[-1])
            prior.append(data_m)
            #prior_raw.append(data)

    #create Dataset (This will always be a Dataset not Dataarray, also for only one variable)    
    prior=xr.merge(prior)
    #copy attributes
    attributes=[]
    for v in prior.keys():
        a=prior[v].attrs
        attributes.append(a)
    
    #compute seasonal means and put them in a dictionary.
    print('Compute seasonal means')
    seasonal_dict={str(v):{} for v in variabs}
    for k,v in dictionary.items(): #k:variable
        path=cfg.basepath+cfg.vp[k]
        #if k!='spei':
        raw_data=load_prior(path,k,bounds=cfg.regional_bounds,time_cut=cfg.prior_cut_time)
        #else: raw_data=xr.open_dataset(path)
        """
        if 'hadcm' in path.lower():
            check_nan=True
            #ensure the data is monthly
            raw_data=raw_data.resample(time='MS').mean('time')
        else:
        """
        check_nan=False    
        for m in v:
#            if k!='spei':
            trigger=False #see below
            if m=='None' or m=='Annual':
                #not necessarly reconstruction time! (think of annual proxies, but reconstructing djf/jja mean!!!)
                m=int(cfg.avg_proxies)i
            elif type(m)==str:
                m=[int(num) for num in m.split(',')]
                if m[0]<int(cfg.avg_proxies): trigger=True
                    #avoid situation where the season mean is longer than it should, e.g. cariaco basin mean for [3,4,5] -> prior length 1000, but c.avg is 4 -> length 999. In this case cut the last.
                    #more cases would be conceivable. so whenever using seasonal means one has to be aware that this might cause a problem (would be easiest to treat all proxies as )
            elif type(m)==list:
                m=[int(num) for num in m]
                if m[0]<int(cfg.avg_proxies): trigger=True
            if 'd18o' in k.lower():
                season_mean=precip_weighted(cfg,avg=m)
            else:
                season_mean=annual_mean(raw_data,m,check_nan)
            if trigger: season_mean=season_mean.transpose('time',...)[:-1] #see explanation above
            #print('Seasonal dict: ', m, ' mean for ', k)
            
            #hotfix
            if m==cfg.avg:
                try:
                    if cfg.avg[0]==6:
                        season_m=season_m.transpose('time',...)[:-1] 
                except:
                    pass
            """
            else:
                #print('Seasonal dict: ', m, ' mean for ', k)
                if m=='None' or m=='Annual' or m=='4': season_m=raw_data['spei_ann'].transpose('time',...); m='4'
                elif m=='[12,1,2]':season_m=raw_data['spei_djf'].transpose('time',...)
                elif m=='[6,7,8]': season_m=raw_data['spei_jja'].transpose('time',...)
            #replace the time according to the prior (else there might be annoying by one shifts)
                #print('Seasonal dict: ', m, ' mean for ', k)
            """
            try:
                season_mean['time']=prior['time'].values
            except:
                raise Exception('Error in computation of seasonal mean')
                #import pdb; pdb.set_trace()

            #DEMEAN HERE!
            season_mean=season_mean-season_mean.sel(time=slice(cfg.anomaly_time[0],cfg.anomaly_time[1])).mean('time')
            seasonal_dict[k][str(m).replace("[","").replace("]","").replace(" ","")]=season_mean #replace hack needed for later in psm
    return prior, attributes, seasonal_dict

def annual_mean(xarray,avg,check_nan=False):
    """
    COMPUTE ANNUAL/SEASONAL MEAN for prior data according to avg: (either integer=start month, or list of integers for specific months/season)
    main part of function has been outsource to model_time_subselect
    """
    xarray_attrs_copy=xarray.attrs
    name=xarray.name
    xarray=model_time_subselect(xarray,avg)
    
    if isinstance(avg,int) or len(avg)==1:
         xarray=xarray.coarsen(time=12,boundary='trim').mean('time') #Not resample! Else I get too many values here
    elif isinstance(avg,np.ndarray) | isinstance(avg,list):
        xarray=xarray.coarsen(time=len(avg),boundary='trim').mean('time')
        #the label is set to the center entry (would prefer left, but ok)
    
    #to be sure to always have the beginning of the year in the time axis
    xarray=xarray.resample(time="YS",closed='left',label='left').mean()    
    #add original attributes (get lost throughout the process)
    xarray.attrs=xarray_attrs_copy
    xarray=xarray.rename(name) 
    """ LEGACY
    check_nan=True
    if check_nan:
        #print('Checking prior for all nans/all zeros at one time step')
        for i,t in enumerate(xarray.time):
            x=xarray.sel(time=t)
            nans=np.count_nonzero(np.isnan(x))
            if nans>0:
                #print('Dropped year', t.values, 'due to nans')
                #xarray=xarray.where(xarray.time!=t, drop=True)
                #print('Only nans in year', t.values, '. Replaced values with previous year')
                xarray.loc[dict(time=t)]=xarray.isel(time=(i-1))
            #additional check, if alls zeros
            all_zeros = not np.any(x) #if any value is !=0 (no matter if float/int), this is true (and thus false with not)
            if all_zeros:
                #print('Only zeroes in year', t.values, '. Replaced values with previous year')
                xarray.loc[dict(time=t)]=xarray.isel(time=(i-1)) 
    """ 
    return xarray

def make_equidistant_target(data,target_time,target_res,method_interpol='nearest',filt=True,min_ts=1):
    """
    Takes a proxy timeseries "data" (fully resolved,with nans in between if no value available) and resamples it equidistantly to the (equidistant) target timeseries
    "target_time"  (DataArray of cftime-objects, we need .dt.time.year accessor), which has the resolution "target_res" (consistency with target_time is not checked).
    We usualy set the target_res to the median resolution.
    
    The resampling procedure is adapted from the Paleospec R-package: https://github.com/EarthSystemDiagnostics/paleospec/blob/master/R/MakeEquidistant.R.
    The time resolution is based on yearly data. Other time resolution (monthly) would require adapting the filtering part.
    
    Code consists of the following steps.
        0. Duplicate first non-nan-data point if this required by target_time spacing
        1. Resample and interpolate time series to 'min_ts'-resolution (yearly makes sense in our case). Nearest neighbor or linear interpolation!
        2. Low_pass filter resampled time_series (High order Butterworth filter used in original R-package, I use filtfilt to avoid time lag)
        3. Resample to target resolution
    
    Comments:
        1. Be aware that some proxy records have huge jumps without data in between. The resampled values there are not meaningful and need to be masked separately.
        2. Use xarray > v2022.06.0 to make use of fast resampling operation (but slowness of old version not really a problem for our time-lengths)
    
    Example:
        Given some time-series with measurements at time [4,9,14,19,24], which we treat as mean-values for the time range centered on these times.
        We want to resample it equidistanced for the times [0,5,10,15,20,25]. These target labels are actually the left edge of a time block 
        (in the DA we effectively reconstruct the mean of the years [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24]])
        Therefore when using the xarray resample method in the final step (down-sampling) it is important to set closed='left' (and eliminate the last element) for logical consistency.
    """
    #drop nan entries in data, extract data which are not nan
    #without dropping nans interpolation wont work
    data=data.dropna('time')
    vals=data.values

    #time_values and years contained in original time_series
    time=data.time.values
    time_years=data.time.dt.year.values

    #For the first year included in proxy timeseries find the nearest year in target_time, which is smaller than the first year.
    #Repeat first data value and append this value and its time to the values. Do not do this if the first year is part of the target_time.
    first_year=time_years[0]
    target_years=target_time.time.dt.year.values

    #find by modulo calcuation and search sorted (could also create new Datetimeobject)
    #take into consideration the start time that might be shifted
    start=first_year-first_year % target_res + target_years[0]%target_res
    if start!=first_year:
        idx = np.searchsorted(target_years, first_year, side="left")
        time_add=target_time[idx-1].values

        #insert time and duplicate first value
        time=np.insert(time,0,time_add)
        vals=np.insert(vals,0,vals[0])

    vals_new=xr.DataArray(data=vals,coords=dict(time=time))

    #1. resampling (upsampling) and interpolating (upsampling)
    min_ts=str(min_ts)+'YS'
    try:
        upsampled=vals_new.resample(time=min_ts).interpolate(method_interpol)
    except:
        if len(vals_new.time)==1:
            #case of only one value, then no interpolation (resampling fails)
            #already have correct time, checked in start!=first_year
            upsampled=vals_new
        else:
            raise Exception("Error in time series resampling and interpolation with length>1)
            #import pdb; pdb.set_trace()

    ##Fill nans (already done in previous step)
    #upsampled=upsampled.interpolate_na('time',method='linear')
    
    #2. LOW PASS FILTER for resampled time series (avoid aliasing)
    from scipy.signal import butter, lfilter, filtfilt
    def butter_lowpass(cutoff, fs, order=6, kf=1):
        # kf:  scaling factor for the lowpass frequency; 1 = Nyquist, 1.2 =
        #' 1.2xNyquist is a tradeoff between reducing variance loss and keeping
        #' aliasing small
        #fs is basic timestep (min_ts)
        #nyquist frequency
        nyq = 0.5 * fs 
        normal_cutoff = cutoff / nyq * kf
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=6):
        #filtfilt does not introduce a time-lag in comparison to butterworth 
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        #y = lfilter(b, a, data)
        return y

    cutoff=1/target_res
    fs=1 #yearly base frequency
    if filt==True:
        try:
            up_filt=butter_lowpass_filter(upsampled,cutoff,fs,order=4)
        except:
            #for short reconstruction time range (e.g.1900-1999), the resampling can not work out, then just take the value as it is
            up_filt=upsampled
    else:
        up_filt=upsampled

    ###3. RESAMPLE TO TARGET RESOLUTION
    #string for resampled option 'YS': year start is very important
    target_res_st=str(target_res)+'YS'
    #convert up_filt to Dataarray in order to use resample method
    up_filt=xr.DataArray(up_filt, coords=dict(time=upsampled['time']))
    resampled=up_filt.resample(time=target_res_st,closed='left').mean('time')
    #reindex time to get back to the global full timescales (non existing values filled with nan)
    final=resampled.reindex(time=target_time)

    return final


def mask_the_gap_alt(resampled_ts, original_ts, time_res,tol):
    """
    Function for masking gaps after the resampling.
    It looks for gaps in the original timeseries and masks them in the resampled timeseries.
    
    Input: 
        resampled: equidistant time-series (from proxy_db beginning to end, containing nans at beginning/start)
        time_res: Resolution of resampled time-series
        original_ts: original time_series of proxy from proxy_db-table (containing nans in between measurement if measurement not yearly)
        tol(erance): size of gap with respect to time_res (tol*time_res), here it is a factor that is multiplied with each timeresolution
        
    """
    #copy
    resampled_ts=resampled_ts.copy()
    
    #maximum allowed gap
    max_gap=tol*time_res

    #screen original timeseries for jumps
    original_ts_years=original_ts.dropna('time').time.dt.year
    gaps=np.abs(np.array(original_ts_years)[1:]-np.array(original_ts_years)[:-1])

    #find index where gap > max_gap (left end)
    args=np.argwhere(gaps>max_gap).flatten()

    #select according years
    #starts=original_ts.dropna('time').time[args].dt.year
    #ends=original_ts.dropna('time').time[args+1].dt.year

    starts=original_ts_years[args]
    ends=original_ts_years[args+1]
    
    target_time_ts=resampled_ts['time']
    target_years_ts=resampled_ts['time'].dt.year

    #in target years, find the ones that are larger than start and smaller than end
    #bisectional search is the most efficient way, list comprehension would be orders of magnitude slower
    #we use bisect-righ for findin the elements (right/left indicates if index to right/left is chosen for equality)
    #For the end we keep the first to the left of the end (because it's influenced by the measurement to the right) and thus
    #select the penultimate one
    
    for ii,t in enumerate(starts):

        #find indices with bisect right
        start_idx=bisect.bisect_right(target_years_ts,starts[ii]) 

        #end index,-2 because slice also selects last element
        end_idx=bisect.bisect_right(target_years_ts,ends[ii])-2

        resampled_ts.loc[dict(time=slice(target_time_ts[start_idx],target_time_ts['time'][end_idx]))]=np.nan
        
    return resampled_ts

#AVERAGE AND MEAN CALCULATOR
@njit(parallel=True)
def anomean_with_numba(array_in,size):
    """
    Calculate anomaly and mean of array_in along axis0.
    
    Input:
        Array_in: (blocksize, values_vector[1]*nens) (Is not changed
        size: subblock size
    The second dimension of array_in is usually ~10^7, so array_in-array_in.mean(axis=0) is usually too slow,
    that's why I had to come up with this numba solution.
    """
    s=array_in.shape
    number=s[0]//size
    #Initialize array for mean and anomaly (latter has original size)
    mean=np.empty((number,s[1]))
    anom=np.empty_like(array_in)
    #looping over blocks if size!=s[0]
    if number>1:
        #loop over all points along axis1
        for i in prange(s[1]):
            for j in prange(number):
                vals=array_in[j*size:(j+1)*size,i]
                m=vals.mean()
                mean[j,i]=m
                anom[j*size:(j+1)*size,i]=vals-m
    else:
        for i in prange(s[1]):
            vals=array_in[:,i]
            m=vals.mean()
            mean[0,i]=m
            anom[:,i]=vals-m        
    return mean,anom

def load_proxies(c):
    """
    Load proxy record data and bring it into the right format.
    Output: proxy_db_all (list, with one entry for each Proxy-DB input file
    """
    start=c.proxy_time[0]
    end=c.proxy_time[1]
    time=xr.cftime_range(start=start,end=end,freq='YS',calendar='365_day')
    proxy_db_all=[]

    for ip,p in enumerate(c.obsdata):
        proxy_db=xr.open_dataset(p,use_cftime=True)#.squeeze(drop=True)
        #slice according to proxy time above
        proxy_db=proxy_db.sel(time=slice(start,end))
        variabs=np.array(list(proxy_db.variables))
        #eliminate potentiall empty proxies (without values in time of interest)
        for idx,s in enumerate(proxy_db.site):
            avail_times=proxy_db.sel(site=s).dropna('time').time.values
            if len(avail_times)==0:
                proxy_db=proxy_db_y.drop_sel(site=s)
        if 'geo_meanLat' in variabs:
            proxy_db=proxy_db.rename({'geo_meanLat':'lat','geo_meanLon':'lon'})
        if 'geo_siteName' in variabs:
            proxy_db=proxy_db.rename({'geo_siteName':'site'})
        try:
            if c.how_many is not None:              
                prox_mems=c.how_many[ip] # absolute number
                if prox_mems>len(proxy_db.site):
                    prox_mems=len(proxy_db.site)
                prox_idx=random_indices(prox_mems,len(proxy_db.site), reps=1,seed=c.seed)
                proxy_db=proxy_db.isel(site=prox_idx[0])                                      
        except:
            pass
        try:
            proxy_db=proxy_db.to_array().squeeze(dim='variable') #do not squeeze everything! (this is a problem for databases of one single site)
        except:
            pass
        #demean proxy db with respect to anomaly time
        proxy_db=proxy_db-proxy_db.sel(time=slice(c.anomaly_time[0],c.anomaly_time[1])).mean('time')
        proxy_db_all.append(proxy_db)
    return proxy_db_all

def variable_prep(proxy_db_all):
    """
    Go through loaded proxy data and extract all the info needed for the PSMs, namely the name of the climate variables and the seasonalities.
    """
    variab_list=[]
    for db in proxy_db_all:
        variab_list.append(np.unique(db['variable_name'].values))
    variab_list=np.unique(np.concatenate(variab_list))
    #1. GET REQUIRED SEASONALITIES FOR PSMS
    #get a dictionary where I calculate how I need to manipulate the prior
    variab_season_dict={str(v):[] for v in variab_list}
    #first get all the seasons
    for db in proxy_db_all:
        for v in variab_list:
            lookup=variab_season_dict[str(v)]
            seasons=np.unique(db['seasonality'].values[(db['variable_name']==v).values].astype(object))
            if len(seasons)>0:
                for s in seasons:
                    if s not in lookup:
                        variab_season_dict[str(v)].append(s)
    return variab_list,variab_season_dict

def resample_wrap(c,proxy_db_all,HXfull_all_fin):
    """
    Wrapper for resampling proxy records to the desired time frequency. Actual resampling done in make_equidistant_target-function.
    Specifications of resampling procedure stored in c.
    Resampled time series stored in dictionary (key for each time scale)
    Function also computes the proxy error according to SNR for each time scale.
    """
    print('Resample proxy records')
    #2. RESAMPLING PROXIES + COMPUTE ERRORS
    length=int(c.proxy_time[1])-int(c.proxy_time[0])
    #times list used for resampling proxy records
    times_list=[xr.DataArray(xr.cftime_range(start=c.proxy_time[0],periods=(length//int(i)+1),freq=str(i)+'YS',calendar='gregorian'),dims='time') for i in np.sort(np.array(c.timescales).astype(int))]
    #adapt times_list (cut end in case it doesn't fit perfectly with largest block size)
    #I left this out here. I will only do reconstructions that make sense.
    #timeresolution set in c.db_timescales
    #resample proxies: list for each time scale
    mask_ = c.mask #masking tolerance factor (mask_ * time_res is max. gap size)
    mode=c.resample_mode
    timescales=np.array(c.timescales) #make sure it's really a numpy array
    #create list of lists for each proxy_db and each timescale (why?)
    lisst=[]
    lisst_r=[]
    #store data in dictionary for all proxies in this database
    dictionary={}
    dictionary_r={}
    for scale in timescales:
        dictionary[str(scale)]=dict(ts=[],sites=[],num=[])
        dictionary_r[str(scale)]=dict(err=[])
    for ii,db in enumerate(proxy_db_all):
        ts_db=np.array(c.db_timescales).astype(int)[ii]
        #as I only allow for one timescale per database, there is one target time for each database
        idx=np.argwhere(np.array(c.timescales).astype(int)==ts_db).squeeze().item()
        target_time=times_list[idx] 
        if str(ts_db)=='1':
            #don't resample proxies, add the database as is (especially for tree rings and corals)
            dictionary[str(1)]['ts'].append(db.values)
            dictionary[str(1)]['sites'].append(db.site.values)
            dictionary[str(1)]['num'].append(len(db.site))
            lisst.append(dictionary)
            #add error correctly
            sites=db.site
            if c.noise_assum[ii]=='None':
                dictionary_r[str(1)]['err'].append(db.coords['regression_params_var'].values)
            elif c.noise_assum[ii]=='equal':
                mask=np.isin(HXfull_all_fin.site.values,db['site'].values,assume_unique=True) #assume unique important to keep the shape correctly. possible doubled names, think this only occurs for tree rings where i don't use that option
                try: fac=c.equal_fac
                except: fac=1
                dictionary_r[str(1)]['err'].append(HXfull_all_fin[:,mask].var('time').values*fac) #no resample option needed here (.resample(time=(str(ts_db)+'YS')).mean('time'))
            else:
                err=c.noise_assum[ii]
                for iii,s in enumerate(sites):
                    #compute error variance according to snr
                    #val_std=np.std(db.isel(site=iii).dropna('time').values)
                    #compute variance over 200 year rolling window (turned out to be more sensible to do it like that)
                    try:
                        val_var=db.isel(site=iii).sel(time=slice('0850','1850')).dropna('time').rolling(time=200).var(skipna=True).mean(skipna=True).values
                    except: #only required for illmani record which is not long enough for a 200 year window
                        val_var=db.isel(site=iii).sel(time=slice('0850','1850')).dropna('time').var(skipna=True).values
                    if np.isnan(val_var):
                        val_var=db.isel(site=iii).sel(time=slice('0850','1850')).dropna('time').var(skipna=True).values
                    r=val_var/(1+err**2)
                    dictionary_r[str(1)]['err'].append(r)
            lisst_r.append(dictionary_r)
        else:
            #resample + go to higher timescales if c.reuse==True 
            sites_l=[]
            values_l=[]
            sites=db.site
            #try: len(sites)
            #except: sites=np.array([sites.item()]) #hack for single site db (else it breakes)
            for iii,s in enumerate(tqdm.tqdm(sites)):
                data=db.isel(site=iii)
                #If res <4 don't use the lowpass filter.
                if ts_db<4: filt=False
                else: filt=True
                resampled=make_equidistant_target(data,target_time,target_res=ts_db,method_interpol=mode,filt=filt,min_ts=1)
                #mask the gaps
                resampled=mask_the_gap_alt(resampled,data, time_res=ts_db,tol=mask_)
                #cut according to reconstruction time
                resampled=resampled.sel(time=slice(c.time[0],c.time[1]))
                #add to dictionary
                values_l.append(resampled.values)
                sites_l.append(s.values.tolist())
                #compute error variance according to snr. use 200 year rolling window, take into account itme scale
                val_var=resampled.sel(time=slice('0850','1850')).rolling(time=int(200/ts_db)).var(skipna=True).mean(skipna=True).values
                if np.isnan(val_var):
                    val_var=resampled.sel(time=slice('0850','1850')).var(skipna=True).mean(skipna=True).values
                if c.noise_assum[ii]=='None':
                    r=val_var/(1+data['noise_snr'].values.item()**2)
                elif c.noise_assum[ii]=='equal':    
                    try: fac=c.equal_fac 
                    except: fac=1
                    r=HXfull_all_fin.sel(site=s).resample(time=(str(ts_db)+'YS')).mean('time').var('time').values.item()*fac #eventually reduce/increase proxy error
                else:
                    err=c.noise_assum[ii]
                    r=val_var/(1+err**2)

                    
                dictionary_r[str(ts_db)]['err'].append(r)
            dictionary[str(ts_db)]['ts'].append(np.array(values_l).T)
            #try: l=len(db.site) #except: l=1
            dictionary[str(ts_db)]['sites'].append(sites_l)
            dictionary[str(ts_db)]['num'].append(len(db.site))
    #if to reuse on higher timescales (won't do that probably, as its a bit dirty to use values multiple times)
        if c.reuse==True:
            idx=np.argwhere(np.array(c.timescales).astype(int)==ts_db).squeeze()
            sites_l=[]
            #noise
            for t_ii,t_i in enumerate(timescales[idx+1:]):
                idx=np.argwhere(np.array(c.timescales).astype(int)==int(t_i)).squeeze().item()
                target_time=times_list[idx]
                sites_l=[]
                values_l=[]
                sites=db.site
                #try: len(sites)
                #except: sites=np.array([sites.item()]) #hack for single site db (else it breakes)
                for iii,s in enumerate(tqdm.tqdm(sites)):
                    data=db.isel(site=iii)
                    #If res <4 don't use the lowpass filter.
                    if ts_db<4: filt=False
                    else: filt=True
                    resampled=make_equidistant_target(data,target_time,target_res=int(t_i),method_interpol=mode,filt=filt,min_ts=1)
                    #mask the gaps
                    resampled=mask_the_gap_alt(resampled,data, time_res=int(t_i),tol=mask_)
                    resampled=resampled.sel(time=slice(c.time[0],c.time[1]))
                    #add to dictionary
                    values_l.append(resampled.values)
                    sites_l.append(s.values.tolist())
                    #compute error variance according to snr
                    val_std=np.std(resampled.dropna('time').values)
                    if c.noise_assum[ii]=='None':
                        r=val_std/(1+data['noise_snr'].values.item()**2)
                    elif c.noise_assum[ii]=='equal':
                        try: fac=c.equal_fac 
                        except: fac=1
                        r=HXfull_all_fin.sel(site=s).resample(time=(str(ts_db)+'YS')).mean('time').var('time').values.item()*fac #eventually reduce/increase proxy error
                    else:
                        err=c.noise_assum[ii]
                        r=val_std/(1+err**2)
                    dictionary_r[str(t_i)]['err'].append(r)
                dictionary[str(t_i)]['ts'].append(np.array(values_l).T)
                dictionary[str(t_i)]['sites'].append(sites_l)
                #try: l=len(db.site)
                #except: l=1
                dictionary[str(t_i)]['num'].append(len(db.site))

    #bring together dictionaries into one for 
    #in dictionaries we have a list for stuff
    final_list=[]
    final_list_r=[]
    for ii,(i,dic) in enumerate(dictionary.items()):
        if len(dic['ts'])>0:
            vals=np.hstack(dic['ts']).squeeze() ##DOES THIS GO RIGHT FOR MORE THAN ONE DB?
            errors=np.hstack(dictionary_r[i]['err'])#
            final_list_r.append(errors)

            sites=np.concatenate(dic['sites'])
            target_time=times_list[ii]
            target_time=target_time.sel(time=slice(c.time[0],c.time[1]))
            #eventually transpose vals due to the way the resampled values are stacked (hacky but works)

            if len(vals.shape)==1: #hack for reconstruction of only one record
                vals=np.array([vals]).T
            data_array=xr.DataArray(vals,coords=dict(time=target_time,site=sites))
            data_array.attrs['DB_members']=dic['num']
            final_list.append(data_array.transpose('time','site'))
        #hack for timescales without any value: create a dummy record that only has nans (will not be used, but still essential for algorithm to run through.
        #will be ignored properly in the kalman filter loop
        else:
            target_time=times_list[ii]
            target_time.sel(time=slice(c.time[0],c.time[1]))
            da = xr.DataArray(np.nan, coords=dict(time=target_time, site=['-1']), dims=("time", "site"))
            da.attrs['DB_members']=[-1]
            final_list.append(da)
            final_list_r.append(np.nan)
    
    return final_list,final_list_r


def load_priors_psm(c,variab_list,variab_season_dict,proxy_db_all):
    """
    Load model (prior) data and apply proxy system model to the model data.
    
    c: config dictionray converted to namespace
    variab_list,variab_season_dict,proxy_db_all: Metadata required for the PSMs, previously extracted from proxy record nc files.
    (see variable_prep() and load_proxies()
    """
    bs=int(c.timescales[-1]) #blocksize: largest time scale
    #time resolutions without bs, reversed (backwards)
    if c.multi_model_prior is None:
        #set range to one such that the loop is just ran once (same as not multi-model prior)
        ran=1
    else:
        print('Will compute a multi-model-prior!')
        #try:
        dicts_paths=c.multi_model_prior
        ran=len(dicts_paths)
        try:
            model_names=list(c.multi_model_prior[0].keys())
        except:
            model_names=list(c.multi_model_prior.keys())
    #empty list where I am going to save the values vectors!
    values_vector_list=[]
    MC_idx_list=[]
    for i in range(ran):
        if c.multi_model_prior is not None:
            current_mod=model_names[i]
            cfg_copy=copy.deepcopy(cfg)
            cfg_copy['vp']=dicts_paths[current_mod]
            try:
                cfg_copy['oro']=oros[current_mod]
            except:
                #oros probably not deeded
                pass
            #workaround to make sure the broken iHadCM3 years are replaced at other place
            #if current_mod!='iHadCM3':
            cfg_copy['check_nan']=False
            """
            elif current_mod=='iHadCM3':
                cfg_copy['check_nan']=True
            """
            c2=SimpleNamespace(**cfg_copy)    
        else:
            c2=c 
        #LOAD THE PRIORS
        prior,attributes,seasonal_dict=prior_preparation(c2,variabs=variab_list,dictionary=variab_season_dict)
        #PSM APPLY FUNCTION
        HXfull_all=[] #List where we append the model proxy estimates to for each database (will be brought together later)
        print('Apply Proxy System Models')
        for i,psm in enumerate(c.psm):
            db=proxy_db_all[i].transpose('time',...) #make sure time is in front, so slice by second index for sites
            lats=db.lat.values
            lons=db.lon.values
            sites=db.site
            #NEW:
            if psm=='individual':
                #loop over the proxy records-> bring them into the right form (One joint dataarray with values)
                hx=[]
                for ii,s in enumerate(sites):
                    variab=db[:,ii]['variable_name'].item()
                    seasonality=db[:,ii]['seasonality'].item()
                    if seasonality=='None' or seasonality=='Annual' : seasonality=str(c.avg_proxies)
                    local_prior=seasonal_dict[variab][seasonality].sel(lat=lats[ii],lon=lons[ii],method='nearest') #has been debiased properly
                    #local_prior['site']=str(i)+'-'+str(ii) #add this preindex to be able to separate the different databases
                    local_prior['site']=db[:,ii]['site'].values.item()
                    local_prior=local_prior.expand_dims('site').transpose('time',...)
                    psm2=db[:,ii]['PSM']
                    if psm2=='linear':
                        hx.append(local_prior*db[:,ii]['regression_params_a']+db[:,ii]['regression_params_b'])
                    elif psm2=='inf_weighted' or 'prec_weighted':
                        hx.append(local_prior)
                hx=xr.concat(hx,dim='site',coords='minimal',compat='override')
                hx['lat']=('site',lats)
                hx['lon']=('site',lons)
                #concat later, there might be time problems
            elif psm=='prec_weighted':
                #do not take d18O_weighted, as this might be seasonally averaged, but here we take the annual average
                hx=seasonal_dict['d18O'][str(c.avg_proxies)].sel(lat=xr.DataArray(lats,dims='site'),lon=xr.DataArray(lons,dims='site'),method='nearest')
                hx['site']=db['site'].values
            HXfull_all.append(hx)

        ##LOOP OVER HXfull_all and create new dataset
        values=[];lat=[];lon=[];site=[];db_members=[]
        for h in HXfull_all: values.append(h.values);lat.append(h.lat.values);lon.append(h.lon.values);site.append(h.site.values); db_members.append(len(h.site))
        HXfull_all_fin=xr.DataArray(np.hstack(values),dims=('time','site'),coords={'time':prior['time'].values,'site':np.concatenate(site),'lat':('site',np.hstack(lat)),'lon':('site',np.hstack(lon))})
        HXfull_all_fin.attrs['DB_members']=db_members
        prior_stacked=prior.stack(site=('lat','lon')).transpose('time','site')
        #save coordinates for later when remaking the prior xarray
        coordinates=prior_stacked['site']
        lengths_init=np.repeat(len(coordinates),len(c.vp.keys()))
        names_init=np.repeat(coordinates.values,len(c.vp.keys()))
        names_short_init=list(c.vp.keys())
        #append observation estimates from prior    
        extra_list=[]
        names=[] #saving latitudes, site names, ..
        lengths=[]
        names_short=[] #to keep track of what the values in length stand for
        for o,i in enumerate(HXfull_all_fin.attrs['DB_members']):
            lengths.append(i)
            string_i='DB_'+str(o)
            names_short.append(string_i)

        values=[]
        for v in prior_stacked:
            values.append(prior_stacked[v].values)
        values=np.concatenate(values,axis=-1)
        values_vector=np.concatenate([values,HXfull_all_fin.transpose('time',...).values],axis=-1)

        names_vector=np.concatenate([names_init,HXfull_all_fin.site.values.tolist()],axis=-1)
        length_vector=np.concatenate([lengths_init,lengths])
        names_short_vector=np.concatenate([names_short_init,names_short])
        split_vector=np.cumsum(length_vector,dtype=int)[:-1] #can be applied to values_vector with numpy split
        values_vector_list.append(values_vector)
        #compute separate monte carlo indices for each model (brings in more randomnes)
        MC_idx=random_indices(c.nens,prior.time.shape[0]-bs-1,c.reps,seed=c.seed + i)
        MC_idx_list.append(MC_idx)

    MC_idx_list=np.array(MC_idx_list)
    
    return split_vector,names_vector,values_vector,names_short_vector,values_vector_list,MC_idx_list,HXfull_all_fin,coordinates

def globalmean(field,name=''):
    """
    Function that calculates the global mean of a climate field 
    (DataArray from one variable, not all variables)
    Does latitude weighting.

    Input:
        - Climate Field (time,lat,lon): GMT computed over lat,lon
        - variable name as string, (needed for naming)
    Output:
        - Global mean as DataArray with name "globalmean_+<name>"
        
    Other possible indices
    - El -Nino index: Sea Surface Temperature(5°N - 5°S), 120° - 170° West (190-240° in 0-360) mode
    (https://en.wikipedia.org/wiki/Multivariate_ENSO_index) (I currently don't have SST)
    - AMOC index: usually defined as the stream function for the zonally integrated meridional volume transport in the ocean
    (at some specific latitude). Steiger 2016: (defined here as the maximum value of the overturn-
    ing stream function in the North Atlantic between 25 and
    70◦ N and between depths of 500 and 2000 m
    - Convert -180,180 to 0,360 -> lon % 360
    """ 
    lat=field.lat
    wgt=np.cos(np.deg2rad(lat))
    field_m=field.weighted(wgt).mean(('lat','lon'))
    try:
        field_m=field_m.rename(('gm_'+name))
    except:
        pass
    return field_m

