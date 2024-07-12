"""
Wrapper for multi-time scale Paleoclimate Data Assimilation code.
All file paths and reconstruction configurations are read in via a dictionary (See paper_experiments Notebook for examples)
Builds on functions in:
- utils_new.py 
- kalman_filters.py

Loosely inspire by the code for the Last Millennium Reanalysis (https://github.com/modons/LMR).
Previous version of the code: https://github.com/mchoblet/paleoda (also allowed for pseudoproxy experiments)

Feel free to contact me if you have questions regarding the code and the structure.

MIT License
Copyright 10.12.2023 Mathurin Choblet

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import xarray as xr
import numpy as np
import cftime
import os
from types import SimpleNamespace

import utils_new
import kalman_filters
import warnings

import tqdm
import copy
import sys
import json

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore",category=DeprecationWarning) 
warnings.simplefilter("ignore",category=FutureWarning) 

def wrapper_paper(cfg):
    c=SimpleNamespace(**cfg)
    #load proxiy databases
    proxy_db_all=utils_new.load_proxies(c)
    #get the list of variables and seasons required for PSMs
    variab_list,variab_season_dict=utils_new.variable_prep(proxy_db_all)
    #load priors, apply psms. multi-model loop is an artefact from times where i used the multi-model ensemble directly. now i perform reconstructions separately and compute the mean afterwards 
    split_vector,names_vector,values_vector,names_short_vector,values_vector_list,MC_idx_list,HXfull_all_fin,coordinates=utils_new.load_priors_psm(c,variab_list,variab_season_dict,proxy_db_all)
    #resample proxies (do this after loading priors as we eventually define the proxy error to be equal to the model variance)
    final_list,final_list_r=utils_new.resample_wrap(c,proxy_db_all,HXfull_all_fin)

    #PREPARATION FOR PALEO DA LOOP
    #Create list of availables sites at each timescale. This is necessary for getting the right proxy estimates during the Multi-time-scale DA
    sites_avail=[]
    for l in final_list:
        sites_avail.append(l.site.values)
    proxy_start=int(split_vector[len(c.vp)-1])
    proxy_end=None
    proxy_names=names_vector[proxy_start:proxy_end]
    
    #HOTFIX: check for doubles/... in proxy names, double names cause problems later.
    doubles,proxy_names_count=np.unique(proxy_names,return_counts=True)
    doubles=doubles[proxy_names_count>1]
    proxy_names_count=proxy_names_count[proxy_names_count>1]
    for d in doubles:
        counter_i=0
        counter_j=0
        for i,nam in enumerate(proxy_names):
            if d==nam:
                proxy_names[i]=proxy_names[i]+'_'+str(counter_i)
                counter_i+=1
        for jj,db_nam in enumerate(sites_avail):
            for j,nam in enumerate(db_nam):
                if d==nam:
                    db_nam[j]=db_nam[j]+'_'+str(counter_j)
                    counter_j+=1
    
    ##PROXY FRAC:
    #the idea is too only use a fraction of proxy measurements (e.g. 75%) and repeat the reconstruction cfg['reps']-times. We thus create a list of proxy indices to use in each reconstruction
    #the concept is extended to multi-timescale DA applying to computing this list for every timescale available
    #the current implementation doesn't exclude that a proxy is not used in a smaller timescale, but on a higher one (due to the reuse option)
    #this would be a bit more complicated to implement
    if c.proxy_frac is not None:
        empty=[]
        for i_lis, lis in enumerate(final_list):
            #proxy frac can either be a fraction, or an absolute number (the latter is especially relevant for PPEs when comparing different timescales)
            l=len(lis.site)
            if c.proxy_frac<1:
                prox_mems=int(c.proxy_frac*l) #e.g 0.75*163=122
            else:
                prox_mems=int(c.proxy_frac) # absolute number
                if prox_mems>l:
                    print('Not enough proxies for timescale ', i_lis)
                    print('Setting number of available proxies to maximum: ', l)
                    prox_mems=l
            prox_idx=utils_new.random_indices(prox_mems,l,c.reps,seed=c.seed)
            empty.append(prox_idx)
        #proxy frac is the list
        proxy_frac=empty 
    
    #for each time scale save the indexes list
    idx_list=[[proxy_start+np.argwhere(proxy_names==al)[0][0] for al in b] if b[0]!='-1' else [] for b in sites_avail]
    
    idx_bs=idx_list[-1]
    #indices of proxy records available for largest timescale (for covariance loc)
    #idx_bs_2=idx_list_2[-1]

    #backwards proxy and proxy error list
    lisst_bw=final_list[:-1][::-1]
    lisst_r_bw=final_list_r[:-1][::-1]
    # number of assimilated values, years and blocks to reconstruct, repetitions and ensemble members
    num_vals=values_vector.shape[1]
    bs=int(c.timescales[-1])
    reps=c.reps

    #resolutions in time backwards
    time_res_r=[int(t) for t in c.timescales][:-1][::-1]

    length_recon=int(c.time[1])-int(c.time[0])
    #times list used for resampling proxy records
    times_list_recon=[xr.DataArray(xr.cftime_range(start=c.time[0],periods=(length_recon//int(i)+1),freq=str(i)+'YS',calendar='gregorian'),dims='time') for i in np.sort(np.array(c.timescales).astype(int))]
    num_times=len(times_list_recon[0])*c.smallest_timescale #to assure that basic time scale can also not be annual
    num_blocks=int(np.ceil(num_times/bs))

    if c.multi_model_prior is None: nens=c.nens
    else: nens=c.nens*len(c.multi_model_prior)

    #Initialize mean and std array for saving reconstruction
    mean_array=np.empty((reps,num_times,num_vals))
    std_array=np.empty((reps,num_times,num_vals))
    print('Start Multitimescale DA loop.')

    #THE MULTI TIME SCALE LOOP 
    #-> I eliminated all references to pseudoproxies, covariance localization and rank histograms
    #Monte carlo repetitions
    for r in tqdm.tqdm(range(reps)):
        #create prior_block form values vector, in the loops we create a similar variable called prior_block (be careful to take deepcopy if necessary)
        #values vector list is always a list (to allow for multi-model prior)
        prior_b=utils_new.prior_block_mme(values_vector_list,bs,MC_idx_list[:,r]) #first axis is the block size, then the ensemble members, last all values
        if c.proxy_frac is not None:
            proxy_frac_idx = [p[r] for p in proxy_frac]
            #reversed proxy_frac_idx without last list
            proxy_frac_idx_r=list(reversed(proxy_frac_idx[:-1]))
        shape=prior_b.shape
        prior_flat=prior_b.reshape(bs,-1)
        m_bs,a_bs=utils_new.anomean_with_numba(prior_flat,bs)
        m_bs=m_bs.reshape((1,shape[1],shape[2]))
        a_bs=a_bs.reshape(shape)
        #Available proxy estimates for largest time_scale
        HXf_bs_m=m_bs[:,:,idx_bs]
        #loop over blocks
        for i in range(num_blocks):
            #assimilate block size means directly (saves one anomean calculation)
            current_time=times_list_recon[-1][i]
            Y=final_list[-1].isel(time=i).values
            R=final_list_r[-1] # used to be set time varying.isel(time=i)
            #eventually only select some proxies
            if c.proxy_frac is not None:
                Y=Y[proxy_frac_idx[-1]]
                R=R[proxy_frac_idx[-1]]
            #indices where Y is not nan
            mask=~np.isnan(Y)
            Y=Y[mask]
            R=R[mask]
            #Get prior forecast (Ne x Nx)
            Xf=m_bs[0].copy() 
            if len(Y)>0:
                #Additionaly mask the the prior estimates as given by availability
                HXf=copy.deepcopy(HXf_bs_m[0])
                if c.proxy_frac is not None:
                    HXf=HXf[:,proxy_frac_idx[-1]]
                HXf=HXf[:,mask]
                #compute Kalman Filter for Block!
                Xf_post=kalman_filters.ETKF(Xf.T,HXf.T,Y,R)
                if np.any(np.isnan(Xf_post)):
                    raise ValueError('Found nan in posterior')
                    # a reason can be a wrong time shape (length must be divisible by block size, else stuff can go wrong)
                
                prior_block=Xf_post.T + a_bs
            else:
                prior_block=copy.deepcopy(prior_b)
            #loop over all other resolutions (backwards)
            for ii,res in enumerate(time_res_r):
                #proxy indices for that time_res
                idx_res=idx_list[:-1][::-1][ii]  
                if res!=1:
                    prior_flat=prior_block.reshape(bs,-1)
                    mean_res,anom_res=utils_new.anomean_with_numba(prior_flat,res) #prior_flat not changed
                    mean_res=mean_res.reshape((bs//res,nens,num_vals))
                    anom_res=anom_res.reshape(shape)
                else:
                    anom_res=np.zeros((shape))
                    mean_res=prior_block
                #loop over sub_index in block, computed via true divison (e.g. 50/25 = 2)
                bs_mod_res=bs//res
                
                for sub_index in range(bs_mod_res):
                    #special treatment for the last block
                    if sub_index * res + i*bs < num_times:
                        #get the current proxies at the right time
                        Y=lisst_bw[ii][i*bs_mod_res+sub_index].values
                        #eventually slice
                        if c.proxy_frac is not None:
                            Y=Y[proxy_frac_idx_r[ii]]
                        #which proxies are available?
                        mask=~np.isnan(Y)
                        Y=Y[mask]
                        if len(Y)>0:
                            R=lisst_r_bw[ii]#[i*bs_mod_res+sub_index]
                            #eventually slice
                            if c.proxy_frac is not None:
                                R=R[proxy_frac_idx_r[ii]]
                            R=R[mask]
                            Xf=mean_res[sub_index,:]
                            #get averaged proxy estimates + available proxies
                            HXf=mean_res[sub_index,:][:,idx_res]
                            #apply proxy fraction
                            if c.proxy_frac is not None:
                                HXf=HXf[:,proxy_frac_idx_r[ii]]
                            #slice according to mask
                            HXf=HXf[:,mask]  # Ne x Ny 
                            Xf_post=kalman_filters.ETKF(Xf.T,HXf.T,Y,R)
                        else:
                            Xf_post=mean_res[sub_index,:].T
                        
                        #sanity check
                        if np.any(np.isnan(Xf_post)):
                            raise ValueError('Found nan in posterior')
                            # a reason can be a wrong time shape (length must be divisible by block size, else stuff can go wrong)
                        
                        start=sub_index*res
                        end=(sub_index+1)*res
                        prior_block[start:end]=Xf_post.T + anom_res[start:end]
                    else:
                        pass
            #compute mean values in block (along ensemble)
            mean_block=np.mean(prior_block,axis=1)
            std_block=np.std(prior_block,axis=1)
            #SPECIAL TREATMENT FOR THE END IF TIMESCALES AND RECONSTRUCTION TIME DON'T PERFECTLY MATCH
            #fill it with the prior then
            #fill mean_array at that part for that repetition
            block_start=bs*i
            block_end=bs*(i+1)
            if block_end>num_times:
                mean_array[r,block_start:block_end,:]=mean_block[:bs-(block_end-num_times),:]
                std_array[r,block_start:block_end,:]=std_block[:bs-(block_end-num_times),:]
            else:
                mean_array[r,block_start:block_end,:]=mean_block
                std_array[r,block_start:block_end,:]=std_block

    #take mean along Monte Carlo
    mean_array_final=mean_array.mean(axis=0)
    std_array_final=std_array.mean(axis=0)      
    print('Finished multitimescale DA')
    
       

    #SAVING: SPLITTING UP THE VECTOR
    #Now we have to resplit everything, and eventually also calculate PPE evaluation metrics
    splitted_mean=np.split(mean_array_final,split_vector,axis=-1)
    splitted_std=np.split(std_array_final,split_vector,axis=-1)
    num_vars=len(c.vp)

    #SAVING ALL VARIABLES
    ds_list=[]
    print('Save variables')
    ###NEW: RESAMPLE THE TIME-SERIES TO THE MINIMAL TIMESCALE 
    splitted_mean=[np.mean(a.reshape(-1,c.smallest_timescale, a.shape[1]),axis=1)  for a in splitted_mean]
    splitted_std=[np.mean(a.reshape(-1,c.smallest_timescale, a.shape[1]),axis=1)  for a in splitted_std]

    #save the datavariables
    for j in range(num_vars):
        name=names_short_vector[j]
        save_mean=xr.DataArray(splitted_mean[j],coords=dict(time=times_list_recon[0],site=coordinates)).unstack('site')
        save_std=xr.DataArray(splitted_std[j],coords=dict(time=times_list_recon[0],site=coordinates)).unstack('site')
        string_m=name+'_mean'
        std_std=name+'_std'
        ds=xr.Dataset(data_vars={string_m: save_mean,std_std:save_std})
        ds_list.append(ds)

    ds=xr.merge(ds_list)        
    #ds=ds.assign_coords(MC_idx=(('reps','nens'),MC_idx))    
    ds=ds.assign_coords(MC_idx=(('model','reps','nens'),MC_idx_list))    

    #Proxy estimates are always saved.
    ds['site']=HXfull_all_fin.site.values
    proxies=np.concatenate(splitted_mean[num_vars:(num_vars+len(c.obsdata))],axis=-1)
    proxies_std=np.std(np.concatenate(splitted_mean[num_vars:(num_vars+len(c.obsdata))],axis=-1),axis=0)
    ds['HXf_m']=(('time','site'),proxies)
    ds['HXf_std']=(('site'),proxies_std)

    #add the proxies at the different resolutions to the Dataset
    for i,l in enumerate(final_list):
        num=c.timescales[i]
        string_a='proxies_res_'+str(num)
        time_dim=('time_res_'+str(num))
        site_dim=('site_'+str(num))
        proxies_at_res=xr.DataArray(l.values,coords={time_dim:l.time.values,site_dim:l.site.values })        
        ds[string_a]=proxies_at_res
      
    if c.proxy_frac_check:
            
        #evaluate correlation for withheld proxy records
        if c.proxy_frac is not None and c.proxy_frac!=False:
   
            ds['corr_withheld_proxies']=('site',np.zeros(len(ds.site))); ds['corr_used_proxies']=('site',np.zeros(len(ds.site)))
            ds['withheld']=(('reps','time','site'),np.zeros((len(ds.reps),len(ds.time),len(ds.site))))
            
            #evalute correlation of withheld proxy records (~15 seconds for 50 reps). Compute correlation at each repetion
            #also added correlation for non withheld proxy records
            time=times_list_recon[0]
            da=xr.DataArray(mean_array,dims=('reps','time','site'),coords={'time':time})
            ds['proxy_est']=da[:,:,proxy_start:proxy_end] #reps,time,site

            #two loops for  withhold/used proxy records
            for m in [False,True]:
                #create empty dictionary to save correlations
                dic={i:{j:[] for j in idx_list[ii]} for ii,i in enumerate(c.timescales)}
                
                #extra dictionary for saving the reconstructed time series (np.nan when not used)
                dic_vals={i:{j:[] for j in idx_list[ii]} for ii,i in enumerate(c.timescales)}
                
                for rep in range(reps):
                    #loop over time scales
                    for i,idx in enumerate(idx_list):
                        prox=proxy_frac[i][rep] #get proxies that were used
                        mask=np.ones(len(idx),dtype=bool) 
                        mask[prox] = m #if m==False, masks sets these to false.
                        rest=np.array(idx)[mask] #proxies that were (not) used. for m==True same as idx.
                        true=final_list[i][:,mask] #get true time series for excluded proxies
                        ts=int(c.timescales[i])
                        if m==False: #set when proxy was used to nan to just keep when proxy was not kept
                            ds['proxy_est'][rep,:,np.array(idx)[~mask]-proxy_start]=np.nan
                        
                        #loop over proxies that have not been used/been used
                        for ii,r in enumerate(rest):
                            #extract true proxies
                            t=true.isel(site=ii)
                            hx=da[rep][:,r]     
                            if ts!=1: hx=hx.resample(time=(str(ts)+'YS')).mean()
                            mask=~np.isnan(t); hx=hx[mask];t=t[mask];corr=np.corrcoef(t,hx)[0,1]
                            dic[str(ts)][r].append(corr)        
                
                #compute mean for each proxy
                for ts in list(dic.keys()):
                    for k in dic[ts].keys():
                        if m==False: ds['corr_withheld_proxies'][int(k)-proxy_start]=np.mean(dic[ts][k]) 
                        else: ds['corr_used_proxies'][int(k)-proxy_start]=np.mean(dic[ts][k])

    #try adding configuration as dataset attribute
    try:
        ds.attrs['cfg']=str(cfg)
    except:pass
    #save Dataset
    try:
        ds.to_netcdf(c.savepath)
    except:
        print('Warning: Overriding', c.savepath)
        os.system(('rm'+c.savepath))
        ds.to_netcdf(c.savepath)
    return ds
