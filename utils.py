import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score


def read_10x_visium(dataset = 'DLPFC', section_id = '151676', ground_truth=False):
    input_dir = os.path.join('data', '10x Visium', dataset, section_id)
    ground_truth_url = os.path.join(input_dir, section_id+'_truth.txt')
    adata = sc.read_visium(path=input_dir, count_file=section_id + '_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    # load ground truth
    if(ground_truth):
        Ann_df = pd.read_csv(ground_truth_url, sep='\t', header=None, index_col=0)
        Ann_df.columns = ['ground_truth']
        adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']
    
    if(dataset == 'DLPFC'):
        if(ground_truth):
            layer_map = {'Layer_1':0, 'Layer_2':1, 'Layer_3':2, 'Layer_4':3, 'Layer_5':4, 'Layer_6':5, 'WM':6, '':-1}
            adata.obs['ground_truth'] = np.array(list(map(layer_map.get, adata.obs['ground_truth'])))

    # for item in adata.obs['ground_truth']:
    #     print(item)

    # load position information
    position_url = os.path.join(input_dir, 'spatial', 'tissue_positions_list.csv')
    spatial=pd.read_csv(position_url,sep=",",header=None,na_filter=False,index_col=0) 
    adata.obs['in_tissue']=spatial[1]
    adata.obs['arr_col']=spatial[2]
    adata.obs['arr_row']=spatial[3]
    adata.obs['image_col']=spatial[4]
    adata.obs['image_row']=spatial[5]

    # select captured samples
    # adata=adata[adata.obs['in_tissue']==1]

    # special for stLearn
    adata.uns["spatial"][section_id]["use_quality"] = 'hires'
    adata.obs['imagerow'] = adata.obs['image_row']
    adata.obs['imagecol'] = adata.obs['image_col']

    # add spatial
    spatial_coor = pd.DataFrame(np.array(adata.obs[['image_row', 'image_col']]))
    adata.obsm['spatial'] = np.array(spatial_coor)

    return adata


def read_seqFISH(dataset = '', section_id = '', ground_truth=False):
    import squidpy as sq
    adata = sq.datasets.seqfish()

    adata.obs['image_col'] = adata.obsm['spatial'][:,0]
    adata.obs['image_row'] = adata.obsm['spatial'][:,1]


    return adata

def read_slideSeqV2(dataset =  'mouse olfactory bulb', section_id = 'Puck_200127_15', ground_truth=False):
    input_dir = os.path.join('data', 'Slide-seqV2', dataset, section_id)
    counts_file = os.path.join(input_dir, section_id + '.digital_expression.txt')
    coor_file = os.path.join(input_dir, section_id + '_bead_locations.csv')

    counts = pd.read_csv(counts_file, sep = '\t', index_col = 0)
    coor_df = pd.read_csv(coor_file, index_col = 'barcode')
    # print(counts.shape, coor_df.shape)

    adata = sc.AnnData(counts.T)
    adata.var_names_make_unique
    coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
    adata.obsm['spatial'] = coor_df.to_numpy()

    adata.obs['image_col'] = adata.obsm['spatial'][:,0]
    adata.obs['image_row'] = adata.obsm['spatial'][:,1]

    # add spatial
    spatial_coor = pd.DataFrame(np.array(adata.obs[['image_row', 'image_col']]))
    adata.obsm['spatial'] = np.array(spatial_coor)


    # load ground truth
    if(ground_truth):
        ground_truth_url = os.path.join(input_dir, section_id+'_truth.txt')
        Ann_df = pd.read_csv(ground_truth_url, sep='\t', header=None, index_col=0).fillna('')
        Ann_df.columns = ['ground_truth']
        adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']

    sc.pp.calculate_qc_metrics(adata, inplace = True)

    barcode_file = os.path.join(input_dir, 'used_barcodes.txt')
    used_barcode = pd.read_csv(barcode_file, sep = '\t', header = None)
    used_barcode = used_barcode[0]

    adata = adata[used_barcode]

    sc.pp.filter_genes(adata, min_cells=50)

    return adata

def read_starMAP(dataset = 'mouse visual cortex', section_id = 'STARmap_20180505_BY3_1k', ground_truth=False):
    input_dir = os.path.join('data', 'starMAP', dataset, section_id)
    h5ad_url = os.path.join(input_dir, section_id+'.h5ad')
    adata = sc.read(h5ad_url)

    adata.obs['image_col'] = adata.obs['X']
    adata.obs['image_row'] = adata.obs['Y']

    # add spatial
    spatial_coor = pd.DataFrame(np.array(adata.obs[['image_row', 'image_col']]))
    adata.obsm['spatial'] = np.array(spatial_coor)


    return adata

def read_merFISH(dataset = 'human osteosarcoma', section_id = 'human osteosarcoma', ground_truth=False):
    input_dir =  os.path.join('data', 'merFISH', dataset, section_id)
    count_file = os.path.join(input_dir, 'count.csv')
    location_file = os.path.join(input_dir, 'location.xlsx')

    count_df = pd.read_csv(count_file, encoding='utf-8', index_col=0)

    import openpyxl
    wb = openpyxl.load_workbook(location_file)
    sheet_name_list = wb.get_sheet_names()

    coor_df_list = []

    for sheet_name in sheet_name_list:
        coor_df_item = pd.read_excel(location_file, index_col=0, sheet_name=sheet_name)
        coor_df_list.append(coor_df_item)

    coor_df = pd.concat(coor_df_list)

    # construct Anndata object

    adata = sc.AnnData(count_df.T, dtype=np.float32)
    adata.var_names_make_unique
    adata.obsm['spatial'] = coor_df.to_numpy()

    adata.obs['image_col'] = adata.obsm['spatial'][:,0]
    adata.obs['image_row'] = adata.obsm['spatial'][:,1]

    # load ground truth
    if(ground_truth):
        ground_truth_url = os.path.join(input_dir, section_id+'_truth.txt')
        Ann_df = pd.read_csv(ground_truth_url, sep='\t', header=None, index_col=0).fillna('')
        Ann_df.columns = ['ground_truth']
        adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']

    # seperate adata to adata list
    adata_list = []
    num_field = len(sheet_name_list)
    for field in range(1, num_field+1):
        contain_name = 'B' + str(field)
        adata_1 = adata[adata.obs_names.str.contains(contain_name)]
        adata_list.append(adata_1)

    return adata_list

def read_merFISH_after_clustering(method='stLearn', dataset = 'human osteosarcoma', section_id = 'human osteosarcoma', ground_truth=False):
    input_dir =  os.path.join('data', 'merFISH', dataset, section_id)
    count_file = os.path.join(input_dir, 'count.csv')
    location_file = os.path.join(input_dir, 'location.xlsx')

    count_df = pd.read_csv(count_file, encoding='utf-8', index_col=0)

    import openpyxl
    wb = openpyxl.load_workbook(location_file)
    sheet_name_list = wb.get_sheet_names()

    coor_df_list = []

    for sheet_name in sheet_name_list:
        coor_df_item = pd.read_excel(location_file, index_col=0, sheet_name=sheet_name)
        coor_df_list.append(coor_df_item)

    coor_df = pd.concat(coor_df_list)

    # construct Anndata object

    adata = sc.AnnData(count_df.T, dtype=np.float32)
    adata.var_names_make_unique
    adata.obsm['spatial'] = coor_df.to_numpy()

    adata.obs['image_col'] = adata.obsm['spatial'][:,0]
    adata.obs['image_row'] = adata.obsm['spatial'][:,1]

    # load ground truth
    if(ground_truth):
        ground_truth_url = os.path.join(input_dir, section_id+'_truth.txt')
        Ann_df = pd.read_csv(ground_truth_url, sep='\t', header=None, index_col=0).fillna('')
        Ann_df.columns = ['ground_truth']
        adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']

    # load clustering result
    save_dir = os.path.join('output', 'merFISH', dataset, section_id, method)
    clustering_url = os.path.join(save_dir, 'clustering_res.csv')
    clustering_df = pd.read_csv(clustering_url,  index_col=0)
    clustering_df.index = adata.obs.index
    adata.obs['clustering_res'] = clustering_df.astype('category')

    # seperate adata to adata list
    adata_list = []
    num_field = len(sheet_name_list)
    for field in range(1, num_field+1):
        contain_name = 'B' + str(field)
        adata_1 = adata[adata.obs_names.str.contains(contain_name)]
        adata_list.append(adata_1)

    return adata_list



def read_stereoSeq(dataset = 'mouse olfactory bulb', section_id = 'mouse olfactory bulb', ground_truth=False):
    input_dir = os.path.join('data', 'Stereo-seq', dataset, section_id)
    counts_file = os.path.join(input_dir, 'RNA_counts.tsv')
    coor_file = os.path.join(input_dir, 'position.tsv')

    counts = pd.read_csv(counts_file, sep='\t', index_col=0)
    coor_df = pd.read_csv(coor_file, sep='\t')
    # print(counts.shape, coor_df.shape)

    counts.columns = ['Spot_'+str(x) for x in counts.columns]
    coor_df.index = coor_df['label'].map(lambda x: 'Spot_'+str(x))
    coor_df = coor_df.loc[:, ['x','y']]

    # print(coor_df.head())

    adata = sc.AnnData(counts.T)
    adata.var_names_make_unique()

    # print(adata)

    coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
    adata.obsm["spatial"] = coor_df.to_numpy()

    adata.obs['image_col'] = adata.obsm['spatial'][:,0]
    adata.obs['image_row'] = adata.obsm['spatial'][:,1]

    # load ground truth
    if(ground_truth):
        ground_truth_url = os.path.join(input_dir, section_id+'_truth.txt')
        Ann_df = pd.read_csv(ground_truth_url, sep='\t', header=None, index_col=0).fillna('')
        Ann_df.columns = ['ground_truth']
        adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']

    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # plt.rcParams["figure.figsize"] = (5,4)
    # sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
    # plt.title("")
    # plt.axis('off')

    used_barcode = pd.read_csv(os.path.join(input_dir, 'used_barcodes.txt'), sep='\t', header=None)
    used_barcode = used_barcode[0]
    adata = adata[used_barcode,]

    # print(adata)

    # plt.rcParams["figure.figsize"] = (5,4)
    # sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
    # plt.title("")
    # plt.axis('off')

    sc.pp.filter_genes(adata, min_cells=50)
    # print('After flitering: ', adata.shape)

    return adata

# def read_seqFISH_plus(dataset = 'mouse OB', section_id = 'mouse OB', ground_truth=False):
    input_dir = os.path.join('data', 'seqFISH+', dataset, section_id)
    count_url = os.path.join(input_dir, dataset + '_counts.csv')
    location_url = os.path.join(input_dir, dataset + '_cellcentroids.csv')
    # anno_url = os.path.join(input_dir, dataset + '_cell_type_annotations.csv')

    count_df = pd.read_csv(count_url, encoding = 'utf-8')
    coor_df = pd.read_csv(location_url)[['X', 'Y']]
    # anno_df = pd.read_csv(anno_url, index_col=0)

    # print('---------- count_df ----------')
    # print(count_df)
    # print('---------- coor_df -----------')
    # print(coor_df)
    # print('---------- anno_df -----------')
    # print(anno_df)
    
    adata = sc.AnnData(count_df)
    adata.var_names_make_unique
    adata.obsm['spatial'] = coor_df.to_numpy()

    adata.obs['image_col'] = adata.obsm['spatial'][:,0]
    adata.obs['image_row'] = adata.obsm['spatial'][:,1]
    # adata.obs['ground_truth'] = anno_df

    if(ground_truth):
        anno_url = os.path.join(input_dir, dataset + '_cell_type_annotations.csv')
        anno_df = pd.read_csv(anno_url, index_col=0)
        adata.obs['ground_truth'] = anno_df

    return adata


def read_seqFISH_plus(dataset = 'mouse OB', section_id = 'mouse OB', ground_truth=False):
    input_dir = os.path.join('data', 'seqFISH+', dataset, section_id)
    count_url = os.path.join(input_dir, dataset + '_counts.csv')
    location_url = os.path.join(input_dir, dataset + '_cellcentroids.csv')
    # anno_url = os.path.join(input_dir, dataset + '_cell_type_annotations.csv')

    count_df = pd.read_csv(count_url, encoding = 'utf-8')
    cell_centroid_df = pd.read_csv(location_url)
    coor_df = cell_centroid_df[['X', 'Y']]
    # anno_df = pd.read_csv(anno_url, index_col=0)

    # print('---------- count_df ----------')
    # print(count_df)
    # print('---------- coor_df -----------')
    # print(coor_df)
    # print('---------- anno_df -----------')
    # print(anno_df)
    
    adata = sc.AnnData(count_df, dtype=np.float32)
    adata.var_names_make_unique
    adata.obsm['spatial'] = coor_df.to_numpy()

    adata.obs['image_col'] = adata.obsm['spatial'][:,0]
    adata.obs['image_row'] = adata.obsm['spatial'][:,1]
    adata.obs['field_of_view'] = cell_centroid_df[['Field of View']].to_numpy()
    # adata.obs['ground_truth'] = anno_df

    if ground_truth:
        anno_url = os.path.join(input_dir, dataset + '_cell_type_annotations.csv')
        anno_df = pd.read_csv(anno_url, index_col=0)
        anno_df.index = adata.obs.index
        adata.obs['ground_truth'] = anno_df  

    # seperate adata to adata list
    adata_list = []
    field_list = set(adata.obs['field_of_view'].values)
    for field in field_list:
        adata_1 = adata[adata.obs['field_of_view'] == field]
        adata_list.append(adata_1)
    
    return adata_list


def read_seqFISH_plus_after_clustering(method='stLearn', dataset = 'mouse OB', section_id = 'mouse OB', ground_truth=False):
    input_dir = os.path.join('data', 'seqFISH+', dataset, section_id)
    count_url = os.path.join(input_dir, dataset + '_counts.csv')
    location_url = os.path.join(input_dir, dataset + '_cellcentroids.csv')
    # anno_url = os.path.join(input_dir, dataset + '_cell_type_annotations.csv')

    count_df = pd.read_csv(count_url, encoding = 'utf-8')
    adata = sc.AnnData(count_df, dtype=np.float32)
    adata.var_names_make_unique

    cell_centroid_df = pd.read_csv(location_url)
    cell_centroid_df.index = adata.obs.index
    coor_df = cell_centroid_df[['X', 'Y']]
    
    adata.obsm['spatial'] = coor_df.to_numpy()

    adata.obs['image_col'] = adata.obsm['spatial'][:,0]
    adata.obs['image_row'] = adata.obsm['spatial'][:,1]
    adata.obs['field_of_view'] = cell_centroid_df['Field of View']


    if ground_truth:
        anno_url = os.path.join(input_dir, dataset + '_cell_type_annotations.csv')
        anno_df = pd.read_csv(anno_url, index_col=0)
        anno_df.index = adata.obs.index
        adata.obs['ground_truth'] = anno_df  

    # load clustering result
    save_dir = os.path.join('output', 'seqFISH+', dataset, section_id, method)
    clustering_url = os.path.join(save_dir, 'clustering_res.csv')
    clustering_df = pd.read_csv(clustering_url, index_col=0)
    clustering_df.index = adata.obs.index
    adata.obs['clustering_res'] = clustering_df.astype('category')

    # seperate adata to adata list
    adata_list = []
    field_list = set(adata.obs['field_of_view'].values)
    for field in field_list:
        adata_1 = adata[adata.obs['field_of_view'] == field]
        adata_list.append(adata_1)
    
    return adata_list


def get_section_list(tech='10x Visium', dataset='DLPFC', section_id=''):
    section_list = {}
    if section_id != '':
        section_list = [section_id]
    else:
        data_dir = os.path.join('data', tech, dataset)
        section_list = os.listdir(data_dir)
        section_list.sort()
    return section_list

# get n_row and n_col of multiple plots
# return: single_n_row, single_n_col, double_n_row, double_n_col
def get_nrow_and_ncol(section_list, ground_truth=False):
    single_n_section = len(section_list)
    if ground_truth:
        double_n_section = 2*single_n_section
    else:
        double_n_section = single_n_section
    
    # get n_row and n_col according to n_section
    if single_n_section < 2:
        single_n_col = 1
    elif single_n_section < 5:
        single_n_col = 2
    elif single_n_section < 17:
        single_n_col = 4
    else:
        single_n_col = 6

    single_n_row = single_n_section / single_n_col
    if(single_n_section % single_n_col != 0):
        single_n_row = single_n_row + 1

    if double_n_section < 2:
        double_n_col = 1
    elif double_n_section < 5:
        double_n_col = 2
    elif double_n_section < 17:
        double_n_col = 4
    else:
        double_n_col = 6

    double_n_row = double_n_section / double_n_col
    if(double_n_section % double_n_col != 0):
        double_n_row = double_n_row + 1

    return single_n_row, single_n_col, double_n_row, double_n_col

def read_anndata(tech='10x Visium', dataset='DLPFC', section_id='', ground_truth=False, after_cluster=False, method=''):
    if tech == '10x Visium':
        adata = read_10x_visium(dataset = dataset, section_id = section_id, ground_truth=ground_truth)
    elif tech == 'seqFISH':
        adata = read_seqFISH(dataset=dataset, section_id=section_id, ground_truth=ground_truth)
    elif tech == 'Slide-seqV2':
        adata = read_slideSeqV2(dataset=dataset, section_id=section_id, ground_truth=ground_truth)
    elif tech == 'starMAP':
        adata = read_starMAP(dataset=dataset, section_id=section_id, ground_truth=ground_truth)
    elif tech == 'merFISH':
        if after_cluster:
            adata_list = read_merFISH_after_clustering(method=method, dataset=dataset, section_id=section_id, ground_truth=ground_truth)
        else:
            adata = read_merFISH(dataset=dataset, section_id=section_id, ground_truth=ground_truth)
    elif tech == 'Stereo-seq':
        adata = read_stereoSeq(dataset=dataset, section_id=section_id, ground_truth=ground_truth)
    elif tech == 'seqFISH+':
        if after_cluster:
            adata_list = read_seqFISH_plus_after_clustering(method=method, dataset=dataset, section_id=section_id, ground_truth=ground_truth)
        else:
            adata = read_seqFISH_plus(dataset=dataset, section_id=section_id, ground_truth=ground_truth)

    # two return result
    if after_cluster:
        return adata_list
    else:
        return adata

def save_image_clustering(method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='', ground_truth=False, spot_size=-1):
    if section_id == '':
        save_dir = os.path.join('output', tech, dataset, 'comprehensive')
    else:
        save_dir = os.path.join('output', tech, dataset, section_id, 'comprehensive')


    if(not os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    parameters = {
        'figure.figsize' : (5,4),
        'axes.titlesize' : 4,
    }

    mpl.rcParams.update(parameters)

    section_list = get_section_list(tech=tech, dataset=dataset, section_id=section_id)
    single_n_row, single_n_col, double_n_row, double_n_col = get_nrow_and_ncol(section_list=section_list, ground_truth=ground_truth)

    fig, clustering_axes = plt.subplots(int(double_n_row), int(double_n_col))

    if(double_n_col == 1):
        clustering_axes = np.array([clustering_axes])

    clustering_axes = clustering_axes.ravel()
    

    index = 0
    for section_id in section_list:
        result_dir = os.path.join('output', tech, dataset, section_id, method)

        # read anndata and add clustering result
        adata = read_anndata(tech=tech, dataset=dataset, section_id=section_id, ground_truth=ground_truth)
        adata.var_names_make_unique()
        
        clustering_url = os.path.join(result_dir, 'clustering_res.csv')
        clustering_res = pd.read_csv(clustering_url, index_col=0)
        clustering_res.index = adata.obs.index

        adata.obs['clustering_res'] = clustering_res.astype('category')

        if(ground_truth):
            obs_df = adata.obs.dropna()
            ARI = adjusted_rand_score(obs_df['clustering_res'], obs_df['ground_truth'])
            ARI = round(ARI, 2)
            adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')

        adata.obs['clustering_res'] = clustering_res.astype('category')

        if spot_size == -1:
            n_cells = adata.obs.shape[0]
            spot_size = 72 * 6 / n_cells / double_n_col

        if(ground_truth):
            if adata.uns.get('spatial') is not None:
                sc.pl.spatial(adata, img_key='hires', color = 'ground_truth', legend_fontsize='xx-small',\
                title=method, ax = clustering_axes[2*index], show = False, \
                    )

                sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                title=section_id + ':%.2f'%ARI, ax = clustering_axes[2*index+1], show = False, \
                    )
            else:
                sc.pl.spatial(adata, img_key='hires', color = 'ground_truth', \
                title=method, ax = clustering_axes[2*index], show = False,
                spot_size=spot_size
                    )

                sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                title=section_id + ':%.2f'%ARI, ax = clustering_axes[2*index+1], show = False, 
                spot_size=spot_size
                    )
        else:
            if adata.uns.get('spatial') is not None:
                sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                title=section_id, ax = clustering_axes[index], show = False,
                    )
            else:
                sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                title=section_id, ax = clustering_axes[index], show = False,
                spot_size=spot_size\
                    )

        plt.axis('off')

        index = index + 1

    # clear axes
    
    for i in range(clustering_axes.shape[0]):
        clustering_axes[i].set_xlabel('')
        clustering_axes[i].set_ylabel('')
        clustering_axes[i].legend().remove()
        clustering_axes[i].axis('off')
    clustering_filename = method + ' clustering.png'
    clustering_file_url = os.path.join(save_dir, clustering_filename)
    
    plt.savefig(clustering_file_url, dpi = 240, bbox_inches='tight', pad_inches=0)


# for multiple clustering image saving
def save_image_clustering_all(method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='', ground_truth=False, spot_size=-1, clustering_method='kemans', n_epoch=1):
    # print(n_epoch)
    if section_id == '':
        save_dir = os.path.join('output', tech, dataset, 'comprehensive')
    else:
        save_dir = os.path.join('output', tech, dataset, section_id, 'comprehensive')


    if(not os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    parameters = {
        'figure.figsize' : (5,4),
        'axes.titlesize' : 4,
    }

    mpl.rcParams.update(parameters)

    section_list = get_section_list(tech=tech, dataset=dataset, section_id=section_id)
    single_n_row, single_n_col, double_n_row, double_n_col = get_nrow_and_ncol(section_list=section_list, ground_truth=ground_truth)

    fig, clustering_axes = plt.subplots(int(double_n_row), int(double_n_col))

    if(double_n_col == 1):
        clustering_axes = np.array([clustering_axes])

    clustering_axes = clustering_axes.ravel()
    

    index = 0
    for section_id in section_list:
        result_dir = os.path.join('output', tech, dataset, section_id, method)

        # read anndata and add clustering result
        adata = read_anndata(tech=tech, dataset=dataset, section_id=section_id, ground_truth=ground_truth)
        adata.var_names_make_unique()
        
        clustering_url = os.path.join(result_dir, 'clustering_res.csv')
        clustering_res = pd.read_csv(clustering_url, index_col=0)
        clustering_res.index = adata.obs.index

        adata.obs['clustering_res'] = clustering_res.astype('category')

        if(ground_truth):
            obs_df = adata.obs.dropna()
            ARI = adjusted_rand_score(obs_df['clustering_res'], obs_df['ground_truth'])
            ARI = round(ARI, 2)
            adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')

        adata.obs['clustering_res'] = clustering_res.astype('category')

        if spot_size == -1:
            n_cells = adata.obs.shape[0]
            spot_size = 72 * 6 / n_cells / double_n_col

        if(ground_truth):
            if adata.uns.get('spatial') is not None:
                sc.pl.spatial(adata, img_key='hires', color = 'ground_truth', legend_fontsize='xx-small',\
                title=method, ax = clustering_axes[2*index], show = False, \
                    )

                sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                title=section_id + ':%.2f'%ARI, ax = clustering_axes[2*index+1], show = False, \
                    )
            else:
                sc.pl.spatial(adata, img_key='hires', color = 'ground_truth', \
                title=method, ax = clustering_axes[2*index], show = False,
                spot_size=spot_size
                    )

                sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                title=section_id + ':%.2f'%ARI, ax = clustering_axes[2*index+1], show = False, 
                spot_size=spot_size
                    )
        else:
            if adata.uns.get('spatial') is not None:
                sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                title=section_id, ax = clustering_axes[index], show = False,
                    )
            else:
                sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                title=section_id, ax = clustering_axes[index], show = False,
                spot_size=spot_size\
                    )

        plt.axis('off')

        index = index + 1

    # clear axes
    
    for i in range(clustering_axes.shape[0]):
        clustering_axes[i].set_xlabel('')
        clustering_axes[i].set_ylabel('')
        clustering_axes[i].legend().remove()
        clustering_axes[i].axis('off')
    clustering_filename = method + ' clustering ' + str(n_epoch) +'.png'
    clustering_file_url = os.path.join(save_dir, clustering_filename)
    
    plt.savefig(clustering_file_url, dpi = 240, bbox_inches='tight', pad_inches=0)



def save_image_umap(method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='', ground_truth=False, spot_size=-1, n_epoch=1000):

    if section_id == '':
        save_dir = os.path.join('output', tech, dataset, 'comprehensive')
    else:
        save_dir = os.path.join('output', tech, dataset, section_id, 'comprehensive')

    if(not os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    parameters = {
        'figure.figsize' : (5,4),
        'axes.titlesize' : 4,
    }

    mpl.rcParams.update(parameters)

    section_list = get_section_list(tech=tech, dataset=dataset, section_id=section_id)
    single_n_row, single_n_col, double_n_row, double_n_col = get_nrow_and_ncol(section_list=section_list, ground_truth=ground_truth)

    fig, umap_axes = plt.subplots(int(double_n_row), int(double_n_col))

    if(double_n_col == 1):
        umap_axes = np.array([umap_axes])

    umap_axes = umap_axes.ravel()
        
    index = 0
    for section_id in section_list:
        result_dir = os.path.join('output', tech, dataset, section_id, method)

        # read anndata and add clustering result
        adata = read_anndata(tech=tech, dataset=dataset, section_id=section_id, ground_truth=ground_truth)
        adata.var_names_make_unique()

        clustering_url = os.path.join(result_dir, 'clustering_res.csv')
        clustering_res = pd.read_csv(clustering_url, index_col=0)
        clustering_res.index = adata.obs.index

        adata.obs['clustering_res'] = clustering_res.astype('category')

        obs_df = adata.obs.dropna()

        if(ground_truth):
            ARI = adjusted_rand_score(obs_df['clustering_res'], obs_df['ground_truth'])
            ARI = round(ARI, 2)
            adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')

        adata.obs['clustering_res'] = clustering_res.astype('category')
        
        # # umap plotting
        # if method in ['STAGATE', 'HGCNAE']:
        if True:
            embedding_url = os.path.join(result_dir, 'embedding' + str(n_epoch) + '.csv')
            embedding_df = pd.read_csv(embedding_url, header=None)

            # from sklearn.decomposition import PCA
            # pca = PCA(n_components=20, random_state=42)
            # embedding_pca = pca.fit_transform(embedding_df)
            # embedding_df = pd.DataFrame(embedding_pca)

            embedding_df.index = adata.obs.index
            adata.obsm['embedding'] = embedding_df

            sc.pp.neighbors(adata, use_rep='embedding')
            sc.tl.umap(adata)
            adata.obs['clustering_res'] = adata.obs['clustering_res'].astype('category')
            sc.tl.paga(adata, groups='clustering_res')

        else:
            adata = load_umap_paga(adata, method=method, tech=tech, dataset=dataset, section_id=section_id)

        if spot_size == -1:
            n_cells = adata.obs.shape[0]
            spot_size = 72 * 6 / n_cells / double_n_col

        if(ground_truth):
            sc.pl.umap(adata, color = 'ground_truth', title = method, ax = umap_axes[2*index], show = False,\
                size = spot_size)
            sc.pl.umap(adata, color = 'clustering_res', title = section_id, ax = umap_axes[2*index+1], show = False, \
                size=spot_size)
        else:
            sc.pl.umap(adata, color='clustering_res', title=section_id, ax=umap_axes[index], show=False, \
                size=spot_size)

        plt.axis('off')

        index = index + 1

    # clear axes
    for i in range(umap_axes.shape[0]):
        umap_axes[i].set_xlabel('')
        umap_axes[i].set_ylabel('')
        umap_axes[i].legend().remove()
        umap_axes[i].set(adjustable='box', aspect='equal')
        umap_axes[i].axis('off')
    umap_filename = method + ' umap.png'
    umap_file_url = os.path.join(save_dir, umap_filename)
    
    plt.savefig(umap_file_url, dpi = 240, bbox_inches='tight', pad_inches=0)

def save_image_paga(method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='', ground_truth=False, spot_size=-1):
    
    if section_id == '':
        save_dir = os.path.join('output', tech, dataset, 'comprehensive')
    else:
        save_dir = os.path.join('output', tech, dataset, section_id, 'comprehensive')
    
    if(not os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    parameters = {
        'figure.figsize' : (5,4),
        'axes.titlesize' : 4,
    }

    mpl.rcParams.update(parameters)

    section_list = get_section_list(tech=tech, dataset=dataset, section_id=section_id)
    
    single_n_row, single_n_col, double_n_row, double_n_col = get_nrow_and_ncol(section_list=section_list, ground_truth=ground_truth)

    fig, paga_axes = plt.subplots(int(single_n_row), int(single_n_col))
    
    if(single_n_col == 1):
        paga_axes = np.array([paga_axes])

    paga_axes = paga_axes.ravel()
        
    index = 0
    for section_id in section_list:
        result_dir = os.path.join('output', tech, dataset, section_id, method)

        # read anndata and add clustering result 
        adata = read_anndata(tech=tech, dataset=dataset, section_id=section_id, ground_truth=ground_truth)
        adata.var_names_make_unique()
        
        clustering_url = os.path.join(result_dir, 'clustering_res.csv')
        clustering_res = pd.read_csv(clustering_url, index_col=0)
        clustering_res.index = adata.obs.index

        adata.obs['clustering_res'] = clustering_res.astype('category')
               
        # umap plotting
        # if method in ['STAGATE']:
        #     embedding_url = os.path.join(result_dir, 'embedding.csv')
        #     embedding_df = pd.read_csv(embedding_url, header=None)
        #     embedding_df.index = adata.obs.index
        #     adata.obsm['embedding'] = embedding_df

        #     sc.pp.neighbors(adata, use_rep='embedding')
        #     sc.tl.umap(adata)
        #     sc.tl.paga(adata, groups='clustering_res')
        # else:
        adata = load_umap_paga(adata, method=method, tech=tech, dataset=dataset, section_id=section_id)
        # paga plotting
        sc.pl.paga(adata, show=False, ax=paga_axes[index], node_size_scale=0.3, edge_width_scale=0.3,
            title=section_id)

        plt.axis('off')
        
        index = index + 1

    # clear axes
    for i in range(paga_axes.shape[0]):
        paga_axes[i].set_xlabel('')
        paga_axes[i].set_ylabel('')
        paga_axes[i].axis('off')

    paga_filename = method + ' paga.png'
    paga_file_url = os.path.join(save_dir, paga_filename)

    plt.savefig(paga_file_url, dpi = 240, bbox_inches='tight', pad_inches=0)

def save_image_data_denoising(method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='', ground_truth=False, spot_size=-1):

    if section_id == '':
        save_dir = os.path.join('output', tech, dataset, 'comprehensive')
    else:
        save_dir = os.path.join('output', tech, dataset, section_id, 'comprehensive')

    if(not os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    parameters = {
        'figure.figsize' : (5,4),
        'axes.titlesize' : 4,
    }

    mpl.rcParams.update(parameters)

    section_list = get_section_list(tech=tech, dataset=dataset, section_id=section_id)
    single_n_row, single_n_col, double_n_row, double_n_col = get_nrow_and_ncol(section_list=section_list, ground_truth=ground_truth)

    fig, denoise_axes = plt.subplots(int(double_n_row), int(double_n_col))

    if(double_n_col == 1):
        denoise_axes = np.array([denoise_axes])

    denoise_axes = denoise_axes.ravel()
            
    index = 0
    for section_id in section_list:
        result_dir = os.path.join('output', tech, dataset, section_id, method)

        # read anndata
        adata = read_anndata(tech=tech, dataset=dataset, section_id=section_id, ground_truth=ground_truth)
        adata.var_names_make_unique()
        
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        adata = adata[:, adata.var['highly_variable']]

        # read reconstructed expression
        recon_url = os.path.join('output', tech, dataset, section_id, method, 'reconstructed.csv')
        recon_df = pd.read_csv(recon_url, header=None)
        recon_df.index = adata.obs.index
        adata.layers['recon'] = recon_df

        plot_gene = 'ATP2B4'

        if spot_size == -1:
            n_cells = adata.obs.shape[0]
            spot_size = 72 * 6 / n_cells / double_n_col

        if adata.uns.get('spatial') is not None:
            sc.pl.spatial(adata, img_key='hires', color=plot_gene, show=False, ax=denoise_axes[2*index], 
                title=method+'_'+plot_gene, vmax='p99', colorbar_loc = None)
            sc.pl.spatial(adata, img_key='hires', color=plot_gene, show=False, ax=denoise_axes[2*index+1], 
                title=section_id, vmax='p99', layer='recon', colorbar_loc = None)
        else:
            sc.pl.spatial(adata, img_key='hires', color=plot_gene, show=False, ax=denoise_axes[2*index], 
                title=method+'_'+plot_gene, vmax='p99', colorbar_loc = None, spot_size=spot_size)
            sc.pl.spatial(adata, img_key='hires', color=plot_gene, show=False, ax=denoise_axes[2*index+1], 
                title=section_id, vmax='p99', layer='recon', colorbar_loc = None, spot_size=spot_size)
        
        plt.axis('off')

        index = index + 1
    
    # clear axes
    for i in range(denoise_axes.shape[0]):
        denoise_axes[i].set_xlabel('')
        denoise_axes[i].set_ylabel('')
        denoise_axes[i].axis('off')

    paga_filename = method + ' denoise.png'
    paga_file_url = os.path.join(save_dir, paga_filename)
    
    plt.savefig(paga_file_url, dpi = 240, bbox_inches='tight', pad_inches=0)

def save_image_svg(method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='', ground_truth=False, spot_size=-1):
    if section_id == '':
        save_dir = os.path.join('output', tech, dataset, 'comprehensive')
    else:
        save_dir = os.path.join('output', tech, dataset, section_id, 'comprehensive')
    
    if(not os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    parameters = {
        'figure.figsize' : (5,4),
        'axes.titlesize' : 4,
    }

    mpl.rcParams.update(parameters)

    section_list = get_section_list(tech=tech, dataset=dataset, section_id=section_id)
    
    # counting the number of plots
    num_plots = 0
    for section_id in section_list:
        num_plots = num_plots+1  # for clustering plot 
    SVG_metagene_url = os.path.join('output', tech, dataset, section_id, method, 'SVG_metagene.txt')

    gene_arr = []
    file_input = open(SVG_metagene_url)
    for line in file_input:
        gene_list = []
        for gene in line.split():
            gene_list.append(gene)
        gene_arr.append(gene_list)

        if(gene_list[0] == '0'):  # for metagene
            if(len(gene_list) == 1):
                continue
            num_plots = num_plots+1
        else:  # for SVG
            num_plots = num_plots + len(gene_list) - 1

    file_input.close()

    if(num_plots < 2):
        n_col = 1
    elif(num_plots < 5):
        n_col = 2
    elif(num_plots < 17):
        n_col = 4
    else:
        n_col = 6

    # maximum plot number
    if num_plots > 24:
        num_plots = 24

    n_row = num_plots / n_col
    if(num_plots % n_col != 0):
        n_row = n_row + 1

    fig, SVG_metagene_axes = plt.subplots(int(n_row), int(n_col))

    if(n_col == 1):
        SVG_metagene_axes = np.array([SVG_metagene_axes])

    SVG_metagene_axes = SVG_metagene_axes.ravel()

    index = 0

    if section_id == '':
        print('please specify section id')
        return
    else:
        result_dir = os.path.join('output', tech, dataset, section_id, method)

        # read anndata and add clustering result
        adata = read_anndata(tech=tech, dataset=dataset, section_id=section_id, ground_truth=ground_truth)

        from scipy.sparse import issparse

        if issparse(adata.X):
            adata.X = adata.X.todense()
        adata.var_names_make_unique()

        clustering_url = os.path.join(result_dir, 'clustering_res.csv')
        clustering_res = pd.read_csv(clustering_url, index_col=0)
        clustering_res.index = adata.obs.index

        adata.obs['clustering_res'] = clustering_res.astype('category')

        if spot_size == -1:
            n_cells = adata.obs.shape[0]
            spot_size = 72 * 6 / n_cells / n_col

        if adata.uns.get('spatial') is not None:
            sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
            title=section_id, ax = SVG_metagene_axes[index], show = False, colorbar_loc = None
                )
        else:
            sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
            title=section_id, ax = SVG_metagene_axes[index], show = False, colorbar_loc = None, 
            spot_size=spot_size\
                )

        index = index + 1

        SVG_metagene_url = os.path.join('output', tech, dataset, section_id, method, 'SVG_metagene.txt')

        gene_arr = []
        file_input = open(SVG_metagene_url)
        for line in file_input:
            gene_list = []
            for gene in line.split():
                gene_list.append(gene)
            gene_arr.append(gene_list)

        file_input.close()

        for index_domain, gene_list in enumerate(gene_arr):
            if(gene_list[0] == '0'):
                # plot metagene
                if(len(gene_list) == 1):
                    continue
                
                posi_gene_list, neg_gene_list = split_metagene(gene_list[1])
                n_cells = adata.obs.shape[0]
                adata.obs['metagene'] = np.zeros((n_cells, 1))
                
                for metagene in posi_gene_list:
                    m = adata.X[:, adata.var.index==metagene]
                    gene_values = pd.DataFrame(m, index=adata.obs.index)[0]
                    adata.obs['metagene'] = adata.obs['metagene'] + gene_values
                
                for metagene in neg_gene_list:
                    m = adata.X[:, adata.var.index==metagene]
                    gene_values = pd.DataFrame(m, index=adata.obs.index)[0]
                    adata.obs['metagene'] = adata.obs['metagene'] - gene_values
                    
                adata.obs['metagene'] = adata.obs['metagene'] - np.min(adata.obs['metagene'])

                if adata.uns.get('spatial') is not None:
                    sc.pl.spatial(adata, img_key='hires', color = 'metagene', \
                    title=str(index_domain)+':meta', ax = SVG_metagene_axes[index], show = False, colorbar_loc = None,
                        )
                else:
                    sc.pl.spatial(adata, img_key='hires', color = 'metagene', \
                    title=str(index_domain)+':meta', ax = SVG_metagene_axes[index], show = False, colorbar_loc = None,
                    spot_size=spot_size\
                        )
                    
                index = index + 1
                
            else:
                # plot each SVG
                for SVG_index in range(1, len(gene_list)):
                    SVG = gene_list[SVG_index]

                    if adata.uns.get('spatial') is not None:
                        sc.pl.spatial(adata, img_key='hires', color = SVG, \
                        title=str(index_domain)+':'+SVG, ax = SVG_metagene_axes[index], show = False, colorbar_loc = None,
                            )
                    else:
                        sc.pl.spatial(adata, img_key='hires', color =  SVG, \
                        title=str(index_domain)+':'+SVG, ax = SVG_metagene_axes[index], show = False, colorbar_loc = None,
                        spot_size=spot_size\
                            )
                    
                    index = index + 1
                    if index == 24:
                        break

            if index == 24:
                    break

        # clear axes
        for i in range(SVG_metagene_axes.shape[0]):
            SVG_metagene_axes[i].set_xlabel('')
            SVG_metagene_axes[i].set_ylabel('')
            SVG_metagene_axes[i].axis('off')
            SVG_metagene_axes[i].legend().remove()

        SVG_metagene_filename = method + ' SVG_metagene.png'
        # ANNOTION: different from previous 'SVG_metagene_url'
        SVG_metagene_url = os.path.join(save_dir, SVG_metagene_filename)

        # plt.axis('off')
        plt.savefig(SVG_metagene_url, dpi=240, bbox_inches='tight', pad_inches=0)

def save_image_batch_correction(method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='', ground_truth=False, spot_size=-1):
    if section_id == '':
        save_dir = os.path.join('output', tech, dataset, 'comprehensive')
    else:
        save_dir = os.path.join('output', tech, dataset, section_id, 'comprehensive')


    if(not os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    parameters = {
        'figure.figsize' : (5,4),
        'axes.titlesize' : 4,
    }

    mpl.rcParams.update(parameters)

    section_list = get_section_list(tech=tech, dataset=dataset, section_id=section_id)
    adata_list = []
    for section_id in section_list:
        adata_list_1 = read_anndata(tech=tech, dataset=dataset, section_id=section_id, ground_truth=ground_truth, after_cluster=True, method='CCST')
        adata_list = adata_list + adata_list_1

    single_n_row, single_n_col, double_n_row, double_n_col = get_nrow_and_ncol(section_list=adata_list, ground_truth=ground_truth)

    fig, clustering_axes = plt.subplots(int(double_n_row), int(double_n_col))

    if(double_n_col == 1):
        clustering_axes = np.array([clustering_axes])

    clustering_axes = clustering_axes.ravel()
    

    index = 0
    for section_id in section_list:
        
        # read anndata and add clustering result
        adata_list = read_anndata(tech=tech, dataset=dataset, section_id=section_id, ground_truth=ground_truth, after_cluster=True, method=method)
        for field, adata in enumerate(adata_list):
            adata.var_names_make_unique()
            if 'field_of_view' in adata.obs:
                field = adata.obs['field_of_view'][0]
            else:
                field = field+1

            if(ground_truth):
                obs_df = adata.obs.dropna()
                ARI = adjusted_rand_score(obs_df['clustering_res'], obs_df['ground_truth'])
                adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')

            if spot_size == -1:
                n_cells = adata.obs.shape[0]
                spot_size = 72 * 6 / n_cells / double_n_col

            if(ground_truth):
                if adata.uns.get('spatial') is not None:
                    sc.pl.spatial(adata, img_key='hires', color = 'ground_truth', legend_fontsize='xx-small',\
                    title=method, ax = clustering_axes[2*index], show = False, \
                        )

                    sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                    title=section_id+'-'+ str(field) + ':%.2f'%ARI, ax = clustering_axes[2*index+1], show = False, \
                        )
                else:
                    sc.pl.spatial(adata, img_key='hires', color = 'ground_truth', \
                    title=method, ax = clustering_axes[2*index], show = False,
                    spot_size=spot_size
                        )

                    sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                    title=section_id +'-'+str(field)+':%.2f'%ARI, ax = clustering_axes[2*index+1], show = False, 
                    spot_size=spot_size
                        )
            else:
                if adata.uns.get('spatial') is not None:
                    sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                    title=section_id +'-'+str(field), ax = clustering_axes[index], show = False,
                        )
                else:
                    sc.pl.spatial(adata, img_key='hires', color = 'clustering_res', \
                    title=section_id+'-'+str(field), ax = clustering_axes[index], show = False,
                    spot_size=spot_size\
                        )

            plt.axis('off')

            index = index + 1

    # clear axes
    
    for i in range(clustering_axes.shape[0]):
        clustering_axes[i].set_xlabel('')
        clustering_axes[i].set_ylabel('')
        clustering_axes[i].legend().remove()
        clustering_axes[i].axis('off')
    clustering_filename = method + ' batch correction.png'
    clustering_file_url = os.path.join(save_dir, clustering_filename)
    
    plt.savefig(clustering_file_url, dpi = 240, bbox_inches='tight', pad_inches=0)



def split_metagene(metagene):
    positive_gene_list = []
    negtive_gene_list = []
    str = ''
    flag = 0  # 0 for positive, 1 for negtive
    left_flag = 0
    for ch in metagene:
        if ch == '+':
            if flag == 0:
                positive_gene_list.append(str)
            else:
                negtive_gene_list.append(str)
            flag = 0
            str = ""
        elif ch == '-' and left_flag == 0:
            if flag == 0:
                positive_gene_list.append(str)
            else:
                negtive_gene_list.append(str)
            flag = 1
            str = ""
        elif ch == '(':
            left_flag = 1
        elif ch == ')':
            left_flag = 0
        else:
            str = str + ch
    
    if flag == 0:
        positive_gene_list.append(str)
    else:
        negtive_gene_list.append(str)

    return positive_gene_list, negtive_gene_list


def save_umap_paga(adata, method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='151676'):
    import pandas as pd
    from scipy import sparse

    save_dir = os.path.join('output', tech, dataset, section_id, method)

    umap_url = os.path.join(save_dir, 'umap.csv')
    paga_conn_url = os.path.join(save_dir, 'paga connectivities.npz')
    paga_conn_tree_url = os.path.join(save_dir, 'paga connectivities tree.npz')
    paga_spot_size_url = os.path.join(save_dir, 'paga spot size.csv')
    groups = 'clustering_res'


    umap_df = pd.DataFrame(adata.obsm['X_umap'], index=adata.obs.index)
    umap_df.to_csv(umap_url, index=False, header=False)

    paga_conn_csr_matrix = adata.uns['paga']['connectivities']
    sparse.save_npz(paga_conn_url, paga_conn_csr_matrix)

    paga_conn_tree_csr_matrix = adata.uns['paga']['connectivities_tree']
    sparse.save_npz(paga_conn_tree_url, paga_conn_tree_csr_matrix)

    paga_spot_size_df = pd.DataFrame(adata.uns[groups + '_sizes'])
    paga_spot_size_df.to_csv(paga_spot_size_url, index=False, header=False)

def load_umap_paga(adata, method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='151676'):
    import pandas as pd
    from scipy import sparse

    save_dir = os.path.join('output', tech, dataset, section_id, method)

    umap_url = os.path.join(save_dir, 'umap.csv')
    paga_conn_url = os.path.join(save_dir, 'paga connectivities.npz')
    paga_conn_tree_url = os.path.join(save_dir, 'paga connectivities tree.npz')
    paga_spot_size_url = os.path.join(save_dir, 'paga spot size.csv')
    groups = 'clustering_res'

    umap_df = pd.read_csv(umap_url, header=None)
    adata.obsm['X_umap'] = umap_df.values

    adata.uns['paga'] = {}

    paga_conn_df = sparse.load_npz(paga_conn_url)
    adata.uns['paga']['connectivities'] = paga_conn_df

    paga_conn_tree_csr_matrix1 = sparse.load_npz(paga_conn_tree_url)
    adata.uns['paga']['connectivities_tree'] = paga_conn_tree_csr_matrix1

    paga_spot_size = pd.read_csv(paga_spot_size_url, header=None).values[:,0]

    adata.uns[groups + '_sizes'] = paga_spot_size

    adata.uns['paga']['groups'] = groups

    return adata




import sklearn.neighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN, MeanShift, OPTICS, SpectralClustering, SpectralBiclustering
from sklearn.cluster import AffinityPropagation as AP
from sklearn.cluster import AgglomerativeClustering as AGG
from sklearn.mixture import BayesianGaussianMixture
import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import networkx as nx
from community import community_louvain
import matplotlib.cm as cm

from sklearn.decomposition import PCA


def do_clustering(method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='151676', num_cluster=-1, clustering_method='kmeans', nepoch=200):
    # read embedding and do clustering
    
    save_dir = os.path.join('output', tech, dataset, section_id, method)
    embedding_url = os.path.join(save_dir, 'embedding'+str(nepoch)+'.csv')
    clustering_url = os.path.join(save_dir, 'clustering_res.csv')

    embedding_df = pd.read_csv(embedding_url, header=None)
    select_cols = embedding_df.columns[:]
    embedding_df = embedding_df[select_cols]
    
    # print(embedding_df)
    
    if clustering_method == 'kmeans':
        if num_cluster == -1:
            print('kmeans do not support unknown clustering number')
            return
        clustering_res = KMeans(n_clusters=num_cluster, n_init=100).fit(embedding_df).labels_
        clustering_df = pd.DataFrame(clustering_res)
    elif clustering_method == 'bgmm':
        if num_cluster == -1:
            print('bgmm do not support unknown clustreing number')
            return
        knowledge = BayesianGaussianMixture(n_components=num_cluster,
                                            weight_concentration_prior_type='dirichlet_process',
                                            weight_concentration_prior=500
        ).fit(embedding_df)
        clustering_res = knowledge.predict(embedding_df)
        clustering_df = pd.DataFrame(clustering_res)
    elif clustering_method == 'AP':
        clustering_res = AP(random_state=0, damping=0.8, preference=-600).fit(embedding_df.values).labels_
        clustering_df = pd.DataFrame(clustering_res)
    elif clustering_method == 'dbscan':
        clustering_res = DBSCAN(eps=2, min_samples=30).fit(embedding_df.values).labels_
        clustering_df = pd.DataFrame(clustering_res)
    elif clustering_method == 'optics':
        clustering = OPTICS(min_samples=20, cluster_method='xi', xi=0.05, min_cluster_size=0.001).fit(embedding_df.values)
        clustering_res = clustering.labels_
        clustering_df = pd.DataFrame(clustering_res)
        reachability = clustering.reachability_[clustering.ordering_]
        plt.scatter(x=range(1, reachability.shape[0]+1), y=reachability)
        plt.show()
    elif clustering_method == 'spectral':
        if num_cluster == -1:
            print('spectral clustering do not support unknown clustering number')
            return
        clustering = SpectralClustering(n_clusters=num_cluster, gamma=2.0, 
                                        n_neighbors=5, affinity='nearest_neighbors',
                                        assign_labels='cluster_qr').fit(embedding_df.values)
        clustering_res = clustering.labels_
        clustering_df = pd.DataFrame(clustering_res)
    elif clustering_method == 'louvain':
        # build networkx graph
        # KNN
        # nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=3).fit(embedding_df.values)
        # distances, indices = nbrs.kneighbors(embedding_df.values)

        # radius
        radius = search_r(embedding_df.values, start_r=0.1, end_r=1.0, min_ratio=6.0, max_ratio=7.0)

        nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(embedding_df.values)
        distances, indices = nbrs.radius_neighbors(embedding_df.values, return_distance=True)

        source_node_list = []
        target_node_list = []
        edge_weight_list = []

        for source_node, target_node_sublist, edge_weight_sub_list in zip(embedding_df.index, indices, distances):
            for target_node, edge_weight in zip(target_node_sublist, edge_weight_sub_list):
                if source_node == target_node:
                    continue
                source_node_list.append(source_node)
                target_node_list.append(target_node)
                edge_weight_list.append(1.0/(edge_weight+0.00001))

        edge_list_df = pd.DataFrame()
        edge_list_df[0] = source_node_list
        edge_list_df['b'] = target_node_list
        edge_list_df['weight'] = edge_weight_list

        # filter edges
        # mean_edge_weight = edge_list_df['weight'].mean()
        # edge_list_df = edge_list_df[edge_list_df['weight'] < mean_edge_weight]

        print(edge_list_df)

        G = nx.from_pandas_edgelist(edge_list_df, 0, 'b', ['weight'])
        partition = community_louvain.best_partition(G, resolution=0.2)
        clustering_res = partition.keys()
        # print(clustering_res)
        clustering_df = pd.DataFrame(clustering_res)

        # draw the graph
        pos = nx.spring_layout(G)
        cmap = cm.get_cmap('viridis', max(clustering_res)+1)
        nx.draw_networkx_nodes(G, pos, clustering_res, node_size=10, cmap=cmap, 
                               node_color=list(clustering_res))
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.show()
        
    elif clustering_method == 'leiden':
        pass
    elif clustering_method == 'mclust':
        # replace the character
        # method = method
        # tech = tech.replace(' ', '%')
        # dataset = dataset.replace(' ', '%')
        # section_id = section_id.replace(' ', '%')
        
        
        # script_cmd = 'Rscript do_mclust.R ' + method + ' ' + tech + ' ' + dataset + ' ' + section_id + ' ' + str(num_cluster) + ' ' + str(nepoch)
        # stats = os.system(script_cmd)
        # print('return stat: ' + str(stats))

        # do mclust
        # dim reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=30, random_state=42)
        embedding_pca = pca.fit_transform(embedding_df)

        # do clustering
        clustering_res = mclust_R(embedding_pca, num_cluster=num_cluster)
        clustering_df = pd.DataFrame(clustering_res, dtype='category')
        clustering_url = os.path.join(save_dir, 'clustering_res.csv')
        clustering_df.to_csv(clustering_url)

        return
    elif clustering_method == 'AGG':
        if num_cluster == -1:
            print('agglomerative clustering do not support unknown clustering number')
            return
        clustering = AGG(n_clusters=num_cluster, linkage='ward').fit(embedding_df)
        clustering_res = clustering.labels_
        clustering_df = pd.DataFrame(clustering_res)
    elif clustering_method == 'meanshift':
        clustering = MeanShift(bandwidth=1.5, bin_seeding=True, min_bin_freq=6).fit(embedding_df)
        clustering_res = clustering.labels_
        clustering_df = pd.DataFrame(clustering_res)
    elif clustering_method == 'SpactralBiClustering':
        clustering = SpectralBiclustering(n_clusters=num_cluster, random_state=0).fit(embedding_df)
        clustering_res = clustering.row_labels_
        clustering_df = pd.DataFrame(clustering_res)
    # save result
    clustering_df.to_csv(clustering_url)



def search_r(features, start_r, end_r, min_ratio, max_ratio):
    num_node = features.shape[0]

    while(end_r - start_r > 0.0001):
        mid_r = (start_r + end_r) / 2
        nbrs = sklearn.neighbors.NearestNeighbors(radius=mid_r).fit(features)
        distances, indices = nbrs.radius_neighbors(features, return_distance=True)
        num_edge = 0
        for distance_list in distances:
            num_edge += distance_list.shape[0]
        edge_node_ratio = num_edge / float(num_node)

        if edge_node_ratio < min_ratio:
            start_r = mid_r
        elif edge_node_ratio > max_ratio:
            end_r = mid_r
        else:
            print('radius: ' + str(mid_r))
            print('edge_node_ratio: ' + str(edge_node_ratio))
            return mid_r
    if edge_node_ratio < min_ratio:
        raise ValueError('you should try bigger end_r or smaller min_ratio')
    elif edge_node_ratio > max_ratio:
        raise ValueError('you should try smaller start_r or bigger max_ratio')
    
from sklearn.decomposition import PCA    
def pca_all():
    section_list = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
    # section_list = ['151676']
    for section_id in section_list:
        save_dir = os.path.join('output', '10x Visium', 'DLPFC', section_id, 'HGCNAE')
        embedding_source_url = os.path.join(save_dir, 'embedding1004.csv')
        embedding_target_url = os.path.join(save_dir, 'embedding1005.csv')

        embedding_df = pd.read_csv(embedding_source_url, header=None, index_col=None)
        select_cols = embedding_df.columns[:]
        embedding_df = embedding_df[select_cols]

        pca = PCA(n_components=30, random_state=42)
        embedding_pca = pca.fit_transform(embedding_df)
        embedding_pca_df = pd.DataFrame(embedding_pca)
        embedding_pca_df.to_csv(embedding_target_url, index=False, header=False)

from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import SpectralEmbedding, Isomap, TSNE
def dim_reduction(method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='151676', reduction_method='pca', from_epoch=-1, to_epoch=-1, degree=-1, coef0=-1, n_components=20):
    save_dir = os.path.join('output', tech, dataset, section_id, method)
    embedding_source_url = os.path.join(save_dir, 'embedding' + str(from_epoch) + '.csv')
    embedding_target_url = os.path.join(save_dir, 'embedding' + str(to_epoch) + '.csv')

    embedding_df = pd.read_csv(embedding_source_url, header=None, index_col=None)
    select_cols = embedding_df.columns[:]
    embedding_df = embedding_df[select_cols]

    if reduction_method == 'pca':
        pca = PCA(n_components=n_components, random_state=42)
        embedding_reduc = pca.fit_transform(embedding_df)
        embedding_reduc_df = pd.DataFrame(embedding_reduc)
    elif reduction_method == 'kernel pca':
        transformer = KernelPCA(n_components=n_components, kernel='linear', random_state=42, eigen_solver='dense', degree=degree, coef0=coef0, remove_zero_eig=True)
        embedding_reduc = transformer.fit_transform(embedding_df)
        embedding_reduc_df = pd.DataFrame(embedding_reduc)
    elif reduction_method == 'robust pca':
        pass
    elif reduction_method == 'lle': # locally linear embedding
        transformer = LLE(n_components=coef0, n_neighbors=degree, random_state=42, method='modified', eigen_solver='dense')
        embedding_reduc = transformer.fit_transform(embedding_df)
        embedding_reduc_df = pd.DataFrame(embedding_reduc)
    elif reduction_method == 'spectral': # Laplacian Eigenmaps
        transformer = SpectralEmbedding(n_components=coef0, random_state=42)
        embedding_reduc = transformer.fit_transform(embedding_df)
        embedding_reduc_df = pd.DataFrame(embedding_reduc)
    elif reduction_method == 'isomap':
        transformer = Isomap(n_components=coef0, n_neighbors=degree)
        embedding_reduc = transformer.fit_transform(embedding_df)
        embedding_reduc_df = pd.DataFrame(embedding_reduc)
    elif reduction_method == 'tsne':
        transformer = TSNE(n_components=coef0, perplexity=degree, random_state=42, init='pca')
        embedding_reduc = transformer.fit_transform(embedding_df)
        embedding_reduc_df = pd.DataFrame(embedding_reduc)

    # embedding_reduc_df.to_csv(embedding_target_url, index=False, header=False)
    return embedding_reduc_df.values

import ot
def refine_label(method='STAGATE', tech='10x Visium', dataset='DLPFC', section_id='151676', radius=40):
    
    adata = read_anndata(tech=tech, dataset=dataset, section_id=section_id)
    save_dir = os.path.join('output', tech, dataset, section_id, method)
    clustering_url = os.path.join(save_dir, 'clustering_res.csv')
    clustering_df_old = pd.read_csv(clustering_url, index_col=0)

    n_neigh = radius
    new_type = []
    old_type = clustering_df_old.values.flatten()
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]]) # find nearest k neighbors
        
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]
    clustering_df_new = pd.DataFrame(new_type)
    
    clustering_df_new.to_csv(clustering_url)   
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type


# def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
#     """\
#     Clustering using the mclust algorithm.
#     The parameters are the same as those in the R package mclust.
#     """
    
#     np.random.seed(random_seed)
#     import rpy2.robjects as robjects
#     robjects.r.library("mclust")

#     import rpy2.robjects.numpy2ri
#     rpy2.robjects.numpy2ri.activate()
#     r_random_seed = robjects.r['set.seed']
#     r_random_seed(random_seed)
#     rmclust = robjects.r['Mclust']
    
#     res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
#     mclust_res = np.array(res[-2])

#     adata.obs['mclust'] = mclust_res
#     adata.obs['mclust'] = adata.obs['mclust'].astype('int')
#     adata.obs['mclust'] = adata.obs['mclust'].astype('category')
#     return adata

def mclust_R(data, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    return mclust_res


def concat_emb():
    tech = '10x Visium'
    dataset = 'DLPFC'
    section_list = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
    for section_id in section_list:
        print(section_id)
        po_emb_url = os.path.join('output', tech, dataset, section_id, 'HGCNAE', 'embedding1000.csv')
        lo_emb_url = os.path.join('output', tech, dataset, section_id, 'HGCNAE', 'embedding1001.csv')
        eu_emb_url = os.path.join('output', tech, dataset, section_id, 'GCNAE', 'embedding1002.csv')

        po_emb_df = pd.read_csv(po_emb_url, header=None, index_col=None)
        lo_emb_df = pd.read_csv(lo_emb_url, header=None, index_col=None)
        eu_emb_df = pd.read_csv(eu_emb_url, header=None, index_col=None)

        # print(po_emb_df.shape)
        # print(lo_emb_df.shape)
        # print(eu_emb_df.shape)

        # normalize part
        # po_emb_df = po_emb_df / (po_emb_df.mean())
        # lo_emb_df = po_emb_df / (lo_emb_df.mean())
        # eu_emb_df = po_emb_df / (eu_emb_df.mean())

        emb_df = pd.concat([po_emb_df, eu_emb_df], axis=1)
        print(emb_df.shape)

        save_dir = os.path.join('output', tech, dataset, section_id, 'HGCLAE')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_url = os.path.join(save_dir, 'embedding1004.csv')
        emb_df.to_csv(save_url, index=False, header=False)

def calculate_ari(tech, dataset, section_id, method):
    data_dir = os.path.join('data', tech, dataset, section_id)
    save_dir = os.path.join('output', tech, dataset, section_id, method)
    ground_truth_url = os.path.join(data_dir, section_id+'_truth.txt')
    clustering_url = os.path.join(save_dir, 'clustering_res.csv')
    
    # ground_truth_df = pd.read_csv(ground_truth_url, sep='\t', header=None, index_col=0).fillna('')
    ground_truth_df = pd.read_csv(ground_truth_url, sep='\t', header=None, index_col=0)
    clustering_df = pd.read_csv(clustering_url, index_col=0)
    clustering_df.index = ground_truth_df.index
    
    obs_df = pd.concat([ground_truth_df, clustering_df], axis=1)
    obs_df = obs_df.dropna()
    obs_df.columns = ['ground_truth', 'clustering_res']

    ARI = adjusted_rand_score(obs_df['ground_truth'], obs_df['clustering_res'])
    # ARI = normalized_mutual_info_score(obs_df['ground_truth'], obs_df['clustering_res'])
    ARI = round(ARI, 2)

    # ARI = adjusted_rand_score(clustering_df.values.flatten(), ground_truth_df.values.flatten())
    
    return ARI

def calculate_nmi(tech, dataset, section_id, method):
    data_dir = os.path.join('data', tech, dataset, section_id)
    save_dir = os.path.join('output', tech, dataset, section_id, method)
    ground_truth_url = os.path.join(data_dir, section_id+'_truth.txt')
    clustering_url = os.path.join(save_dir, 'clustering_res.csv')
    
    # ground_truth_df = pd.read_csv(ground_truth_url, sep='\t', header=None, index_col=0).fillna('')
    ground_truth_df = pd.read_csv(ground_truth_url, sep='\t', header=None, index_col=0)
    clustering_df = pd.read_csv(clustering_url, index_col=0)
    clustering_df.index = ground_truth_df.index
    
    obs_df = pd.concat([ground_truth_df, clustering_df], axis=1)
    obs_df = obs_df.dropna()
    obs_df.columns = ['ground_truth', 'clustering_res']

    # ARI = adjusted_rand_score(obs_df['ground_truth'], obs_df['clustering_res'])
    NMI = normalized_mutual_info_score(obs_df['ground_truth'], obs_df['clustering_res'])
    NMI = round(NMI, 2)

    # ARI = adjusted_rand_score(clustering_df.values.flatten(), ground_truth_df.values.flatten())
    
    return NMI

import torch
import random
def fix_seed(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

    