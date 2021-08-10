import os
import sys 

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(root)
sys.path.append(root)
from external_lib.MedCommon.utils.mask_bounding_utils import MaskBoundingUtils
from external_lib.MedCommon.utils.data_io_utils import DataIO
from external_lib.MedCommon.utils.datasets_utils import DatasetsUtils
from external_lib.MedCommon.utils.image_postprocessing_utils import ImagePostProcessingUtils

from glob import glob
from tqdm import tqdm
import numpy as np

import SimpleITK as sitk

OUT_ROOT = '/data/medical/brain/Cerebrovascular/segmenation/result/analysis_result'
os.makedirs(OUT_ROOT, exist_ok=True)

def analyze_mask_boundary(mask_root = '/data/medical/brain/Cerebrovascular/segmenation/renamed_masks', 
    out_filename='boundary_info_ori_mask.txt'):
    
    # mask_root = '/data/medical/brain/Cerebrovascular/segmenation/renamed_masks'
    max_depth = 0
    max_height = 0
    max_width = 0
    logs = []

    depths = []
    heights = []
    widths = []
    for mask_name in tqdm(os.listdir(mask_root)):
        mask_file = os.path.join(mask_root, mask_name)
        z_min, y_min, x_min, z_max, y_max, x_max = MaskBoundingUtils.extract_mask_file_bounding(mask_file)
        depth = z_max - z_min
        height = y_max - y_min
        width = x_max - x_min
        if depth > max_depth:
            max_depth = depth
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width
        depths.append(depth)
        heights.append(height)
        widths.append(width)

        log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(depth, height, width, mask_name)
        logs.append(log_info)
    log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(max_depth, max_height, max_width, 'max info')
    logs.append(log_info)
    log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(
        np.array(depths).sum()/len(depths), np.array(heights).sum()/len(heights), np.array(widths).sum()/len(widths), 'mean info')
    logs.append(log_info)
    log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(
        depths[len(depths)//2], heights[len(heights)//2], widths[len(widths)//2], 'middle info')
    logs.append(log_info)    

    
    out_file = os.path.join(OUT_ROOT, out_filename)
    with open(out_file, 'w') as f:
        f.write('\n'.join(logs))

    for log in logs:
        print(log)

def analyze_image_boundary():
    image_root = '/data/medical/brain/Cerebrovascular/segmenation/images'
    max_depth = 0
    max_height = 0
    max_width = 0
    logs = []

    depths = []
    heights = []
    widths = []    
    for suid in tqdm(os.listdir(image_root)):
        image_file = os.path.join(image_root, suid)
        image = DataIO.load_dicom_series(image_file)['sitk_image']
        width,height,depth = image.GetSize()
        if depth > max_depth:
            max_depth = depth
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width

        depths.append(depth)
        heights.append(height)
        widths.append(width)            
        log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(depth, height, width, suid)
        logs.append(log_info)
        print(log_info)
    log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(max_depth, max_height, max_width, 'image info')
    logs.append(log_info)
    log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(
        np.array(depths).sum()/len(depths), np.array(heights).sum()/len(heights), np.array(widths).sum()/len(widths), 'mean info')
    logs.append(log_info)
    log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(
        depths[len(depths)//2], heights[len(heights)//2], widths[len(widths)//2], 'middle info')    
    out_file = os.path.join(OUT_ROOT, 'boundary_info_ori_images.txt')
    with open(out_file, 'w') as f:
        f.write('\n'.join(logs))

    for log in logs:
        print(log)            

def mask_denoise(kernel_radius):
    image_root = '/data/medical/brain/Cerebrovascular/segmenation/images'
    mask_root = '/data/medical/brain/Cerebrovascular/segmenation/renamed_masks'
    out_mask_root = '/data/medical/brain/Cerebrovascular/segmenation/masks_erode_k{}'.format(kernel_radius)
    os.makedirs(out_mask_root, exist_ok=True)
    mask_lable = 7
    # kernel_radius = 1
    for suid in tqdm(os.listdir(image_root)):
        mask_file = os.path.join(mask_root, '{}.mha'.format(suid))
        sitk_mask = sitk.ReadImage(mask_file)
        
        new_sitk_mask = None

        erode_filter = sitk.BinaryErodeImageFilter()
        erode_filter.SetForegroundValue(mask_lable)
        erode_filter.SetBackgroundValue(0)
        erode_filter.SetKernelRadius(kernel_radius)
        new_sitk_mask = erode_filter.Execute(sitk_mask)
        
        dilation_filter = sitk.BinaryDilateImageFilter()
        dilation_filter.SetForegroundValue(mask_lable)
        dilation_filter.SetBackgroundValue(0)
        dilation_filter.SetKernelRadius(kernel_radius)
        new_sitk_mask = dilation_filter.Execute(new_sitk_mask)



        out_file = os.path.join(out_mask_root, '{}.nii.gz'.format(suid))
        sitk.WriteImage(new_sitk_mask, out_file)

def generate_image_mask_pairs_onecase(ref_mask_root, out_root, image_root, mask_root, suid):
    out_image_root = os.path.join(out_root, 'images')
    os.makedirs(out_image_root, exist_ok=True)
    out_mask_root = os.path.join(out_root, 'masks')
    os.makedirs(out_mask_root, exist_ok=True)
    ref_mask_file = os.path.join(ref_mask_root, '{}.nii.gz'.format(suid))
    boundary_info = MaskBoundingUtils.extract_mask_file_bounding(ref_mask_file)
    in_image_file = os.path.join(image_root, '{}'.format(suid))
    in_mask_file = os.path.join(mask_root, '{}.mha'.format(suid))
    out_image_file = os.path.join(out_image_root, '{}.nii.gz'.format(suid))
    out_mask_file = os.path.join(out_mask_root, '{}.nii.gz'.format(suid))
    # MaskBoundingUtils.extract_target_area_by_boundary_info(in_image_file, out_image_file, boundary_info, True)
    # MaskBoundingUtils.extract_target_area_by_boundary_info(in_mask_file, out_mask_file, boundary_info, False)        
    MaskBoundingUtils.extract_segmentation_pairs_by_boundary_info(in_image_file, in_mask_file, out_image_file, out_mask_file, boundary_info, True)

def generate_image_mask_pairs_singletask(ref_mask_root, out_root, image_root, mask_root, suids):
    for suid in tqdm(suids):
        try:
            generate_image_mask_pairs_onecase(ref_mask_root, out_root, image_root, mask_root, suid)
        except Exception as e:
            print('====> Error case:\t', suid)
            print(e)

def generate_image_mask_pairs(ref_mask_root, 
        out_root, 
        image_root = '/data/medical/brain/Cerebrovascular/segmenation/images', 
        mask_root = '/data/medical/brain/Cerebrovascular/segmenation/renamed_masks', 
        process_num=12
    ):
    series_uids = []
    series_uids = os.listdir(image_root)
    num_per_process = (len(series_uids) + process_num - 1)//process_num

    # this for single thread to debug
    # generate_image_mask_pairs_singletask(
    #                 ref_mask_root, out_root, image_root, mask_root, series_uids)

    # this for run 
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()

    results = []

    print(len(series_uids))
    for i in range(process_num):
        sub_series_uids = series_uids[num_per_process*i:min(num_per_process*(i+1), len(series_uids))]
        print(len(sub_series_uids))
        result = pool.apply_async(generate_image_mask_pairs_singletask, 
            args=(ref_mask_root, out_root, image_root, mask_root, sub_series_uids))
        results.append(result)

    pool.close()
    pool.join()    


def split_ds():
    image_root = '/data/medical/brain/Cerebrovascular/segmenation_erode_k2/images'
    out_config_dir = '/data/medical/brain/Cerebrovascular/segmenation_erode_k2/config'
    DatasetsUtils.split_ds(image_root, out_config_dir, 0.8, 0.001)
    # image_root = '/fileser/zhangwd/data/lung/airway/segmentation/images'
    # out_config_dir = '/fileser/zhangwd/data/lung/airway/segmentation/config'
    # DatasetsUtils.split_ds(image_root, out_config_dir, 0.8, 0.001)

def fix_mask_label_to_1(mask_root='/data/medical/brain/Cerebrovascular/segmenation_erode_k2/masks', out_mask_root='/data/medical/brain/Cerebrovascular/segmenation_erode_k2/masks_label1'):
    os.makedirs(out_mask_root, exist_ok=True)
    for f in tqdm(os.listdir(mask_root)):
        infile = os.path.join(mask_root, f)
        sitk_mask = sitk.ReadImage(infile)
        sitk_mask_new = ImagePostProcessingUtils.fix_mask_label(sitk_mask, 7)
        out_file = os.path.join(out_mask_root, f)
        sitk.WriteImage(sitk_mask_new, out_file)
    

if __name__ == '__main__':
    # 1. 标注的原始mask不做任何处理，统计边界信息
    # analyze_mask_boundary()
    # analyze_image_boundary()
    
    # 2. 消除杂质点
    # mask_denoise(1)
    # mask_denoise(2)

    # 3. 在经过开（先腐蚀后膨胀）操作后，统计mask边界信息
    # analyze_mask_boundary('/data/medical/brain/Cerebrovascular/segmenation/masks_erode_k1', 'boundary_info_ori_mask_erode_k1.txt')
    # analyze_mask_boundary('/data/medical/brain/Cerebrovascular/segmenation/masks_erode_k2', 'boundary_info_ori_mask_erode_k2.txt')
    # 4. 根据步骤3的mask边界结果，利用原始影像和mask生成训练用的数据对
    # generate_image_mask_pairs(
    #     '/data/medical/brain/Cerebrovascular/segmenation/masks_erode_k1', 
    #     '/data/medical/brain/Cerebrovascular/segmenation_erode_k1'
    #     )
    # generate_image_mask_pairs(
    #     '/data/medical/brain/Cerebrovascular/segmenation/masks_erode_k2', 
    #     '/data/medical/brain/Cerebrovascular/segmenation_erode_k2'
    #     )
    # 5. 划分数据集
    split_ds()

    # 6. 修改mask文件的label
    # fix_mask_label_to_1('/data/medical/brain/Cerebrovascular/segmenation_erode_k1/masks', '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/masks_label1')
    # fix_mask_label_to_1('/data/medical/brain/Cerebrovascular/segmenation_erode_k2/masks', '/data/medical/brain/Cerebrovascular/segmenation_erode_k2/masks_label2')



        


