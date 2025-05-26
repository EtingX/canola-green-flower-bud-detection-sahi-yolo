

from sahi_predict_yx import *
#
# model_address_list = [r"J:\final_canola_dataset_method\640_crop_dataset\final_model\5th_640_n_Head_C3K2_PPA_SPDConv_final\weights\best.pt",
#                       r"J:\final_canola_dataset_method\640_crop_dataset\final_model\640_n\weights\best.pt",
#                       r'J:\final_canola_dataset_method\960_crop_dataset\960_FASFFHead_C3K2_PPA_SPDConv\960_FASFFHead_C3K2_PPA_SPDConv\weights\best.pt',
#                       r"J:\final_canola_dataset_method\960_crop_dataset\960_n\960_n\weights\best.pt",
#                       r"J:\final_canola_dataset_method\single_640_crop_dataset\single_new_640_n_FASFFHead_C3K2_PPA_SPDConv\single_new_640_n_FASFFHead_C3K2_PPA_SPDConv\weights\best.pt",
#                       r"J:\final_canola_dataset_method\single_640_crop_dataset\single_new_640_n_small_layer\single_new_640_n_small_layer\weights\best.pt"]
#
model_address_list = [r"J:\迅雷下载\960_n_SPDConv\weights\best.pt",
                      r"J:\迅雷下载\960_n_C3K2_PPA\weights\best.pt",
                      r"J:\迅雷下载\960_n_FASFFHead\weights\best.pt",
                      r"J:\迅雷下载\960_n_SPDConv_C3k2_PPA\weights\best.pt",
                      r"J:\迅雷下载\960_n_SPDConv_FASFFHead\weights\best.pt",
                      r"J:\迅雷下载\960_n_C3K2_PPA_FASFFHead\weights\best.pt"]


for model_address in model_address_list:
    for threshold in [0.6]:
        if str(640) in model_address:
            predict(
                model_type="ultralytics",
                model_path=model_address,
                model_device="cuda:0",  # or 'cuda:0'
                model_confidence_threshold=threshold,
                source=r"J:\final_canola_dataset_method\final_labeled_dataset\images\test",
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.45,
                overlap_width_ratio=0.45,
                export_txt=True
            )

        elif str(960) in model_address:
            predict(
                model_type="ultralytics",
                model_path=model_address,
                model_device="cuda:0",  # or 'cuda:0'
                model_confidence_threshold=threshold,
                source=r"J:\final_canola_dataset_method\final_labeled_dataset\images\test",
                slice_height=960,
                slice_width=960,
                overlap_height_ratio=0.45,
                overlap_width_ratio=0.45,
                export_txt=True
            )

        else:
            print('Nothing')

# model_address = r"J:\迅雷下载\960_SPDConv_FASFFHead\weights\best.pt"
# threshold = 0.6
# predict(
#         model_type="ultralytics",
#         model_path=model_address,
#         model_device="cuda:0",  # or 'cuda:0'
#         model_confidence_threshold=threshold,
#         source=r"J:\final_canola_dataset_method\final_labeled_dataset\images\test",
#         slice_height=640,
#         slice_width=640,
#         overlap_height_ratio=0.45,
#         overlap_width_ratio=0.45,
#         export_txt=True
# )