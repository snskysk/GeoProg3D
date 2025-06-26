your_project_root="/workspace/t2024"
config_file="${your_project_root}/geoprog3d/config/config.yml"
process_finish_file="${your_project_root}/LoG/renders4visprog/process_finish.txt"

cd ${your_project_root}/LoG
python apps/get_visprog_cam_and_npy.py --config ${config_file}

cd ${your_project_root}/LangSplat/eval_custom
python get_visprog_simPix.py --config ${config_file}

cd ${your_project_root}/LoG
python apps/get_gaussians_indices_at_arbitrary_view.py --config ${config_file}

cd ${your_project_root}/LoG
python apps/visualize_masked_topdown_gs.py --config ${config_file}

touch ${process_finish_file}