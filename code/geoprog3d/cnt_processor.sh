your_project_root="/workspace/t2024"
config_file="${your_project_root}/geoprog3d/config/config.yml"
process_finish_file="${your_project_root}/LoG/renders4visprog/process_finish.txt"
ins_fname_json="${your_project_root}/geoprog3d/ins_fname.json"

cd ${your_project_root}/LoG
python apps/get_visprog_gaussians_indices.py --config ${config_file}

cd ${your_project_root}/LoG
python apps/counting_topdown_view.py --config ${config_file}

touch ${process_finish_file}