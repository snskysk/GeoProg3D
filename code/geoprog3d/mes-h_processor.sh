your_project_root="/workspace/t2024"
config_file="${your_project_root}/geoprog3d/config/config.yml"
process_finish_file="${your_project_root}/LoG/renders4visprog/process_finish.txt"

cd ${your_project_root}/LoG
python apps/get_visprog_area_height.py --config ${config_file}

touch ${process_finish_file}