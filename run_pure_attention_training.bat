@echo off
echo 启动纯自注意力LDM训练（优化版本）
echo =====================================

rem 设置输出目录
set OUTPUT_DIR=ldm_pure_attention

rem 检查CUDA是否可用
python -c "import torch; print('CUDA可用:',torch.cuda.is_available())"

echo 开始训练...
python train_with_pure_attention.py ^
    --output_dir %OUTPUT_DIR% ^
    --batch_size 8 ^
    --gradient_accumulation_steps 2 ^
    --epochs 200 ^
    --lr 1e-4 ^
    --beta_schedule cosine ^
    --scheduler_type ddim ^
    --mixed_precision fp16 ^
    --save_images ^
    --eval_steps 500 ^
    --logging_steps 50

echo 训练完成！ 