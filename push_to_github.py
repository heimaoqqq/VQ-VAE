"""
将本地更改推送到GitHub
"""
import os
import subprocess

def run_cmd(cmd):
    """运行命令并打印输出"""
    print(f"执行: {cmd}")
    process = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(process.stdout)
    if process.stderr:
        print(f"错误: {process.stderr}")
    return process.returncode

def push_to_github():
    """提交更改并推送到GitHub"""
    # 添加所有更改
    run_cmd("git add .")
    
    # 提交更改
    commit_message = "修复VQ-VAE下采样倍数计算，确保正确的潜在空间尺寸"
    run_cmd(f'git commit -m "{commit_message}"')
    
    # 推送到GitHub
    run_cmd("git push")
    
    print("已成功将更改推送到GitHub")

if __name__ == "__main__":
    push_to_github() 