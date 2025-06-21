"""
将修改推送到GitHub
"""

import subprocess

def run_command(cmd):
    """运行Shell命令"""
    print(f"执行: {cmd}")
    process = subprocess.run(cmd, shell=True, text=True)
    return process.returncode == 0

def push_to_github():
    """提交更改并推送到GitHub"""
    # 添加所有更改
    if not run_command("git add ."):
        print("添加文件失败")
        return False
    
    # 提交更改
    commit_message = "添加VQ-VAE潜在空间分析工具：用于评估潜在空间质量及扩散适应性"
    if not run_command(f'git commit -m "{commit_message}"'):
        print("提交更改失败")
        return False
    
    # 推送到GitHub
    if not run_command("git push"):
        print("推送到GitHub失败")
        return False
    
    print("成功将更改推送到GitHub!")
    return True

if __name__ == "__main__":
    push_to_github() 