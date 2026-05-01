"""
重新生成 figures_new 目录下的图片 2-9
使用最新的融合特征级联结果
"""
import os
import sys
import subprocess
import time

# 设置输出目录为 figures_new
os.environ['FIGURES_DIR'] = 'figures_new'
os.environ['RESULTS_DIR'] = 'results_new'

# 确保目录存在
os.makedirs('figures_new', exist_ok=True)
os.makedirs('results_new', exist_ok=True)

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# 切换到项目根目录
os.chdir(base_dir)

def run_script(script_name, description):
    """运行单个脚本并显示进度"""
    print(f"\n{'='*60}")
    print(f"正在生成: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*60}")
    
    script_path = os.path.join(script_dir, script_name)
    if not os.path.exists(script_path):
        print(f"警告: 脚本不存在: {script_path}")
        return False
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=base_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[OK] 完成 ({elapsed:.1f}秒)")
            if result.stdout:
                # 只显示最后几行输出
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        print(f"  {line}")
            return True
        else:
            print(f"[错误] 失败 ({elapsed:.1f}秒)")
            if result.stderr:
                print("错误信息:")
                for line in result.stderr.strip().split('\n')[-10:]:
                    if line.strip():
                        print(f"  {line}")
            return False
    except Exception as e:
        print(f"[异常] {str(e)}")
        return False

def main():
    print("\n" + "="*60)
    print("重新生成 figures_new 目录下的图片 2-9")
    print("使用最新的融合特征级联结果")
    print("="*60)
    
    # 定义要运行的脚本列表
    scripts = [
        ("generate_figure2_combined.py", "Figure 2: 特征空间演变"),
        ("generate_figure3_combined.py", "Figure 3: 物理机制验证"),
        ("generate_figure5_combined.py", "Figure 5: L2 选择策略（使用融合特征）"),
        ("generate_figure7_l2_results.py", "Figure 7: L2 结果综合分析（使用融合特征级联结果）"),
        ("generate_figure8_combined.py", "Figure 8: 鲁棒性和安全性"),
        ("generate_figure9_combined.py", "Figure 9: 物理可解释性"),
    ]
    
    print(f"\n总共 {len(scripts)} 个图片待生成")
    print(f"输出目录: figures_new/")
    print(f"结果目录: results_new/")
    
    # 自动继续（非交互模式）
    print("\n开始生成...")
    
    # 运行所有脚本
    success_count = 0
    fail_count = 0
    failed_scripts = []
    
    for script_name, description in scripts:
        success = run_script(script_name, description)
        if success:
            success_count += 1
        else:
            fail_count += 1
            failed_scripts.append((script_name, description))
    
    # 汇总结果
    print("\n" + "="*60)
    print("生成完成！")
    print("="*60)
    print(f"成功: {success_count}/{len(scripts)}")
    print(f"失败: {fail_count}/{len(scripts)}")
    
    if failed_scripts:
        print("\n失败的脚本:")
        for script_name, description in failed_scripts:
            print(f"  - {script_name}: {description}")
    
    print(f"\n所有图片已保存到: figures_new/")
    print(f"所有结果已保存到: results_new/")
    print("="*60)

if __name__ == "__main__":
    main()
