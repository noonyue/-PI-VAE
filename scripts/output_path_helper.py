"""
输出路径辅助函数
用于支持将图片和结果保存到指定目录（通过环境变量）
"""
import os

def get_figures_dir():
    """获取图片输出目录"""
    return os.environ.get('FIGURES_DIR', 'figures')

def get_results_dir():
    """获取结果输出目录"""
    return os.environ.get('RESULTS_DIR', 'results')

def get_figure_path(filename):
    """获取图片文件的完整路径"""
    return os.path.join(get_figures_dir(), filename)

def get_result_path(filename):
    """获取结果文件的完整路径"""
    return os.path.join(get_results_dir(), filename)
