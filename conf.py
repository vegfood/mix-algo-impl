# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))  # 解释：`../..` 表示项目根目录
sys.path.insert(0, os.path.abspath('/home/gr-arm-7xx2/gr3_arm_ros/catkin_ws/src/gr3_pri'))

project = 'gr3_pri'
copyright = '2025, xx'
author = 'xx'
release = 'tianjin_project_union'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # 从代码提取文档
    'sphinx.ext.viewcode', # 添加源代码链接
    'sphinx.ext.napoleon', # 支持 Google 和 NumPy 风格的文档注释
]
autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = ['../../dist/*']

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
print("Exclude patterns:", exclude_patterns)