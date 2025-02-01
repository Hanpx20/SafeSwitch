from setuptools import setup, find_packages

setup(
    name='internal_safety',
    version='1.0.0',
    author='Peixuan Han',
    author_email='kaola_fei@163.com',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    install_requires=[
    ],
    python_requires='>=3.6',
    include_package_data=True,
    keywords='NLP, safety',
    project_urls={},
    package_dir={"": "src"},
)
