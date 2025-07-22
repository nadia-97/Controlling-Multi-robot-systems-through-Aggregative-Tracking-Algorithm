from setuptools import find_packages, setup
import glob

package_name = 'task2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nadia',
    maintainer_email='nadia@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task2_2 = task2.task2_2:main',
            'task2_3 = task2.task2_3:main',
            'task2_2_plotter = task2.task2_2_plotter:main',
            'task2_3_plotter = task2.task2_3_plotter:main',

        ],
    },
)
