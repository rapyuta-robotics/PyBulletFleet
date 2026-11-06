from glob import glob

from setuptools import setup

package_name = "pybullet_fleet_ros"

setup(
    name=package_name,
    version="0.1.3",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yu Okamoto",
    maintainer_email="yuokamoto1988@gmail.com",
    description="ROS 2 bridge for PyBulletFleet multi-robot simulation",
    license="Apache-2.0",
    tests_require=["pytest", "launch_testing"],
    entry_points={
        "console_scripts": [
            "bridge_node = pybullet_fleet_ros.bridge_node:main",
            "state_publisher = pybullet_fleet_ros.state_publisher_node:main",
            "fleet_nav_demo = pybullet_fleet_ros.fleet_nav_demo:main",
            "fleet_rviz = pybullet_fleet_ros.fleet_rviz:main",
        ],
    },
)
