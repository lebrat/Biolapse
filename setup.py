from setuptools import setup

setup(
    name="Biolapse",
    version="1.0",
    author_email="leo.lebrat@gmail.com",
 	license="MIT",
    install_requires=[
			'imageio>=2.5.0',
			'Keras>=2.2.4',
			'Keras-Applications>=1.0.8',
			'Keras-Preprocessing>=1.1.0',
			'numpy>=1.17.0',
			'opencv-python>=4.1.0.25',
			'Pillow>=6.1.0',
			'PyQt5>=5.13.0',
			'PyQt5-sip>=4.19.18',
			'scikit-image>=0.15.0',
			'tensorboard>=1.14.0',
			'tensorflow>=1.14.0',
			'tensorflow-estimator>=1.14.0',
			'utils>=0.9.0',
      ])