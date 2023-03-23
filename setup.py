from setuptools import setup, find_packages

setup(
    name="impulse_response_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'scikit-learn',
        'librosa',
        'matplotlib',
        'seaborn'
    ],
    entry_points={
        'console_scripts': [
            'train=impulse_response_classifier.train:main',
            'evaluate=impulse_response_classifier.evaluate:main',
            'standardize_data=impulse_response_classifier.standardize_data:main',
        ],
    },
)