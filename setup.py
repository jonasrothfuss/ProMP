from setuptools import setup

setup(name="maml_zoo",
      version='0.1',
      description='Framework that provides multiple Model Agnostic Meta-Learning (MAML) algorithms for reinforcement learning',
      url='https://github.com/jonasrothfuss/maml-zoo',
      author='Dennis Lee, Ignasi Clavera, Jonas Rothfuss',
      author_email='jonas.rothfuss@berkeley.edu',
      license='MIT',
      packages=['maml_zoo'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
        'joblib==0.12.2',
        'PyPrind',
        'numpy',
        'scipy',
        'gym==0.10.5',
        'python_dateutil',
        'tensorflow>=1.4.0'
      ],
      zip_safe=False)