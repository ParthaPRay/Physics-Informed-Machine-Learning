# Physics-Informed-Neural-Networks
This repo contains for Physics Informed Neural Networks



# Physics Informed Neural Network Basics




* Physics Informed Deep Learning: Data-driven Solutions and Discovery of Nonlinear Partial Differential Equations

  https://maziarraissi.github.io/PINNs/

  https://github.com/maziarraissi/PINNs


* [Artificial Neural Networks for Solving Ordinary and Partial Differential Equations]https://www.cs.uoi.gr/~lagaris/papers/TNN-LLF.pdf), Lagaris etal, IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 9, NO. 5, SEPTEMBER 1998

* [Physics Informed Deep Learning (Part I): Data-driven, Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/pdf/1711.10561.pdf)

* [Raissi et al, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations], (https://doi.org/10.1016/j.jcp.2018.10.045)

* [Scientific Machine Learning Through Physics–Informed Neural Networks: Where we are and What’s Next](https://link.springer.com/article/10.1007/s10915-022-01939-z)

* Tutorials on deep learning, Python, and dissipative particle dynamics (https://github.com/lululxvi/tutorials/tree/master)

  Deep Learning

  2022-07-22, [Deep Learning for Scientists and Engineers: Introduction](https://github.com/lululxvi/tutorials/blob/master/20220722_sciml/sciml_introduction.pdf)
    2021-12-10, [Physics-informed neural network (PINN)](https://github.com/lululxvi/tutorials/blob/master/20211210_pinn/pinn.pdf) [Video in Chinese](http://tianyuan.xmu.edu.cn/cn/minicourses/637.html)
    2021-12-08, [Deep learning: Software installation & Code implementation](https://github.com/lululxvi/tutorials/blob/master/20211208_deep_learning_handson/deep_learning_handson.pdf) [Video in Chinese](http://tianyuan.xmu.edu.cn/cn/minicourses/637.html)
    2021-12-08, [Deep learning: Concepts & Algorithms](https://github.com/lululxvi/tutorials/blob/master/20211208_deep_learning/deep_learning.pdf) [Video in Chinese](http://tianyuan.xmu.edu.cn/cn/minicourses/637.html)
    2020-05-22, [Double-descent phenomenon in deep learning](https://github.com/lululxvi/tutorials/blob/master/20200522_double_descent/double_descent.pdf)
    2019-08-16, [AutoML: Automated machine learning](https://github.com/lululxvi/tutorials/blob/master/20190816_autoML/autoML.pdf)
    2019-08-16, [Improving simple models with confidence profiles](https://github.com/lululxvi/tutorials/blob/master/20190816_ProfWeight/ProfWeight.pdf)
    2019-01-25, [Introduction to residual neural network (ResNet)](https://github.com/lululxvi/tutorials/blob/master/20190125_ResNet/ResNet.pdf)
    2018-08-03, [Neural networks 101: Implementing feedforward neural nets using TensorFlow](https://github.com/lululxvi/tutorials/blob/master/20180803_neural_network/nn_tutorial.pdf)
    2018-02-16, [4 years of generative adversarial networks (GANs)](https://github.com/lululxvi/tutorials/blob/master/20180216_GAN/gan.pdf)

Python
    [2018-01-24, Python 101: From zero to hero](https://github.com/lululxvi/tutorials/blob/master/20180124_Python/python_tutorial.pdf)

Dissipative Particle Dynamics (DPD)
    2015-10-08, [Scaling in DPD](https://github.com/lululxvi/tutorials/blob/master/20151008_DPD_scaling/DPD_scaling.pdf)


* DAS-PINNs

  A deep adaptive sampling method for solving high-dimensional partial differential equations.
  
  [TensorFlow implementation for DAS-PINNs: A deep adaptive sampling method for solving high-dimensional partial differential equations](https://arxiv.org/abs/2112.14038)

  https://github.com/MJfadeaway/DAS


* **Tutorial Series**
  
  * Physics Informed Machine Learning: High level overview of AI and ML in science and engineering
    
    https://youtu.be/JoFW2uSd3Uo?si=YjMDbddPnz5_hPNj
  
  
  * AI/ML + Physics Part 1: Choosing what to model [Physics Informed Machine Learning]
  
    https://youtu.be/ARMk955pGbg?si=YrV_fUGNAEDnqlgx
  
  * AI/ML + Physics Part 2: Curating training data [Physics Informed Machine Learning]
  
    https://youtu.be/g-S0m2zcKUg?si=ugt7yfpRTCz0oar8
  
  * AI/ML + Physics Part 3: Designing an architecture [Physics Informed Machine Learning]
  
    https://youtu.be/fiX8c-4K0-Q?si=8nlOz5C_WqtQXasI





# Physics Informed Neural Network Tools 

* IDRLnet

  IDRLnet, a Python toolbox for modeling and solving problems through Physics-Informed Neural Network (PINN) systematically.

  https://github.com/idrl-lab/idrlnet


  
* PINA

  PINA is an open-source Python library providing an intuitive interface for solving differential equations using PINNs, NOs or both together. Based on PyTorch and PyTorchLightning, PINA offers a simple and intuitive way to formalize a specific (differential) problem and solve it using neural networks . The approximated solution of a differential equation can be implemented using PINA in a few lines of code thanks to the intuitive and user-friendly interface.
  
  https://mathlab.github.io/PINA/

  https://github.com/mathLab/PINA


* PINN

  Simple PyTorch Implementation of Physics Informed Neural Network (PINN)

  https://github.com/nanditadoloi/PINN


* FBPINNs

  Finite basis physics-informed neural networks (FBPINNs). Solve forward and inverse problems related to partial differential equations using finite basis physics-informed neural networks (FBPINNs)

   ![FBPINN](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/b436a113-6e9b-446c-ab9b-4dc54db7001c)
   ![PINN](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/ef7643c6-da72-4c13-a46b-3192e099d563)
   ![test-loss](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/f23f4561-be17-410b-ad14-04e2de0e2c8c)

    FBPINN vs PINN solving the high-frequency 1D harmonic oscillator

   Physics-informed neural networks (PINNs) are a popular approach for solving forward and inverse problems related to PDEs
    - However, PINNs often struggle to solve problems with high frequencies and/or multi-scale solutions
    - This is due to the spectral bias of neural networks and the heavily increasing complexity of the PINN optimisation problem
    - FBPINNs improve the performance of PINNs in this regime by combining them with domain decomposition, individual subdomain normalisation and flexible subdomain training schedules
    - Empirically, FBPINNs significantly outperform PINNs (in terms of accuracy and computational efficiency) when solving problems with high frequencies and multi-scale solution



   https://github.com/benmoseley/FBPINNs




* Simulai

   ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/af318d02-c58f-4708-a339-f7501ea2b55b)

  A toolkit with data-driven pipelines for physics-informed machine learning. An extensible Python package with data-driven pipelines for physics-informed machine learning.

  https://github.com/IBM/simulai



* deepxde


  A library for scientific machine learning and physics-informed learning

  DeepXDE is a library for scientific machine learning and physics-informed learning. DeepXDE includes the following algorithms:

   ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/9e5a6933-02d9-45f7-8086-88e4ce950308)



    ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/987bd82f-c0ce-4691-a0a8-ae8c56e4a126)

    ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/97c99b52-df7d-4612-aed4-5c8c2fccd7df)

     ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/82892319-86e2-4151-ab9d-00eb158010aa)


 DeepXDE has implemented many algorithms as shown above and supports many features:

    - enables the user code to be compact, resembling closely the mathematical formulation.
    - complex domain geometries without tyranny mesh generation. The primitive geometries are interval, triangle, rectangle, polygon, disk, ellipse, star-shaped, cuboid, sphere, hypercube, and hypersphere. Other geometries can be constructed as constructive solid geometry (CSG) using three boolean operations: union, difference, and intersection. DeepXDE also supports a geometry represented by a point cloud.
    - 5 types of boundary conditions (BCs): Dirichlet, Neumann, Robin, periodic, and a general BC, which can be defined on an arbitrary domain or on a point set; and approximate distance functions for hard constraints.
    - 3 automatic differentiation (AD) methods to compute derivatives: reverse mode (i.e., backpropagation), forward mode, and zero coordinate shift (ZCS).
    different neural networks: fully connected neural network (FNN), stacked FNN, residual neural network, (spatio-temporal) multi-scale Fourier feature networks, etc.
    many sampling methods: uniform, pseudorandom, Latin hypercube sampling, Halton sequence, Hammersley sequence, and Sobol sequence. The training points can keep the same during training or be resampled (adaptively) every certain iterations.
    - 4 function spaces: power series, Chebyshev polynomial, Gaussian random field (1D/2D).
    - data-parallel training on multiple GPUs.
    - different optimizers: Adam, L-BFGS, etc.
    - conveniently save the model during training, and load a trained model.
    - callbacks to monitor the internal states and statistics of the model during training: early stopping, etc.
    - uncertainty quantification using dropout.
    - float16, float32, and float64.
    - many other useful features: different (weighted) losses, learning rate schedules, metrics, etc.

  https://deepxde.readthedocs.io/
  
  https://github.com/lululxvi/deepxde


* neurodiffeq
  
  A library for solving differential equations using neural networks based on PyTorch, used by multiple research groups around the world, including at Harvard IACS. The neurodiffeq is a package for solving differential equations with neural networks. Differential equations are equations that relate some function with its derivatives. They emerge in various scientific and engineering domains. Traditionally these problems can be solved by numerical methods (e.g. finite difference, finite element). While these methods are effective and adequate, their expressibility is limited by their function representation. It would be interesting if we can compute solutions for differential equations that are continuous and differentiable.

  https://github.com/NeuroDiffGym/neurodiffeq


* TensorDiffEq


   ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/70695e9f-1b92-4f31-99e8-43b8bb60d6e6)

  Efficient and Scalable Physics-Informed Deep Learning and Scientific Machine Learning on top of Tensorflow for multi-worker distributed computing

  What makes TensorDiffEq different?

    - Completely open-source
    
    - Self-Adaptive Solvers for forward and inverse problems, leading to increased accuracy of the solution and stability in training, resulting in less overall training time
    
    - Multi-GPU distributed training for large or fine-grain spatio-temporal domains
    
    - Built on top of Tensorflow 2.0 for increased support in new functionality exclusive to recent TF releases, such as XLA support, autograph for efficent graph-building, and grappler support for graph optimization* - with no chance of the source code being sunset in a further Tensorflow version release
    
    - Intuitive interface - defining domains, BCs, ICs, and strong-form PDEs in "plain english"
 
  Use TensorDiffEq if you require:

    - A meshless PINN solver that can distribute over multiple workers (GPUs) for forward problems (inference) and inverse problems (discovery)
    - Scalable domains - Iterated solver construction allows for N-D spatio-temporal support
    - support for N-D spatial domains with no time element is included
    - Self-Adaptive Collocation methods for forward and inverse PINNs
    - Intuitive user interface allowing for explicit definitions of variable domains, boundary conditions, initial conditions, and strong-form PDEs

  http://docs.tensordiffeq.io/
  
  https://github.com/tensordiffeq/TensorDiffEq


* PINNS-Torch

  PINNs-Torch, Physics-informed Neural Networks (PINNs) implemented in PyTorch.

   ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/c9b89012-ae92-4292-80ea-8b38e6a2f8d7)


  Our package introduces Physics-Informed Neural Networks (PINNs) implemented using PyTorch. The standout feature is the incorporation of CUDA Graphs and JIT Compilers (TorchScript) for compiling models, resulting in significant performance gains up to 9x compared to the original TensorFlow v1 implementation
 

   ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/84a34808-c561-4da2-b66c-ba0c9c5427e0)
  Each subplot corresponds to a problem, with its iteration count displayed at the top. The logarithmic x-axis shows the speed-up factor w.r.t the original code in TensorFlow v1, and the y-axis illustrates the mean relative error.
  

  https://github.com/rezaakb/pinns-torch


* AttnPINN-for-RUL-Estimation

  A Framework for Remaining Useful Life Prediction Based on Self-Attention and Physics-Informed Neural Networks
  
  https://github.com/XinyuanLiao/AttnPINN-for-RUL-Estimation


* Hippynn

  It is a python library for atomistic machine learning. The hippynn python package - a modular library for atomistic machine learning with pytorch.

  https://lanl.github.io/hippynn/

  https://github.com/lanl/hippynn



* GravNN

  Repository for Gravity Field Modeling and Recovery using Machine Learning Methods.

  This repository contains the GravNN python package whose purpose is to train Physics-Informed Neural Networks Gravity Models (PINN-GMs). The package itself contains the tensorflow models, physics constraints, hyperparameter configurations, data generators, and visualization tools used in training such models.

  The Physics-Informed Neural Network Gravity Model (PINN-GM) aims to solve many of the limitations of past gravity representations including the ubiquitous spherical harmonic and polyhedral gravity models (used for planetary and small-body exploration respectively).
  

  https://github.com/MartinAstro/GravNN

* PINNS-TF2
  
  PINNs-TF2, Physics-informed Neural Networks (PINNs) implemented in TensorFlow V2.


   ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/43b3aaf0-5648-4f12-b191-01bcc97a316e)

  https://github.com/rezaakb/pinns-tf2



* LPINNs

 To address some of the failure modes in training of physics informed neural networks, a Lagrangian architecture is designed to conform to the direction of travel of information in convection-diffusion equations, i.e., method of characteristic; The repository includes a pytorch implementation of PINN and proposed LPINN with periodic boundary condition.

 https://github.com/rmojgani/LPINNs


* PINN-FWI

  PINN-FWI: performing physics-informed neural network for FWI

  In this repository, I implemented the physics-informed neural network for full-waveform inversion. The networks is an autoencoder based on Dhara and Sen (2022) with some modifications. The architecture of their study is shown in the following figure.
  ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/fd674146-0c68-42f3-8ec2-bf736e458239)

  
  https://github.com/AmirMardan/pinn_fwi

* EP-PINNs

  EP-PINNs implementation for 1D and 2D forward and inverse solvers for the Aliev-Panfilov cardiac electrophysiology model. Also includes Matlab finite-differences solver for data generation. Implementation of Physics-Informed Neural Networks (PINNs) for the solution of electrophysiology (EP) problems in forward and inverse mode. 

https://github.com/martavarela/EP-PINNs


* pinns_tf2

  Yet another PINN implementation

  Tensoflow 2 implementation of Physics-Informed Neural Networks (PINNs) based on Physics Informed Deep Learning (Part I): [Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/pdf/1711.10561). Impleneted equations are

    - Heat,
    - Wave,
    - Reaction-Diffusion,
    - Stationary Advection-Diffusion,
    - Poisson,
    - Schrodinger's,
    - Burger's,
    - Klein Gordon.
    - and Transport.

  https://github.com/nimahsn/pinns_tf2



* pinn4hcf


  Physics-informed neural networks for highly compressible flows 🧠🌊

  https://repository.tudelft.nl/islandora/object/uuid:6fd86786-153e-4c98-b4e2-8fa36f90eb2a
  
  https://github.com/wagenaartje/pinn4hcf


* BOPINN


  Bayesian optimized physics-informed neural network for parameter estimation

  https://github.com/mahindrautela/BOPINN


* porch

  A PyTorch library for Physics-Informed Neural Networks (PINNs). Porch is a Python library that enables users to train Physics-Informed Neural Networks (PINNs) using the PyTorch framework. PINNs combine the power of neural networks with the governing laws of physics, allowing for the solution of complex physical problems with limited or noisy data.

  https://github.com/leiterrl/porch


* PINN-fokker-planck

  Physics Informed Neural Networks for Fokker-Planck. TensorFlow PINN study for a couple of Fokker-Planck equations.

  https://github.com/ElHuaco/PINN-fokker-planck


* deepsysid

  A system identification toolkit for multistep prediction using deep learning and hybrid methods. System identification toolkit for multistep prediction using deep learning and hybrid methods.

  https://github.com/AlexandraBaier/deepsysid


  
* Adaptive-optimization-of-PINN

  Optimizing Physics-Informed NN using Multi-task Likelihood Loss Balance Algorithm and Adaptive Activation Function Algorithm. This repository is used to reproduce PINN's inverse problem code about Burgers equation, plus multi-task likelihood loss balance algorithm and adaptive activation function algorithm.

  These frameworks and algorithms are derived from the following papers:

      - [Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
      
      - [Cipolla R, Gal Y, Kendall A. Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics[C]. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)
      
      - [Jagtap A D, Kawaguchi K, Karniadakis G E. Adaptive activation functions accelerate convergence in deep and Physics-informed neural Networks[J]. Journal of Computational Physics, 2020, 404: 109136.](https://www.sciencedirect.com/science/article/pii/S0021999119308411)

  https://github.com/XinyuanLiao/Adaptive-optimization-of-PINN
  

* Adaptive self-supervision for PINNs

  This repository contains PyTorch code to run adaptive resampling strategies for the self-supervision loss term in physics informed neural networks. The self-supervision data (collocation) points are resampled using the gradient of the loss and scheduled using a cosine scheduler that balances adaptivity with uniform sampling at user-defined intervals during training. For more details, see the [article](https://arxiv.org/abs/2207.04084)

 ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/fb7db6dc-769c-4496-af44-5bf4b350db33)

  https://github.com/ShashankSubramanian/adaptive-selfsupervision-pinns



* PararealML

  PararealML is a differential equation solver library featuring a [Parareal](https://en.wikipedia.org/wiki/Parareal) framework based on a unified interface for initial value problems and various solvers including a number of machine learning accelerated ones. The library's main purpose is to provide a toolset to investigate the properties of the Parareal algorithm, especially when using machine learning accelerated coarse operators, across various problems. The library's implementation of the Parareal algorithm uses [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) via [mpi4py](https://mpi4py.readthedocs.io/en/stable/).

  ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/2155ff37-7c74-4cbc-b3e2-309193b8e0dd)


 https://github.com/ViktorC/PararealML


* neuraloperator

  Learning in infinite dimension with neural operators. neuraloperator is a comprehensive library for learning neural operators in PyTorch. It is the official implementation for Fourier Neural Operators and Tensorized Neural Operators.

Unlike regular neural networks, neural operators enable learning mapping between function spaces, and this library provides all of the tools to do so on your own data.

NeuralOperators are also resolution invariant, so your trained operator can be applied on data of any resolution.

  https://neuraloperator.github.io/neuraloperator/dev/index.html

  https://github.com/neuraloperator/neuraloperator


* Burger-PINN

  A Physics-Informed Neural Network to solving Burgers' equation.

 Reference: [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)
 Author's Github page: [https://maziarraissi.github.io/PINNs/](https://maziarraissi.github.io/PINNs/)

  Check out other PINN project: [heat-pinn](https://github.com/314arhaam/heat-pinn)

  https://github.com/314arhaam/burger-pinn


* Heat-PINN
  
  A Physics-Informed Neural Network to solve 2D steady-state heat equation. A Physics-Informed Neural Network, to solve 2D steady-state heat equation, based on the methodology introduced in: Physics Informed Deep Learning (Part I): [Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)

  https://github.com/314arhaam/heat-pinn


* NES

   [Neural Eikonal Solver](https://github.com/sgrubas/NES): framework for modeling traveltimes via solving eikonal equation using neural networks. Neural Eikonal Solver (NES) is framework for solving factored eikonal equation using physics-informed neural network, for details see our paper: [early arXiv version](https://arxiv.org/abs/2205.07989) and [published final version](https://doi.org/10.1016/j.jcp.2022.111789). NES can simulate traveltimes of seismic waves in complex inhomogeneous velocity models.

   https://github.com/sgrubas/NES

    https://sgrubas.github.io/NES/



* neuraloperator

  Learning in infinite dimension with neural operators. neuraloperator is a comprehensive library for learning neural operators in PyTorch. It is the official implementation for Fourier Neural Operators and Tensorized Neural Operators.

Unlike regular neural networks, neural operators enable learning mapping between function spaces, and this library provides all of the tools to do so on your own data.

NeuralOperators are also resolution invariant, so your trained operator can be applied on data of any resolution.

  https://neuraloperator.github.io/neuraloperator/dev/index.html

  https://github.com/neuraloperator/neuraloperator
  

* ScIANN

  SciANN is a high-level artificial neural networks API, written in Python using Keras and TensorFlow backends. It is developed with a focus on enabling fast experimentation with different networks architectures and with emphasis on scientific computations, physics informed deep learing, and inversion. Being able to start deep-learning in a very few lines of code is key to doing good research.

  Use SciANN if you need a deep learning library that:

    - Allows for easy and fast prototyping.
    - Allows the use of complex deep neural networks.
    - Takes advantage TensorFlow and Keras features including seamlessly running on CPU and GPU.
 
  For more details, check out [review paper] at (https://arxiv.org/abs/2005.08803) and the [documentation] at (https://sciann.com/).

  https://github.com/sciann/sciann




* SciMLoutput
  
  Tutorials for doing scientific machine learning (SciML) and high-performance differential equation solving with open source software.

  SciMLTutorials.jl holds PDFs, webpages, and interactive Jupyter notebooks showing how to utilize the software in the [SciML Scientific Machine Learning ecosystem] (https://sciml.ai/). This set of tutorials was made to complement the documentation and the devdocs by providing practical examples of the concepts. For more details, please consult the docs.

  https://docs.sciml.ai/SciMLTutorialsOutput/stable/

  https://github.com/SciML/SciMLTutorialsOutput


* SciML

  SciML: Open Source Software for Scientific Machine Learning

  SciML contains a litany of modules for automating the process of model discovery and fitting. Tools like DiffEqParamEstim.jl and DiffEqBayes.jl provide classical maximum likelihood and Bayesian estimation for differential equation based models, while SciMLSensitivity.jl enables the deep hand-optimized methods for forward and adjoint senstivity (i.e. derivatives) of equation solvers. This enables the training of embedded neural networks inside of differential equations (neural differential equations or universal differential equations) for discovering unknown dynamical equations.

  Advanced Equation Solvers
  
The library DifferentialEquations.jl is a library for solving ordinary differential equations (ODEs), stochastic differential equations (SDEs), delay differential equations (DDEs), differential-algebraic equations (DAEs), and hybrid differential equations which include multi-scale models and mixtures with agent-based simulations. Other solvers like NonlinearSolve.jl for f(x)=0 rootfinding problems, Optimization.jl for nonlinear optimization, etc. expand SciML to more equation types, offering high features and performance while integrating with machine learning frameworks through differentiability. 

  https://sciml.ai/

  https://github.com/SciML/
# Physics Informed Neural Networks Apps

* Physics-Informed Deep Learning and its Application in Computational Solid and Fluid Mechanics
  
  This repository is dedicated to provide users of interest with the ability to solve forward and inverse hydrodynamic shock-tube problems and plane stress linear elasticity boundary value problems using Physics-Informed Deep Learning (PIDL) techniques (W-PINNs-DE & W-PINNs). This repository contains PINNs code from each problem in Physics-Informed Deep Learning and its Application in Computational Solid and Fluid Mechanics (Papados, 2021)

  https://github.com/alexpapados/Physics-Informed-Deep-Learning-Solid-and-Fluid-Mechanics


* Physics informed neural network in JAX

  https://github.com/ASEM000/Physics-informed-neural-network-in-JAX

* NVIDIA Modulus Core (Latest Release)

  Open-source deep-learning framework for building, training, and fine-tuning deep learning models using state-of-the-art Physics-ML methods

   [Base module that consists of the core components of the framework for developing Physics-ML models]. (https://docs.nvidia.com/deeplearning/modulus/modulus-core/index.html)
  
   ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/f81d84c4-21d3-44b3-b390-8e9fef511def)

   https://docs.nvidia.com/modulus/index.html

   Modulus is an open source deep-learning framework for building, training, and fine-tuning deep learning models using state-of-the-art Physics-ML methods.

Whether you are exploring the use of Neural operators like Fourier Neural Operators or interested in Physics informed Neural Networks or a hybrid approach in between, Modulus provides you with the optimized stack that will enable you to train your models at real world scale.

This package is the core module that provides the core algorithms, network architectures and utilities that cover a broad spectrum of physics-constrained and data-driven workflows to suit the diversity of use cases in the science and engineering disciplines.



    https://github.com/NVIDIA/modulus

    https://developer.nvidia.com/modulus


    Modulus Packages
    
      - Modulus (Beta): Open-source deep-learning framework for building, training, and fine-tuning deep learning models using state-of-the-art Physics-ML methods.
      - [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym): Framework providing pythonic APIs, algorithms and utilities to be used with Modulus core to physics inform model training as well as higher level abstraction for domain experts.
      
      Domain Specific Packages
       - [Earth-2 MIP (Beta)](https://github.com/NVIDIA/earth2mip): Python framework to enable climate researchers and scientists to explore and experiment with AI models for weather and climate.



 * NVIDIA Modulus Sym (Latest Release)

   NVIDIA Modulus Sym is a deep learning framework that blends the power of physics and partial differential equations (PDEs) with AI to build more robust models for better analysis.

  There is a plethora of ways in which ML/NN models can be applied for physics-based systems. These can depend based on the availability of observational data and the extent of understanding of underlying physics. Based on these aspects, the ML/NN based methodologies can be broadly classified into forward (physics-driven), data-driven and hybrid approaches that involve both the physics and data assimilation.

   ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/922d0aa1-18ca-4465-8db2-b3fb6db6a2de)



    https://docs.nvidia.com/deeplearning/modulus/modulus-sym/index.html

    


* GPT-PINN

  Generative Pre-Trained Physics-Informed Neural Networks Implementation.

  Generative Pre-Trained Physics-Informed Neural Networks toward non-intrusive Meta-learning of parametric PDEs

   ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/f3fa372f-7b92-4c4a-ad6e-492ac21cba58)

   Paper: https://arxiv.org/abs/2303.14878
   Talk: https://www.youtube.com/embed/KWaWH7xeVEg


    https://github.com/skoohy/GPT-PINN


  

# Resources 

* Steve Brunton, https://youtube.com/@Eigensteve?si=M1MmT4WQsXqa2v-h
* https://github.com/topics/physics-informed-neural-networks
* https://github.com/openhackathons-org/End-to-End-AI-for-Science

   ![image](https://github.com/ParthaPRay/Physics-Informed-Neural-Networks/assets/1689639/595ad8bb-6c15-4d6f-bfa6-9b7a70a1a3e0)

* https://github.com/pkestene/MS-HPC-AI-GPU
* https://github.com/MartinuzziFrancesco/awesome-scientific-machine-learning








