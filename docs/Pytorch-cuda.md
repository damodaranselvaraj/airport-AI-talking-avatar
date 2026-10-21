Understanding Ditto: All the Technologies Explained in Simple English

For your AI Talking Avatar – Airport Guide project, Ditto is the component that makes a person's face in a video appear to speak using your generated answer and audio.

To run Ditto on your RTX 4050 laptop GPU, you need to understand three things:

GPU technologies: CUDA, cuDNN, PyTorch, and TensorRT.
Supporting Python libraries: librosa, OpenCV, MediaPipe, ONNX Runtime, and others.
Model files and downloads: ONNX files, checkpoints, Hugging Face CLI, and related configuration files.

Let's understand each one, using simple examples.

1. PyTorch

What is it?

PyTorch is a Python framework used to build and run AI models.

Think of PyTorch as an engine that executes the mathematical operations required by an AI model.

For example, Ditto needs to process facial images and audio to generate a talking-avatar video. PyTorch can execute the neural network that performs this task.

Example in your project:

You provide:

A photograph of an airport information assistant.
An audio recording of the answer.

The PyTorch model processes these inputs and helps generate video frames corresponding to the speech.

Why do we need it?

Ditto provides a PyTorch inference path, so you can run the model without setting up the TensorRT inference path.

Remember:

PyTorch is the framework that runs the AI model.

2. Torch

What is it?

Torch is commonly used as shorthand for PyTorch in Python projects.

When you install PyTorch, you typically import it using:

import torch

For example, the torch package provides functionality to:

Load trained AI models.
Perform mathematical operations.
Use the GPU.
Process images and other data.
Run neural networks.

What is the difference between Torch and PyTorch?

Term	Meaning
PyTorch	The AI framework and its ecosystem.
torch	The Python package you import to use PyTorch.
torchvision	A related package containing computer vision datasets, transformations, and pretrained models.
torchaudio	A related package containing audio processing functionality.

Important: You don't need to install all three automatically for every project. Install the packages that Ditto actually requires.

3. CUDA

What is it?

CUDA is NVIDIA's platform for running computations on its GPUs.

Your laptop has an NVIDIA RTX 4050 GPU. CUDA allows compatible software to use that GPU for AI processing.

Without GPU acceleration, many AI operations would have to run on your CPU, which could be significantly slower.

Simple example

Imagine processing 1,000 images.

CPU: Can process the work, but may be slower for highly parallel AI operations.
GPU using CUDA: Can perform many suitable mathematical operations in parallel.

This is why CUDA is important for running AI models efficiently.

How does CUDA relate to PyTorch?

PyTorch can use CUDA to execute model computations on your NVIDIA GPU.

The relationship is:

PyTorch → CUDA-enabled operations → NVIDIA GPU

For your project, installing a CUDA-enabled PyTorch build is important if you want PyTorch to use the RTX 4050.

Important distinction: The CUDA version shown by nvidia-smi represents the maximum CUDA version supported by your installed NVIDIA driver, not necessarily the CUDA toolkit installed on your machine.

4. cuDNN

What is it?

cuDNN stands for CUDA Deep Neural Network library.

It is an NVIDIA library that provides optimized operations commonly used by deep learning models.

For example, neural networks frequently perform operations such as:

Convolutions.
Tensor calculations.
Other deep learning operations.

cuDNN provides optimized implementations of many of these operations for supported NVIDIA GPUs.

How does it relate to CUDA and PyTorch?

Think of the technologies as layers:

Technology	Responsibility
NVIDIA GPU	Performs the physical computations.
CUDA	Provides the platform for GPU computing.
cuDNN	Optimizes supported deep learning operations.
PyTorch	Executes the AI model and can use these GPU capabilities.
Ditto	Uses the model to generate talking-avatar video.

Do you need to install cuDNN separately?

Not necessarily.

Some PyTorch installations include or provide access to the required CUDA and cuDNN runtime libraries through their Python packages.

The correct installation depends on the PyTorch build and Ditto's requirements. Avoid installing an additional CUDA toolkit or cuDNN version without first checking whether it is necessary.

5. TensorRT

What is it?

TensorRT is NVIDIA's inference optimization technology.

Inference means using a trained AI model to produce an output.

For example:

Training a model means teaching it using data.
Inference means using the trained model to generate a result.

TensorRT optimizes supported models to execute efficiently on NVIDIA GPUs.

TensorRT vs. PyTorch

Ditto provides two inference paths in the setup you described.

Feature	PyTorch inference	TensorRT inference
Main purpose	Run the AI model	Run an optimized AI model
Setup	Generally simpler	Requires compatible TensorRT components and model files
Model format	Depends on Ditto's checkpoint and configuration	Often uses TensorRT engine files
GPU support	Supports NVIDIA GPUs through compatible CUDA builds	Designed to optimize inference on supported NVIDIA GPUs
Portability	Usually easier to configure across compatible environments	Engine compatibility can depend on the GPU and software environment
Best starting point for your project	Yes	Consider later if needed
Why does Ditto mention tensorrt==8.6.1?

That is a specific TensorRT version required by the relevant setup instructions.

If you choose the TensorRT path, you need to check its compatibility with your Python version, CUDA environment, and other dependencies.

If you choose the PyTorch path, you may be able to avoid installing TensorRT altogether.

However, you must first verify that Ditto's source code allows the PyTorch path to run without importing TensorRT unconditionally.

For your RTX 4050 laptop, I recommend starting with the PyTorch path because your initial goal is to get the avatar working, not to optimize inference speed.

6. Python Libraries Required by Ditto

These libraries support different parts of the talking-avatar pipeline.

A. librosa

Purpose: Audio analysis and processing.

Librosa is a Python library for analyzing audio.

It can help extract useful information from sound, including:

Audio features.
Spectrograms.
Frequency-related information.
Timing and signal characteristics.
How does it help Ditto?

Ditto uses speech audio to generate corresponding facial movements.

Audio processing can help prepare the sound or extract the features required by the model.

Simple example: The user asks about airport parking. Your LLM generates an answer, and a text-to-speech system converts it into audio. Audio-processing components prepare that audio for the avatar pipeline.

Librosa is one possible supporting component in that process, depending on Ditto's implementation.

B. OpenCV

Purpose: Image and video processing.

OpenCV is a computer vision library.

It provides functionality for:

Reading images.
Resizing images.
Processing video frames.
Manipulating image pixels.
Saving video output.
How does it help Ditto?

Suppose you provide a photograph of your virtual airport assistant.

OpenCV can help load and prepare that image for the model.

When Ditto generates facial frames, OpenCV can also support image processing and video-related operations.

Remember: OpenCV handles images and video; it does not independently generate realistic talking-avatar animation.

C. MediaPipe

Purpose: Detecting and locating important features in images.

MediaPipe provides tools for perception tasks, including face-related processing.

A face contains identifiable landmarks, such as:

Eye corners.
The nose.
Mouth corners.
Other facial reference points.

These landmarks help describe the geometry of a face.

How does it help Ditto?

Imagine providing a photograph of your airport assistant.

A face-processing pipeline needs to determine where the face is located and identify relevant facial regions.

MediaPipe's face-related functionality can provide landmark information used by supported pipelines.

Important: MediaPipe is not the talking-avatar generator itself. It is a supporting computer vision technology.

What is a MediaPipe Face Landmarker task?

A task is a packaged MediaPipe solution configured for a particular purpose.

The Face Landmarker task estimates facial landmarks from supported image, video, or live-stream inputs.

A .task file is a model asset used by that task.

Ditto's exact implementation determines which landmarks it needs and how they are used.

D. ONNX Runtime

Purpose: Running AI models saved in ONNX format.

ONNX stands for Open Neural Network Exchange.

It defines a common format for representing machine learning models.

ONNX Runtime is software that loads and executes models in this format.

How does it help Ditto?

Suppose Ditto uses separate models for particular supporting tasks, such as face detection.

Those models might be supplied as ONNX files.

ONNX Runtime can execute them without requiring you to run every model through PyTorch.

Depending on the installation, ONNX Runtime can execute models on the CPU or through supported GPU execution providers.

What is onnxruntime-gpu?

It is the GPU-enabled ONNX Runtime package for supported NVIDIA CUDA configurations.

The important distinction is:

Package	Purpose
onnxruntime	CPU-oriented ONNX Runtime package
onnxruntime-gpu	ONNX Runtime with supported GPU execution capabilities

For your project, GPU execution may improve performance, but it is not guaranteed for every model or operation.

You must also ensure that the installed version is compatible with your CUDA and cuDNN runtime requirements.

Important: Installing onnxruntime-gpu does not automatically mean every ONNX model will run on the GPU. The relevant execution provider must be available and selected.

E. cuda-python

Purpose: Accessing CUDA functionality through Python.

CUDA is primarily a GPU computing platform with APIs that software can access.

cuda-python provides Python bindings for CUDA functionality.

Bindings allow Python programs to interact with functionality that would otherwise be accessed through lower-level APIs.

Why might Ditto need it?

A project may use CUDA APIs directly for GPU operations, memory management, or other functionality.

Whether Ditto needs this package depends on its actual implementation and dependency requirements.

Important: cuda-python is not the same thing as PyTorch.

PyTorch is an AI framework.
CUDA is NVIDIA's GPU computing platform.
cuda-python provides Python access to CUDA APIs.

You should install the version required by Ditto rather than choosing an arbitrary latest version.

F. torchvision

Purpose: Computer vision support for PyTorch.

torchvision is part of the PyTorch ecosystem.

It provides functionality for:

Image transformations.
Image datasets.
Pretrained computer vision models.
Other computer vision utilities.
Does Ditto need it?

Not necessarily.

Some projects import torchvision directly; others do not.

It is often installed alongside PyTorch when a project needs computer vision functionality, but it should not be treated as mandatory without checking Ditto's dependencies.

If you install it, choose a version compatible with your PyTorch version.

G. torchaudio

Although it wasn't the main focus of your question, torchaudio was mentioned in your installation plan.

It provides audio-related functionality for PyTorch.

It can support audio loading and processing workflows.

However, if Ditto uses librosa or other audio libraries instead, torchaudio may not be necessary.

Again, check the actual project requirements before installing it.

7. What Are ONNX Files?

An ONNX file contains a representation of a machine learning model.

For example, a repository might contain:

face_detection.onnx
face_landmark.onnx
audio_model.onnx

These names are illustrative, not verified filenames from Ditto's repository.

Why might Ditto contain several ONNX files?

A complicated AI application may use multiple models, each responsible for a different task.

For example:

Model type	Possible responsibility
Face detection model	Locate the face in an image
Face landmark model	Identify facial reference points
Audio-related model	Process audio-related features
Main PyTorch model	Generate the talking-avatar output

The exact responsibilities depend on the files actually supplied by Ditto.

Does ONNX mean the model is trained differently?

No.

ONNX is primarily a model representation and exchange format. A trained model can be exported to ONNX so compatible runtimes can execute it.

Remember: ONNX is a model format, while ONNX Runtime is software that runs ONNX models.

8. What Is a Checkpoint?

A checkpoint contains saved model information, typically learned parameters and sometimes additional training or configuration state.

For inference, the checkpoint provides the trained parameters needed by the model.

Consider a simple analogy.

Imagine an AI model as a student.

The model architecture defines how the student is structured.
Training teaches the student.
The checkpoint preserves what the student has learned.
Inference uses that learned knowledge to produce an answer.

In Ditto, the main model checkpoint contains learned information needed to generate the talking-avatar output.

However, the checkpoint alone may not be enough.

The software may also need:

The model architecture.
A configuration file.
Auxiliary model files.
An audio input.
A reference image.
What is a HubERT checkpoint?

HuBERT is a pretrained speech representation model.

It learns useful representations from speech audio.

A speech-processing pipeline can use these representations as input features for subsequent models.

In your Ditto setup, a file described as a HuBERT checkpoint may provide the parameters for an audio-feature extraction component.

The precise role depends on the repository's implementation.

Important: Do not assume every file associated with HuBERT is interchangeable. A PyTorch checkpoint, an ONNX model, and a TensorRT engine can represent related model functionality in different formats.

9. What Is a Conda Environment?

A Conda environment is an isolated software environment.

It lets you install a particular Python version and a collection of compatible packages without unnecessarily changing other Python projects on your laptop.

Why does your Ditto project need one?

Your computer might already contain:

Another Python version.
Packages used by other projects.
Different versions of NumPy.
Other CUDA-related libraries.

Installing everything into your main Python environment could cause dependency conflicts.

Instead, you can create a dedicated environment for Ditto.

For example:

Component	Intended choice
Environment name	ditto
Python	3.10, if supported by the repository
PyTorch	A compatible version
CUDA support	Compatible with your NVIDIA driver
Additional packages	Versions compatible with Ditto

When you activate this environment, Python uses the packages installed there.

Remember: A Conda environment does not create a separate GPU or install an entirely separate NVIDIA driver.

It isolates software dependencies.

10. What Does torch 2.5.1 with cu124 Mean?

This notation identifies a PyTorch version and its associated CUDA runtime build.

torch 2.5.1 means PyTorch version 2.5.1.
cu124 indicates a build associated with CUDA 12.4.

This is different from installing the complete CUDA toolkit separately.

Why does the version matter?

AI packages depend on compatible software versions.

For example, the following components must work together:

Your NVIDIA GPU driver.
The PyTorch build.
The CUDA runtime libraries.
Any required cuDNN libraries.
Other GPU-dependent packages, including ONNX Runtime GPU.

Your RTX 4050 is an NVIDIA GPU, but that alone does not guarantee that every combination of these software versions will work.

My recommendation: Treat PyTorch 2.5.1 with CUDA 12.4 as a candidate configuration, not a confirmed Ditto requirement. First verify the repository's installation instructions and dependency constraints.

11. What Is Hugging Face?

Hugging Face is a platform that hosts machine learning models, datasets, and related files.

A repository hosted there can contain:

Model checkpoints.
ONNX models.
TensorRT engine files.
Configuration files.
Documentation.

If Ditto's model files are hosted on Hugging Face, you can download the files required to run the application.

Website: Hugging Face

What is huggingface-cli download?

It refers to a command-line method for downloading files from Hugging Face repositories.

A command-line interface, or CLI, lets you interact with software by entering commands in a terminal.

Instead of manually downloading each model file through a browser, the CLI can download selected files from a repository.

Why is this useful for Ditto?

Suppose the repository contains all of these:

PyTorch checkpoints.
TensorRT engines.
ONNX auxiliary models.
Configuration files.
Example assets.

You may not need all of them.

If you are using PyTorch inference, selectively downloading the required PyTorch checkpoints and supporting files can save disk space and bandwidth.

However, you must confirm the exact file dependencies before excluding files.

12. What Is Git LFS, and Why Avoid a Full Clone?

Git is a version-control system used to manage source code.

Git Large File Storage (Git LFS) is an extension that handles large files, such as model weights and other binary assets.

AI repositories frequently contain files that are much larger than ordinary Python source files.

What happens when you clone a repository?

A regular Git clone retrieves the repository's version-controlled contents. When Git LFS is configured, large-file downloads can also occur.

If the repository includes several model formats, a full clone may download large files you never use.

For example:

Repository contents	Required for a PyTorch-only setup?
Python source code	Yes
Required PyTorch checkpoints	Yes
Required ONNX auxiliary models	If Ditto uses them
TensorRT engines	Not necessarily
TensorRT-specific configuration	Depends on implementation
Example audio and images	Useful for testing

These are illustrative categories, not a verified inventory of the repository.

Why selectively download files?

Selective downloading can reduce:

Download time.
Disk usage.
Unnecessary model files.

But there is an important detail: you must obtain the Python source code as well as the model files.

Downloading checkpoints alone does not give you the complete application.

Also, a file named config.pkl should not automatically be assumed to contain only configuration values. Pickle files can contain Python objects, and loading untrusted pickle files can execute code. Download model assets only from a repository you trust.

13. How Everything Fits Together in Your Airport Guide

Let's connect these technologies to your project.

Imagine a passenger asks:

"Where can I find the baggage claim area?"

Your system retrieves the relevant answer from the fictional airport's documents.

An LLM generates the response, and a text-to-speech system converts it into audio.

Ditto then uses the audio and a reference image to generate a talking-avatar video.

The following diagram shows the conceptual flow.

Passenger asks a question
          |
          v
LLM / RAG system
          |
          v
Answer generated
          |
          v
Text-to-Speech
          |
          v
Speech audio + Avatar image
          |
          v
       DITTO
          |
          +---------------------+
          |                     |
          v                     v
    PyTorch model          Auxiliary models
          |                     |
          v                     v
    CUDA / GPU            ONNX Runtime
          |                     |
          +----------+----------+
                     |
                     v
             Talking-avatar video
                     |
                     v
             Airport kiosk display

This diagram is conceptual. Ditto's exact internal processing order and model dependencies must be confirmed from its implementation.

Where do the libraries fit?
Technology	Role in the airport kiosk
PyTorch	Runs the main model when using the PyTorch inference path
CUDA	Enables GPU computation on your NVIDIA GPU
cuDNN	Provides optimized deep learning operations where used
TensorRT	Alternative inference optimization path
librosa	Audio analysis and processing
OpenCV	Image and video processing
MediaPipe	Face-related perception and landmark detection
ONNX Runtime	Executes supporting ONNX models
cuda-python	Provides Python access to CUDA APIs
torchvision	Additional computer vision functionality when required
Hugging Face CLI	Downloads model files from a repository
Conda	Isolates the project's Python environment
14. My Recommendation for Your RTX 4050 Laptop

Since your goal is to build the AI Talking Avatar kiosk, I would approach the setup in this order.

Phase 1: Establish the environment
Confirm your NVIDIA driver is installed and working.
Create a dedicated Conda environment.
Install a Python version supported by Ditto.
Install a compatible PyTorch build.
Verify that PyTorch can detect and use your RTX 4050.
Phase 2: Download the required models
Locate the official Ditto repository.
Identify the supported PyTorch inference path.
Determine which checkpoints it requires.
Identify the auxiliary ONNX models and configuration files.
Download only the required assets.
Phase 3: Install supporting libraries

Install the compatible versions of the libraries Ditto actually imports.

Pay particular attention to:

NumPy compatibility.
MediaPipe requirements.
ONNX Runtime GPU compatibility.
PyTorch and CUDA runtime compatibility.
Any unconditional TensorRT imports.
Phase 4: Run a basic inference test

Before integrating the airport LLM, test Ditto independently using a reference image and a short audio recording.

Confirm that:

The model loads successfully.
The GPU is being used where expected.
The audio is processed correctly.
The output video is generated.
GPU memory usage remains within the available capacity.

Only after this works should you integrate Ditto with your RAG, LLM, and text-to-speech components.

One final caution: Your RTX 4050 laptop GPU may have approximately 6 GB of physical VRAM, with the usable amount varying by model and system configuration. You should measure actual memory usage during inference rather than assuming that a particular checkpoint will fit.

The PyTorch path may be easier to establish initially, but it is not guaranteed to use less VRAM or run faster than TensorRT.

The most important concepts to remember

If you remember just these six points, you will understand most of the installation plan:

PyTorch runs the main AI model.
CUDA enables GPU computation.
cuDNN optimizes supported deep learning operations.
TensorRT provides an alternative, optimized inference path.
ONNX Runtime runs models stored in ONNX format.
Hugging Face CLI helps download model files without necessarily downloading every large asset.

Everything else supports a particular part of the system.