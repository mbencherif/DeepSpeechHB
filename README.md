<div align="center">    
 
# Deepspeech ASR on EC2 Gaudi Processors
AWS Habana 
Deep Learning Challenge 

This work has been inspired from : https://github.com/jiwidi/DeepSpeech-pytorch.git
Many changes have been done and files added, to fit with EC2-Habana Gaudi processors.
## Description   
This is a first implementation on how to implement pytorch deepspeech ASR on EC2 DL1 instanes : Gaudi Processor Based instance
Please take a look at : https://youtu.be/snuXBMYjs5k 

Running On EC2-DL1 instance 
===========================
 
STEP1 :
-------

 To run the code you need to download the : https://github.com/HabanaAI/Model-References.git
inside a directory called /work3 (this directory will be mapped insde your docker later on).
Go to /work3/Model-References/PyTorch/examples/computer_vision 

STEP2 :
------- 

 run :
 git clone https://github.com/mbencherif/DeepSpeechHB.git

STEP3 :
-------

 From the DOCKS3 directory of this git.
  
Use the Dockerfile:
 -run make_docker3.sh
       It will do the build : docker build . --tag deepspeech_hb3:latest
STEP4:
--------
 -go one level up, run w3.sh : 
       It will do :
          docker run -it -v ~/work3:/work3 --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice  --net=host --ipc=host deepspeech_hb3:latest

There are two tasks done but with some errors to correct:
  1. python demo_single_dsphb.py 
   Toy example to port the pytorch deepspeech to habana Gaudi processors, with a tweaked torchaudio support.
   The torchaudio is not fully installed by only torch modules have been rearranged, so the example does not show torchaudio dependencies.
   A negative dmension error needs to be solved. 
 
 2. python mainhb.py 
 To test the app using pytorchlightning.



 Running On your local machine :
================================

If you want to test if the programs work on your local machine 
 pip install -r requirements.txt
 run : python main.py
 
  It will download the dev-clean and test-lean of Librispeech, and run for one iteration, 
 you can modify all the parameters, once the code is running correctly.
 
 Errors I bypassed:
 ==================
 I solved the torchaudio, issue as when installing a code depending on torchaudio, it will break the torch version, installed the Docker and the habana framework will not run.
 when you import torchaudio, it will check for : 
 synapse_logger INFO. pid=334 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cpp:340 Done command: restart
Traceback (most recent call last):
  File "train-hb.py", line 21, in <module>
    from project.model.deepspeech_main import DeepSpeech
  File "/work3/DeepSpeech-pytorch/project/model/deepspeech_main.py", line 13, in <module>
    import torchaudio
  File "/work3/DeepSpeech-pytorch/torchaudio/__init__.py", line 1, in <module>
    from torchaudio import _extension  # noqa: F401
  File "/work3/DeepSpeech-pytorch/torchaudio/_extension.py", line 27, in <module>
    _init_extension()
  File "/work3/DeepSpeech-pytorch/torchaudio/_extension.py", line 21, in _init_extension
    torch.ops.load_library(path)
  File "/usr/local/lib/python3.8/dist-packages/torch/_ops.py", line 110, in load_library
    ctypes.CDLL(path)
  File "/usr/lib/python3.8/ctypes/__init__.py", line 373, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libc10_cuda.so: cannot open shared object file: No such file or directory

 I took the basic fucntions of torchaudio and put them sepaately in another file, as they are based on torch, it is not an optimal solutioN, but it solved my needs - Reading a wavefile by changing to soundfile reader,
- Using Melspectrum , Freqauency masking and time masking transforms from trochaudio as simple as torch networks using nn.module.
 All the stuff is inside the tweaked_torchaudio.py. It is not a replacement for torchaudio, but just to avoid code breaking as python install torchaudio --no-deps did not function.
 
 Error with numpy==1.22:
 Librosa can not be used as it requires numba that requires numpy<1.21, while the torch on the habana framework is based on numpy==1.22, that is why i used soundfile for reading the flac file. it is working in a simple way.
 
                                                                        
 Errors on hold :
==================
oot@ip-172-31-94-254:/work3/DeepSpeech3# python mainhb.py 
Loading Habana modules from /usr/local/lib/python3.8/dist-packages/habana_frameworks/torch/lib
synapse_logger INFO. pid=473 at /home/jenkins/workspace/cdsoftwarebuilder/create-pytorch---bpt-d/repos/pytorch-integration/pytorch_helpers/synapse_logger/synapse_logger.cpp:340 Done command: restart
/work3/DeepSpeech3/tweaked_torchaudio.py:437: UserWarning: At least one mel filterbank has all zero values. The value for `n_mels` (128) may be set too high. Or, the value for `n_freqs` (201) may be set too low.
  warnings.warn(
Global seed set to 17
hmp:verbose_mode  False
hmp:opt_level O1
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: True, using: 1 HPUs
torch.Size([8, 3, -124, 403])
Traceback (most recent call last):
  File "mainhb.py", line 142, in <module>
    run_cli()
  File "mainhb.py", line 138, in run_cli
    main(args)
  File "mainhb.py", line 101, in main
    trainer.fit(model)
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py", line 749, in fit
    self._call_and_handle_interrupt(
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py", line 694, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py", line 786, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py", line 1214, in _run
    self._dispatch()
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py", line 1294, in _dispatch
    self.training_type_plugin.start_training(self)
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py", line 202, in start_training
    self._results = trainer.run_stage()
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py", line 1304, in run_stage
    return self._run_train()
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py", line 1321, in _run_train
    self._pre_training_routine()
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py", line 1316, in _pre_training_routine
    self.call_hook("on_pretrain_routine_start")
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py", line 1510, in call_hook
    callback_fx(*args, **kwargs)
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/callback_hook.py", line 148, in on_pretrain_routine_start
    callback.on_pretrain_routine_start(self, self.lightning_module)
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/callbacks/model_summary.py", line 56, in on_pretrain_routine_start
    model_summary = summarize(pl_module, max_depth=self._max_depth)
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/utilities/model_summary.py", line 481, in summarize
    model_summary = ModelSummary(lightning_module, max_depth=max_depth)
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/utilities/model_summary.py", line 211, in __init__
    self._layer_summary = self.summarize()
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/utilities/model_summary.py", line 268, in summarize
    self._forward_example_input()
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/utilities/model_summary.py", line 304, in _forward_example_input
    model(input_)
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/work3/DeepSpeech3/deepspeech_main.py", line 126, in forward
    x = x.reshape(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
RuntimeError: invalid shape dimension -372
 
 
 
 
 


