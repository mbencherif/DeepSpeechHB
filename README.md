<div align="center">    
 
# Deepspeech ASR on EC2 Gaudi Processors
AWS Habana 
Deep Learning Challenge 
  
## Description   
This is a first implementation on how to implement pytorch deepspeech ASR on EC2 DL1 instanes : Gaudi Processor Based instance

On EC2 instance 
================
STEP1 :
--------
To run the code you need to Download the : https://github.com/HabanaAI/Model-References.git
inside a directory called /work3 
Go to /work3/Model-References/PyTorch/examples/computer_vision 
STEP2 :
-------- 
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


On your local machine :
If you want to test if the programs work pon your local machine 
 just type python main.py
  It will download the dev-clean and test-lean of Librispeech, and run for one iteration, 
 you can modify all the parameters, once the code is running correctly.


