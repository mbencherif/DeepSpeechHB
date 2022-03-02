docker run -it -v ~/work3:/work3 --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice  --net=host --ipc=host deepspeech_hb3:latest

