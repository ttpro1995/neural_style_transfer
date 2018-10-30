Neural Style Transfer 

### Contents and Styles 
put content images in ./contents directory and style images in ./styles directory


### How to run Pytorch Neural Style project
```
cd pytorch_neural_style
```

```
CUDA_VISIBLE_DEVICES=1 python main.py \
--content /path/to/content/file.jpg \
--style /path/to/style/file.jpg \
--steps 1000 \
--output /path/to/outputfolder/  \
--save_every 20
```

Or run the shell script

```
# content and style file must be put in ./contents and ./styles
sh run_style_transfer content.jpg style.jpg 1
```