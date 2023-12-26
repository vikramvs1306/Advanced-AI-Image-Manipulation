Dependencies:

1.	!pip install -q git+https://github.com/huggingface/transformers.git
2.	!pip install torchvision
3.	!pip uninstall cublas_cu11
4.	!pip install transformers

Files and Folders:
1.	subSelectBirdImages.py
a.	run cmd: python3 subSelectBirdImages.py ./images/bird-1.jpeg ./images/bird-2.jpeg …and so on
b.	output: 5 selected filenames will be printed in the terminal

2.	subSelectSquirrelImages.py
a.	run cmd: python3 subSelectSquirrelImages.py ./images/squirrel-1.jpeg ./images/ squirrel -2.jpeg …and so on
b.	output: 5 selected filenames will be printed in the terminal

3.	segmentBirds.py
a.	run cmd: python3 segmentBirds.py ./images/squirrel-1.jpeg ./images/ squirrel -2.jpeg …and so on
b.	output: segmented and masked images will be put in folder “birdmask-images”

4.	segmentSequirrels.py
a.	run cmd: python3 segmentSequirrels.py ./images/squirrel-1.jpeg ./images/ squirrel -2.jpeg …and so on
b.	output: segmented and masked images will be put in folder “squirrelmask-images”


5.	replaceBirds.py
a.	run cmd: python3 replaceBirds.py
b.	output: bird replaced and masked images will be put in folder “birds-replaced”

6.	removeSquirrels.py
a.	run cmd: python3 removeSquirrels.py
b.	output: bird replaced and masked images will be put in folder “squirrels-removed”

7.	generateBirdFeederImagesFromText.py
a.	run cmd: python3 generateBirdFeederImagesFromText.py
b.	output: bird replaced and masked images will be put in folder “generated-images”
