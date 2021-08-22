QuakeCity: Large scale synthetic dataset of physics-based earthquake damaged buildings in cities

##########################
Please cite references here https://sail.cive.uh.edu/quakecity/
V. Hoskere, “Developing autonomy in structural inspections through computer vision and graphics,” University of Illinois at Urbana-Champaign, 2021
V. Hoskere, Y. Narazaki, B. F. Spencer Jr., “3D Synthetic Environments with Physics-based Graphics Models for Vision-based Post-earthquake Condition Assessment”. Coming soon! 
Please check https://sail.cive.uh.edu/quakecity/ for updated list of references

##########################
DATASET DETAILS:

images: 1920x1080 png
labels: 1920x1080 png

train.csv: list of synthetic training images (buildings)
test.csv: list of synthetic test images (buildings)

FORMAT:
images in train.csv and corresponding annotations have the same file name

Directory structure:
- image	(3684 train + 1004 test = 4688 total)
	A10001.png
	A10002.png
	...
- labels (3684 train per label = 3684 x 6 = 22104 total)
	- crack
		A10001.png
		...
	- spall
		A10001.png
		...
	- rebar
		A10001.png
		...
	- ds
		A10001.png
		...
	- component
		A10001.png
		...
	- depth
		A10001.png
		...

images in the list in test.csv are in the same folder "image" above, but they don't have correspondances in labels

#############################
ANNOTATIONS:

Reading annotations: 
The annotations are saved as single channel indexed png images with a color palette. To read the annotations as a single channel indexed image, use of a package like PIL is recommended. Recommended usage:

	from PIL import Image
	annotation = Image.open("/path/to/label/annotation/")

Please also note that if you use a package like cv2 to read the image, the image will be read as a 3-channel rgb with palette colors instead of the annotation index. 

Format:
	Annotation:
		annotation index - annotation class name - color palette
		
		Evaluation

	Component:
		0 - wall - (202,150,150)
		1 - beam - (198,186,100)
		2 - column - (167,183,186)
		3 - window frame - (255,255,133)
		4 - window pane - (192,192,206)
		5 - balcony - (32,80,160)
		6 - slab - (193,134,1)
		100 - ignore - (70,70,70)
	
		Evaluation: 
			Pixels with index >= 7 will not be used for evaluation.
			
	Damage:
		Crack:
			0 - no cracks - (0,0,0)
			1 - cracks - (255,0,0)
		Spall:
			0 - no spalls - (0,0,0)
			1 - spalls - (255, 192, 203)
		Rebar:
			0 - no exposed rebar - (0, 0, 0)
			1 - exposed rebar - (255, 255, 50)
		
		Evaluation:
			IoU of pixels labelled as 1 will be calculated for evaluation. Other pixels will be discarded.
			
	Damage state:
		0 - no damage - (0,255,0)
		1 - light damage - (150,250,0)
		2 - moderate damage - (255,225,50)
		3 - severe damage - (255,0,0)
		100 - ignore - (128,128,128)
		
		Evaluation: 
			Pixels with index >= 4 will not be used for evaluation.		

Other annotations:
	Depth
		Integer values (uint8). Linearly scaled depth between 1 m and 50m.
		0 corresponds <= 1m, and (2^8-1) corresponds to >= 50m