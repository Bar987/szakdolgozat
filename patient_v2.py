class Patient:
	def __init__(self, pathology, gender, weight, height, images, image_idxs, contours):
		self.pathology = pathology 
		self.gender = gender
		self.weight = weight
		self.height = height
		self.images = images
		self.image_idxs = image_idxs
		self.contours = contours