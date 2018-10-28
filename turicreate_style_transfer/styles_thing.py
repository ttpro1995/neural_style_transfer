import turicreate as tc


model = tc.load_model("mymodel_pencil.model")

# Load some test images
test_images = tc.load_images('../contents/')

# Stylize the test images
stylized_images = model.stylize(test_images)

print(type(stylized_images))
print(stylized_images)

stylized_images['stylized_image'][1].save("test1.png")