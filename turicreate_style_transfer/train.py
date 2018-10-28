import turicreate as tc

# disable gpu
# tc.config.set_num_gpus(0)
# pip uninstall -y mxnet && pip install mxnet-cu90==1.1.0
# pip uninstall -y mxnet-cu90 && pip install mxnet-cu91==1.1.0  for cuda 90
# Load the style and content images
styles = tc.load_images('../styles/pencil-portrait-10.jpg')
content = tc.load_images('../contents/')

# Create a StyleTransfer model
model = tc.style_transfer.create(styles, content, batch_size=3, max_iterations=500)


# Save the model for later use in Turi Create
model.save('mymodel_pencil.model')
#
# # Export for use in Core ML
# model.export_coreml('MyStyleTransfer.mlmodel')

# Load some test images
test_images = tc.load_images('../contents/')

# Stylize the test images
stylized_images = model.stylize(test_images)

