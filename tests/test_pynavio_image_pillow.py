import base64
import numpy as np
import pytest
from pynavio.image import imread, imwrite, img_from_b64, img_to_b64, _import_image

# Sample image path for reading and writing tests
SAMPLE_IMAGE_PATH = '/Users/clarareolid/Documents/cw_projects/Navio/repos/craftworks-pynavio-private/tests/test_pynavio/fixtures/Images/num_img.jpeg'
SAMPLE_IMAGE_PATH_OUTPUT = '/Users/clarareolid/Documents/cw_projects/Navio/repos/craftworks-pynavio-private/tests/test_pynavio/fixtures/Images/num_img_out.jpeg'


@pytest.fixture
def sample_image_array():
    """Fixture to load a sample image into a numpy array for testing."""
    Image = _import_image()
    with Image.open(SAMPLE_IMAGE_PATH) as img:
        return np.array(img).astype(float)


def test_imread():
    """Test reading an image file and encoding it to base64."""
    encoded_str = imread(SAMPLE_IMAGE_PATH)
    assert isinstance(encoded_str, str)
    # Further checks can include decoding and comparing to original file's bytes,
    # but this requires reading the original file again.


def test_imwrite(sample_image_array):
    """Test writing a numpy array as an image file."""
    # Assuming sample_image_array is an RGB image
    imwrite(SAMPLE_IMAGE_PATH_OUTPUT, sample_image_array.astype(np.uint8))
    # Verify file exists or reopen and check content matches expected image content.
    # This might require reading the image back and comparing arrays.


def test_img_from_b64():
    """Test decoding a base64 string into a numpy array."""
    with open(SAMPLE_IMAGE_PATH, "rb") as img_file:
        b64_str = base64.b64encode(img_file.read()).decode()

    img_array = img_from_b64(b64_str)
    assert isinstance(img_array, np.ndarray)
    # Additional checks can be made on the properties of the array (shape, dtype, etc.)


def test_img_to_b64(sample_image_array):
    """Test converting a numpy array (or PIL image) to a base64 string."""
    Image = _import_image()
    image = Image.fromarray(sample_image_array.astype('uint8'))
    encoded_str = img_to_b64(image, rgb=True)
    assert isinstance(encoded_str, str)
    # You can decode and compare to the original image, but this might require
    # converting the decoded data back into an image and comparing pixels.

