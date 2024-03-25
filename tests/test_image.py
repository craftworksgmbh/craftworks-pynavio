import base64
from pathlib import Path
from io import BytesIO
from PIL import Image as PILImage

import numpy as np
import pytest

from pynavio.image import (_import_image, img_from_b64, img_to_b64,
                           imread, imwrite)


@pytest.fixture
def sample_image_paths(fixtures_path):
    """Fixture to return the paths of the sample image and output image."""
    sample_image_path = Path(fixtures_path, "Images/num_img.jpeg")
    sample_image_output_path = Path(fixtures_path, "Images/num_img_out.jpeg")
    return sample_image_path, sample_image_output_path


@pytest.fixture
def sample_image_array(sample_image_paths):
    """Fixture to load a sample image into a numpy array for testing."""
    Image = _import_image()
    with Image.open(sample_image_paths[0]) as img:
        return np.array(img).astype(float)


def test_imread(sample_image_paths):
    """Test reading an image file and encoding it to base64."""
    encoded_str = imread(sample_image_paths[0])
    assert isinstance(encoded_str, str)


def test_imwrite(sample_image_array, sample_image_paths):
    """Test writing a numpy array as an image file."""
    # Assuming sample_image_array is an RGB image
    imwrite(sample_image_paths[1], sample_image_array.astype(np.uint8))


def test_img_from_b64(sample_image_paths):
    """Test decoding a base64 string into a numpy array."""
    with open(sample_image_paths[0], "rb") as img_file:
        b64_str = base64.b64encode(img_file.read()).decode()

    img_array = img_from_b64(b64_str)
    assert isinstance(img_array, np.ndarray)


def test_img_to_b64(sample_image_array):
    """Test converting a numpy array (or PIL image) to a base64 string."""
    Image = _import_image()
    image = Image.fromarray(sample_image_array.astype('uint8'))
    encoded_str = img_to_b64(image, rgb=True)
    assert isinstance(encoded_str, str)


def test_encode_decode_image(sample_image_paths):
    """Test encoding an image as base64 and then decoding it back."""
    with open(sample_image_paths[0], "rb") as img_file:
        original_image_bytes = img_file.read()

    original_b64_str = base64.b64encode(original_image_bytes).decode()
    decoded_image_bytes = base64.b64decode(original_b64_str)
    decoded_image = PILImage.open(BytesIO(decoded_image_bytes))

    Image = _import_image()
    with Image.open(sample_image_paths[0]) as original_img:
        original_array = np.array(original_img)

    decoded_array = np.array(decoded_image)
    assert np.array_equal(original_array, decoded_array)
