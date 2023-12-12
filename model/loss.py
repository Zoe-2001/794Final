import colorspacious as cs
from skimage.metrics import structural_similarity as ssim


def ssim_loss_component(predicted, target):
    return 1 - ssim(predicted, target, data_range=predicted.max() - predicted.min(), multichannel=True)

def lab_to_hsv(lab_image):
    # convert lab to rgb
    rgb_image = cs.cspace_convert(lab_image, start={'name': 'CIELab'}, end={'name': 'sRGB1'})

    # Convert rgb to hsv
    hsv_image = cs.cspace_convert(rgb_image, start={'name': 'sRGB1'}, end={'name': 'CIELab'})

    return hsv_image


# Define the HSV loss function
def hsv_loss(predicted_lab, target_lab, hue_weight=1.0, saturation_weight=1.0, value_weight=1.0):
    # Convert Lab to HSV
    predicted_hsv = lab_to_hsv(predicted_lab)
    target_hsv = lab_to_hsv(target_lab)

    # Extract HSV channels
    predicted_hue = predicted_hsv[..., 0]
    predicted_saturation = predicted_hsv[..., 1]
    predicted_value = predicted_hsv[..., 2]

    target_hue = target_hsv[..., 0]
    target_saturation = target_hsv[..., 1]
    target_value = target_hsv[..., 2]

    # Calculate HSV loss
    loss_hue = ssim_loss_component(predicted_hue, target_hue)
    loss_saturation = ssim_loss_component(predicted_saturation, target_saturation)
    loss_value = ssim_loss_component(predicted_value, target_value)

    # Combine all losses
    total_loss = hue_weight * loss_hue + saturation_weight * loss_saturation + value_weight * loss_value

    return total_loss


