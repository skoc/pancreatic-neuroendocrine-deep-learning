###########################################
#             VISUALIZATION               #
###########################################
def color_to_np_color(color: str) -> np.ndarray:
    """
    Convert strings to NumPy colors.
    Args:
        color: The desired color as a string.
    Returns:
        The NumPy ndarray representation of the color.
    """
    colors = {
        "white": np.array([255, 255, 255]),
        "pink": np.array([255, 108, 180]),
        "black": np.array([0, 0, 0]),
        "red": np.array([255, 0, 0]),
        "purple": np.array([225, 225, 0]),
        "yellow": np.array([255, 255, 0]),
        "orange": np.array([255, 127, 80]),
        "blue": np.array([0, 0, 255]),
        "green": np.array([0, 255, 0])
    }
    return colors[color]


def add_predictions_to_image(
        xy_to_pred_class: Dict[Tuple[str, str], Tuple[str, float]],
        image: np.ndarray, prediction_to_color: Dict[str, np.ndarray],
        patch_size: int) -> np.ndarray:
    """
    Overlay the predicted dots (classes) on the WSI.
    Args:
        xy_to_pred_class: Dictionary mapping coordinates to predicted class along with the confidence.
        image: WSI to add predicted dots to.
        prediction_to_color: Dictionary mapping string color to NumPy ndarray color.
        patch_size: Size of the patches extracted from the WSI.
    Returns:
        The WSI with the predicted class dots overlaid.
    """
    for x, y in xy_to_pred_class.keys():
        prediction, __ = xy_to_pred_class[x, y]
        x = int(x)
        y = int(y)

        # Enlarge the dots so they are visible at larger scale.
        start = round((0.9 * patch_size) / 2)
        end = round((1.1 * patch_size) / 2)
        image[x + start:x + end, y + start:y +
              end, :] = prediction_to_color[prediction]

    return image


def get_xy_to_pred_class(window_prediction_folder: Path, img_name: str
                         ) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """
    Find the dictionary of predictions.
    Args:
        window_prediction_folder: Path to the folder containing a CSV file with the predicted classes.
        img_name: Name of the image to find the predicted classes for.
    Returns:
        A dictionary mapping image coordinates to the predicted class and the confidence of the prediction.
    """
    xy_to_pred_class = {}

    with window_prediction_folder.joinpath(img_name).with_suffix(".csv").open(
            mode="r") as csv_lines_open:
        csv_lines = csv_lines_open.readlines()[1:]

        predictions = [line[:-1].split(",") for line in csv_lines]
        for prediction in predictions:
            x = prediction[0]
            y = prediction[1]
            pred_class = prediction[2]
            confidence = float(prediction[3])
            # Implement thresholding.
            xy_to_pred_class[(x, y)] = (pred_class, confidence)
    return xy_to_pred_class


def visualize(wsi_folder: Path, preds_folder: Path, vis_folder: Path,
              classes: List[str], num_classes: int, colors: Tuple[str],
              patch_size: int) -> None:
    """
    Main function for visualization.
    Args:
        wsi_folder: Path to WSI.
        preds_folder: Path containing the predicted classes.
        vis_folder: Path to output the WSI with overlaid classes to.
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
        colors: Colors to use for visualization.
        patch_size: Size of the patches extracted from the WSI.
    """
    # Find list of WSI.
    whole_slides = get_all_image_paths(master_folder=wsi_folder)
    print(f"{len(whole_slides)} whole slides found from {wsi_folder}")
    prediction_to_color = {
        classes[i]: color_to_np_color(color=colors[i])
        for i in range(num_classes)
    }
    # Go over all of the WSI.
    for whole_slide in whole_slides:
        # Read in the image.
        whole_slide_numpy = imread(uri=whole_slide)[..., [0, 1, 2]]
        print(f"visualizing {whole_slide} "
              f"of shape {whole_slide_numpy.shape}")

        assert whole_slide_numpy.shape[
            2] == 3, f"Expected 3 channels while your image has {whole_slide_numpy.shape[2]} channels."

        # Save it.
        output_path = Path(
            f"{vis_folder.joinpath(whole_slide.name).with_suffix('')}"
            f"_predictions.jpg")

        # Confirm the output directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Temporary fix. Need not to make folders with no crops.
        try:
            # Add the predictions to the image and save it.
            imsave(uri=output_path,
                   im=add_predictions_to_image(
                       xy_to_pred_class=get_xy_to_pred_class(
                           window_prediction_folder=preds_folder,
                           img_name=whole_slide.name),
                       image=whole_slide_numpy,
                       prediction_to_color=prediction_to_color,
                       patch_size=patch_size))
        except FileNotFoundError:
            print(
                "WARNING: One of the image directories is empty. Skipping this directory"
            )
            continue

    print(f"find the visualizations in {vis_folder}")