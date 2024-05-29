import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ImageLoader:
    def __init__(self, db_images_file, predictions_file, query_file, images_directory, ground_truth_file=None):
        self.db_images = np.load(db_images_file)
        self.predictions = np.load(predictions_file)
        self.queries = np.load(query_file)
        self.images_directory = images_directory
        if ground_truth_file is not None:
            self.gt_images = np.load(ground_truth_file, allow_pickle=True)

    def display_image(self, index):
        """Display the image corresponding to the given index in the database."""
        image_path = os.path.join(self.images_directory, self.db_images[index])
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def show_query_with_predictions(self, query_index):
        """Show the query image and its corresponding predictions."""
        query_image_path = os.path.join(
            self.images_directory, self.queries[query_index])
        query_image = plt.imread(query_image_path)
        query_predictions = self.predictions[query_index]

        fig, axes = plt.subplots(
            1, len(query_predictions) + 1, figsize=(15, 5))
        axes[0].imshow(query_image)
        axes[0].set_title(f"Query Image: {query_index}")
        axes[0].axis('off')

        for i, prediction_index in enumerate(query_predictions):
            prediction_image_path = os.path.join(
                self.images_directory, self.db_images[prediction_index])
            if self.gt_images is not None:
                if prediction_index in self.gt_images[query_index]:
                    gt_color = 'green'
                else:
                    gt_color = 'red'
                rect = patches.Rectangle(
                    (0, 0), 1, 1, transform=axes[i+1].transAxes, color=gt_color, linewidth=2, fill=False)
                axes[i+1].add_patch(rect)

                prediction_image = plt.imread(prediction_image_path)
                axes[i+1].imshow(prediction_image)
                axes[i+1].set_title(f'Prediction {i+1}')
                axes[i+1].axis('off')

            else:
                prediction_image = plt.imread(prediction_image_path)
                axes[i+1].imshow(prediction_image)
                axes[i+1].set_title(f'Prediction {i+1}')
                axes[i+1].axis('off')

        plt.show()

    def get_image_path(self, index):
        """Get the path of the image corresponding to the given index in the database."""
        return self.db_images[index]

    def get_predictions_for_query(self, query_index):
        """Get the predictions for the given query index."""
        return self.predictions[query_index]

    def show_multiple_queries_with_predictions(self, query_indices):
        """Show multiple queries with their corresponding predictions."""
        for query_index in query_indices:
            self.show_query_with_predictions(query_index)

    def get_query_image(self, query_index):
        """Get the query image given its index."""
        return plt.imread(self.db_images[self.queries[query_index]])

    def save_results(self, output_file):
        """Save the predictions to a file."""
        np.save(output_file, self.predictions)
