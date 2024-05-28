import os
import numpy as np
import matplotlib.pyplot as plt

class ImageLoader:
    def __init__(self, db_images_file, predictions_file, query_file, images_directory):
        self.db_images = np.load(db_images_file)
        self.predictions = np.load(predictions_file)
        self.queries = np.load(query_file)
        self.images_directory = images_directory
    
    def display_image(self, index):
        """Display the image corresponding to the given index in the database."""
        image_path = os.path.join(self.images_directory, self.db_images[index])
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    def show_query_with_predictions(self, query_index):
        """Show the query image and its corresponding predictions."""
        query_image_path = os.path.join(self.images_directory, self.queries[query_index])
        query_image = plt.imread(query_image_path)
        query_predictions = self.predictions[query_index]

        fig, axes = plt.subplots(1, len(query_predictions) + 1, figsize=(15, 5))
        axes[0].imshow(query_image)
        axes[0].set_title(f"Query Image: {query_index}")
        axes[0].axis('off')


        for i, prediction_index in enumerate(query_predictions):
            prediction_image_path = os.path.join(self.images_directory, self.db_images[prediction_index])
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

    

# # Example usage:
# image_loader = ImageLoader('./datasets/SF_XS/sfxs_test_dbImages.npy', './predictions/test_set_20240528_201151.npy', './datasets/SF_XS/sfxs_test_qImages.npy', r'E:\MLDL24\Datasets\sf_xs')

# # Display an image
# image_loader.display_image(0)

# # Get the path of an image
# image_path = image_loader.get_image_path(0)
# print("Path of the image:", image_path)

# # Show a query with its predictions
# image_loader.show_query_with_predictions(0)

# # Show multiple queries with  predictions
# image_loader.show_multiple_queries_with_predictions([0, 1, 2])

# # Get predictions for a query
# query_predictions = image_loader.get_predictions_for_query(0)
# print("Predictions for query:", query_predictions)
