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

    # def display_image(self, index):
    #     """Display the image corresponding to the given index in the database."""
    #     image_path = os.path.join(self.images_directory, self.db_images[index])
    #     image = plt.imread(image_path)
    #     plt.imshow(image)
    #     plt.axis('off')

    def display_image(self, index):
        """Display the image corresponding to the given index in the database."""
        image_path = os.path.join(self.images_directory, self.db_images[index])
        image = plt.imread(image_path)
        plt.title(f"Image {index}")
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def display_query_image(self, index):
        """Display the image corresponding to the given index in the database."""
        image_path = os.path.join(self.images_directory, self.queries[index])
        image = plt.imread(image_path)
        plt.title(f"Image {index}")
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def display_random_images(self, n=4):
        """Display random images from the database as a table."""
        num_images = min(n, len(self.db_images))
        num_rows = 2
        num_cols = 3
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(6, 4))
        axes = axes.flatten()
        for i in range(num_images):
            index = np.random.randint(0, len(self.db_images))
            image_path = os.path.join(
                self.images_directory, self.db_images[index])
            image = plt.imread(image_path)
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(f"Image {index}")
        plt.tight_layout()
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

    def show_all_queries_with_predictions(self, query_indices):
        """Show specified query images and their corresponding predictions in a single figure."""
        num_queries = len(query_indices)
        # Including the query image
        num_predictions = max(
            len(self.predictions[i]) for i in query_indices) + 1

        fig, axes = plt.subplots(num_queries, num_predictions, figsize=(
            num_predictions * 5, num_queries * 5))

        for row, query_index in enumerate(query_indices):
            query_image_path = os.path.join(
                self.images_directory, self.queries[query_index])
            query_image = plt.imread(query_image_path)
            query_predictions = self.predictions[query_index]

            axes[row, 0].add_patch(patches.Rectangle(
                (0, 0), 1, 1, transform=axes[row, 0].transAxes, color='blue', linewidth=6, fill=False))
            axes[row, 0].imshow(query_image)
            if row == 0:
                axes[row, 0].set_title(f"Queries ", weight='bold')
            axes[row, 0].axis('off')

            for col, prediction_index in enumerate(query_predictions):
                prediction_image_path = os.path.join(
                    self.images_directory, self.db_images[prediction_index])
                prediction_image = plt.imread(prediction_image_path)

                if self.gt_images is not None and prediction_index in self.gt_images[query_index]:
                    gt_color = 'green'
                else:
                    gt_color = 'red'

                rect = patches.Rectangle(
                    (0, 0), 1, 1, transform=axes[row, col+1].transAxes, color=gt_color, linewidth=6, fill=False)  # Thicker patch
                axes[row, col+1].add_patch(rect)
                axes[row, col+1].imshow(prediction_image)
                if row == 0:
                    axes[row, col +
                         1].set_title(f'Prediction {col+1}', weight='bold')
                axes[row, col+1].axis('off')

            # Hide unused subplots
            for j in range(len(query_predictions) + 1, num_predictions):
                axes[row, j].axis('off')

        # Adjust layout to fit the title
        plt.tight_layout()
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

    def show_ground_truth(self, query_indices):
        """Show the ground truth for the given query index."""
        num_queries = len(query_indices)
        # Including the query image
        num_gts = max(
            len(self.gt_images[i]) for i in query_indices) + 1

        fig, axes = plt.subplots(num_queries, num_gts, figsize=(
            num_gts * 5, num_queries * 5))

        for row, query_index in enumerate(query_indices):
            query_image_path = os.path.join(
                self.images_directory, self.queries[query_index])
            query_image = plt.imread(query_image_path)
            query_gt_images = self.gt_images[query_index]

            axes[row, 0].add_patch(patches.Rectangle(
                (0, 0), 1, 1, transform=axes[row, 0].transAxes, color='blue', linewidth=6, fill=False))
            axes[row, 0].imshow(query_image)
            if row == 0:
                axes[row, 0].set_title(f"Queries ", weight='bold')
            axes[row, 0].axis('off')

            for col, prediction_index in enumerate(query_gt_images):
                prediction_image_path = os.path.join(
                    self.images_directory, self.db_images[prediction_index])
                prediction_image = plt.imread(prediction_image_path)

                if self.gt_images is not None and prediction_index in self.gt_images[query_index]:
                    gt_color = 'green'
                else:
                    gt_color = 'red'

                rect = patches.Rectangle(
                    (0, 0), 1, 1, transform=axes[row, col+1].transAxes, color=gt_color, linewidth=6, fill=False)  # Thicker patch
                axes[row, col+1].add_patch(rect)
                axes[row, col+1].imshow(prediction_image)
                if row == 0:
                    axes[row, col +
                         1].set_title(f'Prediction {col+1}', weight='bold')
                axes[row, col+1].axis('off')

            # Hide unused subplots
            for j in range(len(query_gt_images) + 1, num_gts):
                axes[row, j].axis('off')

        # Adjust layout to fit the title
        plt.tight_layout()
        plt.show()

    def print_gt_stats(self):
        ct = 0
        for x in range(6):
            gtx = [1 if len(self.gt_images[i]) ==
                   x else 0 for i in range(len(self.gt_images))]
            ct += sum(gtx)
            print(
                f"Number of queries with {x} ground truth images: {sum(gtx)}")
        print(f"Total number of queries with less than 5: {ct}")
