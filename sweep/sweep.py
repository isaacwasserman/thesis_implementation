from GNEMS import *
import wandb
wandb.login()

representative_test_images = [
    "gt_2.png_1.1.01.tiff_1.1.04.tiff", # easy
    "gt_2.png_1.1.01.tiff_1.1.12.tiff", # easy
    "gt_2.png_1.1.02.tiff_1.1.05.tiff", # medium
    "gt_2.png_1.1.02.tiff_1.1.13.tiff", # medium
    "gt_2.png_1.1.03.tiff_1.1.02.tiff", # hard
    "gt_2.png_1.1.03.tiff_1.1.04.tiff"  # hard
]

def run_experiment(config=None):
    f1_scores = np.zeros(len(representative_test_images))
    for image_index, test_image_name in enumerate(representative_test_images):
        image = np.array(Image.open(f'test_images/{test_image_name}_image.png').resize(config.image_size, resample=Image.NEAREST)) / 255
        pixelwise_labels = np.array(Image.open(f'test_images/{test_image_name}_labels.png').resize(config.image_size, resample=Image.NEAREST)) > 0
        image = np.array(Image.fromarray(image[:-3, :]).resize(config.image_size, resample=Image.NEAREST))
        pixelwise_labels = np.array(Image.fromarray(pixelwise_labels[:-3, :]).resize(config.image_size, resample=Image.NEAREST))
        segmentor = GraphicallyGuidedEMSegmentor(d=config.d, n_filters=config.n_filters, dropout=config.dropout, lambda_=config.lambda_, size=config.image_size, lr=config.lr, iterations=config.iterations, subset_size=config.subset_size, prediction_stride=config.prediction_stride, seed=config.seed)
        segmentor.fit(image)
        segmentation = segmentor.predict()
        accuracy, report, conf_matrix = evaluate(pixelwise_labels, segmentation, plot=False)
        f1_scores[image_index] = report["weighted avg"]["f1-score"]
    return f1_scores.mean()

if __name__ == "__main__":
    wandb.init(project="Graphically Guided Neural EM for Unsupervised Image Segmentation")
    score = run_experiment(wandb.config)
    wandb.log({"Avg. F1 Score": score})