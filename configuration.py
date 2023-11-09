class ArgValus:
    def __init__(self):
        self.dataset_name = "USPS"
        self.data_dimension = [16, 16]

        # k in knn algorithm
        self.k = 10

        # self.band_with = None

        self.num_of_classes = 0

        self.sample_per_class = 100

        self.train_percent = 0.01

        self.num_iter = 100
