from .colpali import load_model


class ModelManager:
    _instance = None
    model = None
    processor = None
    use_dummy_model = True

    @staticmethod
    def get_instance():
        if ModelManager._instance is None:
            ModelManager._instance = ModelManager()
            if not ModelManager.use_dummy_model:
                ModelManager._instance.initialize_model_and_processor()
        return ModelManager._instance

    def initialize_model_and_processor(self):
        if self.model is None or self.processor is None:  # Ensure no reinitialization
            self.model, self.processor = load_model()
            if self.model is None or self.processor is None:
                print("Failed to initialize model or processor at startup")
            else:
                print("Model and processor loaded at startup")
