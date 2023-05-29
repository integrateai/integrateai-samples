from integrate_ai_sdk.base_class import IaiBaseModule


class TemplateModel(IaiBaseModule):
    def __init__(self):
        """
        Here you should instantiate your model layers based on the configs.
        """
        super(TemplateModel, self).__init__()

    def forward(self):
        """
        The forward path of a model. Can take an input tensor and return a prediction tensor
        """
        pass


if __name__ == "__main__":
    template_model = TemplateModel()
