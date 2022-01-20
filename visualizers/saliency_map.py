import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

from image_utils import preprocess
class SaliencyMap:
    def compute_saliency_maps(self, X, y, model):
        """
        Compute a class saliency map using the model for images X and labels y.

        Input:
        - X: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the saliency map.

        Returns:
        - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
        images.
        """
        # Make sure the model is in "test" mode
        model.eval()

        # Wrap the input tensors in Variables
        X_var = Variable(X, requires_grad=True)
        y_var = Variable(y, requires_grad=False)
        saliency = None

        ##############################################################################
        # TODO: Implement this function. Perform a forward and backward pass through #
        # the model to compute the gradient of the correct class score with respect  #
        # to each input image. You first want to compute the loss over the correct   #
        # scores, and then compute the gradients with a backward pass.               #
        ##############################################################################
        # forward pass
        scores = model(X_var)

        # Get the score of the correct class, scores are [5] Tensor
        scores = scores.gather(1, y_var.view(-1, 1)).squeeze()

        #Reverse calculation, a series of gradient calculations from the output score to the input image
        scores.backward(torch.FloatTensor([1.0,1.0,1.0,1.0,1.0])) # The parameter is the gradient initialization of the corresponding length
        # scores.backward() must have parameters, because the scores at this time are non-scalar, a vector of 5 elements

        # Get the correct score corresponding to the gradient of the input image pixel point
        saliency = X_var.grad.data

        saliency = saliency.abs() # Take the absolute value
        saliency, i = torch.max(saliency,dim=1)  # Take the value of the channel with the largest absolute value from the 3 color channels
        saliency = saliency.squeeze() # Remove 1 dimension

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return saliency

    def show_saliency_maps(self, X, y, class_names, model):
        # Convert X and y from numpy arrays to Torch Tensors
        X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
        y_tensor = torch.LongTensor(y)

        # Compute saliency maps for images in X
        saliency = self.compute_saliency_maps(X_tensor, y_tensor, model)
        # Convert the saliency map from Torch Tensor to numpy array and show images
        # and saliency maps together.
        saliency = saliency.numpy()

        N = X.shape[0]
        for i in range(N):
            plt.subplot(2, N, i + 1)
            plt.imshow(X[i])
            plt.axis('off')
            plt.title(class_names[y[i]])
            plt.subplot(2, N, N + i + 1)
            plt.imshow(saliency[i], cmap=plt.cm.gray)
            plt.axis('off')
            plt.gcf().set_size_inches(12, 5)
        plt.savefig('visualization/saliency_map.png')
        plt.show()
