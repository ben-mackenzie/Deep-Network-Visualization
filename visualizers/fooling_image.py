import torch
from torch.autograd import Variable

class FoolingImage:
    def make_fooling_image(self, X, target_y, model):
        """
        Generate a fooling image that is close to X, but that the model classifies
        as target_y.

        Inputs:
        - X: Input image; Tensor of shape (1, 3, 224, 224)
        - target_y: An integer in the range [0, 1000)
        - model: A pretrained CNN

        Returns:
        - X_fooling: An image that is close to X, but that is classifed as target_y
        by the model.
        """

        model.eval()

        # Initialize our fooling image to the input image, and wrap it in a Variable.
        X_fooling = X.clone()
        X_fooling_var = Variable(X_fooling, requires_grad=True)

        # We will fix these parameters for everyone so that there will be
        # comparable outputs

        learning_rate = 10  # learning rate is 1
        max_iter = 100  # maximum number of iterations

        for it in range(max_iter):

            ##############################################################################
            # TODO: Generate a fooling image X_fooling that the model will classify as   #
            # the class target_y. You should perform gradient ascent on the score of the #
            # target class, stopping when the model is fooled.                           #
            # When computing an update step, first normalize the gradient:               #
            #   dX = learning_rate * g / ||g||_2                                         #
            #                                                                            #
            # Inside of this loop, write the update rule.                                #
            #                                                                            #
            # HINT:                                                                      #
            # You can print your progress (current prediction and its confidence score)  #
            # over iterations to check your gradient ascent progress.                    #
            ##############################################################################
            scores = model(X_fooling_var)

            #target score
            scores_true = scores[:, target_y]
            scores_true.backward()
            grad = X_fooling_var.grad

            #grad ascent
            X_fooling_var.data += learning_rate * grad / torch.norm(grad)
            with torch.no_grad():
                grad = X_fooling_var.grad
                X_fooling_var += learning_rate * grad/ torch.norm(grad) #torch.norm(grad)

            #forward pass to see if it matches
            new_scores = model.forward(X_fooling_var)
            right_class = torch.argmax(new_scores, axis=1)
            if(right_class == target_y):
                break

            X_fooling_var.grad.zero_()
            if target_y == scores.data.max(1)[1][0]:
                break
            ##############################################################################
            #                             END OF YOUR CODE                               #
            ##############################################################################

        X_fooling = X_fooling_var.data

        return X_fooling
