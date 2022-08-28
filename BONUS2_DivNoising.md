DivNoising is one of the latest unsupervised denoising methods and follows a somewhat different approach. Instead of a U-Net, DivNoising employs the power of a Variational Auto-Encoder (VAE), but adds the use of a noise model to it in a suitable way.

A nice perk of this approach is that you will be able to sample diverse interpretations of a noisy image. Why is that useful? If the diverse samples look all the same or very similar to each other you can infer that the data is not very ambigious and you might decide to trust the result more. If on the other hand, the samples look quite different you know that you might not want to trust any of the denoised "interpretations". In the DivNoising paper you can also see how the diverse samples can be used in meaningful ways for imporved downstream analysis.

Since you've made it this far, you're clearly a pro so we will now take off the training wheels, essentially putting you in the position you would find yourself in when you come across a method you find interesting and want to check it out.

Clone the Div Noising repository from here: https://github.com/juglab/DivNoising, follow the setup instructions there and run through the Convallaria example.
