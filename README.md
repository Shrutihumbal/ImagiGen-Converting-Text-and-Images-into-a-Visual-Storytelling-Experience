# PRINCIPLES OF DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS (DC-GAN) AND ITS APPLICATION IN TEXT-IMAGE SYNTHESIS

This is a Tensorflow implementation of synthesizing images. The images are synthesized using the GAN-CLS Algorithm. This implementation is built on top of the excellent DCGAN in Tensorflow.
The architecture used is given below:

![image](https://github.com/Sayak007/Text-to-Image-Synthesis-using-DCGAN/blob/main/Images/img2.jpg)
Image Source: [Generative Adversarial Text-to-Image Synthesis Paper](https://arxiv.org/abs/1605.05396)

The main objective function used is:

![image](https://github.com/Sayak007/Text-to-Image-Synthesis-using-DCGAN/blob/main/Images/img1.jpg)<br /> 
G = Generator, D = Discriminator, Pd (x) = distribution of real data, P(z) = distribution of generator, 
x = sample from Pd (x), z = sample from P(z), D(x) = Discriminator network, G(z) = Generator network

## Datasets
We executed our algorithm on the Oxford-102 flower dataset. For the Oxford-102 dataset, it has 102 classes, which contains 82 training classes and 20 test classes. Each of the images in the dataset has 10 corresponding text descriptions. We have used the DC-GAN architecture for training the model on this dataset and we have done twice, once for 100 epochs with one set of captions and another for 200 epochs with a different set of captions. 
<br/>Source: [Oxford-102 flower dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## Software Specs
* Python 3 or above
* Anaconda/ Jupyter/ Google Colab
* TensorFlow 
* Tensor Layer
* scikit-learn : for skip thought vectors
* NLTK : for skip thought vectors
* Numpy 

## Hardware used
* Processor: AMD Ryzen 5 2400G
* Clock Speed: 3.6 GHz
* Graphics: RX Vega 11 Graphics
* RAM: 8GB DDR4
* RAM Speed: 2933 MHz
* Hard Drive: 1 TB SATA
* OS: Windows 10 pro
* Architecture: 64bit

## Parameters
* z_dim: Noise Dimension. Default is 100.
* t_dim: Text feature dimension. Default is 256.
* batch_size: Batch Size. Default is 64.
* image_size: Image dimension. Default is 64.
* gf_dim: Number of conv in the first layer generator. Default is 64.
* df_dim: Number of conv in the first layer discriminator. Default is 64.
* gfc_dim: Dimension of gen untis for for fully connected layer. Default is 1024.
* caption_vector_length: Length of the caption vector. Default is 1024.
* data_dir: Data Directory. Default is Data/.
* learning_rate: Learning Rate. Default is 0.0002.
* beta1: Momentum for adam update. Default is 0.5.
* epochs: Max number of epochs. Default is 100/200.
* resume_model: Resume training from a pretrained model path.
* data_set: Data Set to train on. Default is flowers.

## Algorithm
* Import Dataset
* Save the following  as pickles:
  * Caption Pickle – Merge captions with image in a vector format
  * Image Test, Train Pickle – Train-Test Split of the Pickle
  * vocab Pickle – Synthesizing of the words in the captions
* Load data from pickle
* Define main_train :
  * CNN encoding
  * Defining Loss functions
  * Loop until number of epochs:
    * Noise to Generator:
      * Batch Normalization
      * Leaky ReLU
      * Convolution
      * Transpose Convolution
    * Discriminator:
      * Inputs the real image pickle data
      * Inputs the generated samples from generator G(z)
      * Leaky ReLU
      * Batch Normalization
      * Convolutions(4x)
    * Decision Making Real or Fake:
      * If fake the discriminator output serves as a feedback input for the generator in the next epoch
    * Print the generated image for nth epoch, along with the Generator losses, discriminator losses and RNN-losses.
  * End loop

## Experimental Results:
A. Quantitative Results
After an extensive 24-hour training period spanning
approximately 450 epochs, the Image Generator Model 
achieved significant progress at a resolution of 64*64
pixels. The model, incorporating both Discriminator
and Generator networks crucial for GAN-based text
to-image generation, exhibited visible advancement by
the 438th epoch. At this point, the Generator loss stood
at 1.3284586668014526, and the Discriminator loss at
1.2313429117202759, indicating nearing equilibrium
between the networks. The diminishing generator loss
signifies enhanced capability in generating realistic
images from textual prompts. In contrast, the higher loss
of the discriminator, following the adversarial training
principle, underscores the efficacy of the network in
distinguishing genuine images from synthesized ones 
For the 128*128 resolution, training lasted 48 hours
across 1000 epochs, yielding a generator loss of
0.6933209896087646 and a discriminator loss of
1.3896995782852173. Analyzing these outcomes, it’s
evident that the model trained on 64*64 resolution
images exhibited faster convergence and lower loss
values compared to the model trained on 128*128
resolution images. This implies that the 64*64 resolution
model reached a more stable state in terms of image
generation within a shorter training period. Table2.
Experimental analysis on 102-flowers dataset with
128*128 resolution

B. Qualitative Results
The textual description accompanying the flower image
emphasizes the distinct visual attributes. Subsequently,
upon examining the test output result, it reveals a mul
titude of smaller images. These images are the gener
ated outputs produced by the model in response to the
provided textual description. The presence of numerous
smaller images suggests the model’s attempts to interpret
and generate various visual representations corresponding
to the textual description of a purple-coloured flower
with oval-shaped petals. This observation underscores
the model’s efforts in generating diverse outputs to en
capsulate the potential variations and nuances inherent
in the given textual input, aiming to produce a range
of plausible floral images that align with the described characteristics. Further evaluation and analysis of these
outputs would provide deeper insights into the model’s
ability to interpret textual descriptions

#### Result:
![image](https://github.com/Sayak007/Text-to-Image-Synthesis-using-DCGAN/blob/main/Result/train_200.png)

## Report
* https://drive.google.com/file/d/1Ej1CG2DD9E8114FHshV007oLOQTcIpDe/view?usp=sharing 

## References

[1] Liao, W., Hu, K., Yang, M. Y., and Rosenhahn, B.,”Text to image
generation with semantic-spatial aware gan,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pp. 18187-18196, 2022.
[2] Zhu, M., Pan, P., Chen, W., and Yang, Y., ”Dm-gan: Dynamic
memory generative adversarial networks for text-to-image
synthesis,”in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pp.5802-5810, 2019.
[3] Liu, X., Gong, C., Wu, L., Zhang, S., Su, H., and Liu,
Q, ”sedream: Training-free text-to-image generation with
improved clip + gan space optimization,”in arXiv preprint
arXiv:2112.01573, 2021.
[4] Ruan, S., Zhang, Y., Zhang, K., Fan, Y., Tang, F., Liu, Q., and
Chen E, ”Dae-gan: Dynamic aspect-aware gan for text-to-image
synthesis,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, pp. 13960-13969, 2021
[5] Wang, H., Lin, G., Hoi, S. C., and Miao, C., ”Cycle-consistent
inverse GAN for text-to-image synthesis,” in Proceedings of the
29th ACM International Conference on Multimedia, pp. 630-638,
2021
[6] Hang, C., and Peng, Y, ”Stacking VAE and GAN for context
aware text-to-image generation,” in 018 IEEE Fourth international
conference on multimedia big data (BigMM)(pp. 1-5), 2018.
[7] Tan H., Liu X., Yin B., and Li X,”DC-GAN: Distribution
regularization for text-to-image generation”, in Transactions
on Neural Networks and Learning Systems,Vol. 8, pp. 153113
153122, 2022.
[8] Mahajan, S., Nighrunkar, M., Kulkarni, A., Joshi, A., and
Sawant, S.,”Theft detection system using cGAN approach. In
AIP Conference Proceedings”, in AIP Publishing,Vol. 2745, No.
1, 2023.
[9] Tao M., Tang H., Wu F., Jing X. Y., Bao B. K., and Xu, C.,”Df
gan: A simple and effective baseline for text-to-image synthesis”,
in
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition,(pp. 16515-16525),2022.
[10] Ding, M., Zheng, W., Hong, W., and Tang, J.,”Cogview2: Faster
and better text-to-image generation via hierarchical transformers”,
in Neural Information Processing Systems,35,16890-16902,2022
