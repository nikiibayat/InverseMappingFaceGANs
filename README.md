# Inverse Mapping of Face GANs
### Abstract
Generative adversarial networks (GANs) synthesize realistic images from a random latent vector. While many studies have explored various training configurations and architectures for GANs, the problem of inverting a generative model to extract latent vectors of given input images has been inadequately investigated. Although there is exactly one generated image per given random vector, the mapping from an image to its recovered latent vector can have more than one solution. We train a ResNet architecture to recover a latent vector for a given face that can be used to generate a face nearly identical to the target. We use a perceptual loss to embed face details in the recovered latent vector while maintaining visual quality using a pixel loss. The vast majority of studies on latent vector recovery perform well only on generated images, we argue that our method can be used to determine a mapping between real human faces and latent-space vectors that contain most of the important face style details. In addition, our proposed method projects generated faces to their latent-space with high fidelity and speed. At last, we demonstrate the performance of our approach on both real and generated faces.

### Paper: [arxiv](https://arxiv.org/pdf/2009.05671.pdf)

---
In order to generate fake faces from random latent vectors, we use a progressive growing GAN trained on 128x128 CelebA faces (Karras et al. 2017). The trained model can be downloaded from tensorflow hub. 
[Here is the link to download progan from tensorflow hub](https://tfhub.dev/google/progan-128/1)

The model recieves 512 size random vectors sampled from normal distribution and generates fake faces. You can run below command in order to generate 50 fake faces that include faces used in the paper.

```python generate_faces.py```

---
In this work, we first try to map face images into latent-space using a ResNet18 architecture. The proposed framework is represented in below figure.
![](https://github.com/nikiibayat/Inverse_Mapping_Face_GANs/blob/main/figures/Generated_ResNet.png?raw=true)

<div align="center">
The proposed framework to mapping generated faces.
</div>


In order to train the ResNet18, we generated 100k faces using progan (you can use generate_faces.py to do the same) to use as train dataset. ResNet18 is trained based on a combination of z-loss (The loss between features extracted from last layer of ResNet and ground truth z vectors) and perceptual loss between original generated faces and the reconstructed peers. Perceptual loss is computed based on MSE loss between features of all concatenation layers of FaceNet (Schroff, Kalenichenko, and Philbin 2015) for reconstructed and original faces. You must refer to the address of your 100k generated faces in PATH_TRAIN_generated. You should also save the corresponding z vectors for the training set in "z_vectors/generated_face_100k.npy"

- You can download FaceNet pre-trained on MS-Celeb-1M from [this link](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/).
- ResNet18 architecture can be downloaded from [this link](https://github.com/qubvel/classification_models).
- PATH_TRAIN_real is not required at this stage.
- Select a proper log and checkpoint directory.

In order to train ResNet, you can run below command:

```python resnet.py```


In order to recover latent vectors of generated faces, download our pretrained model from [This link](https://drive.google.com/drive/folders/1lk1Qf-kMO4e5AoBz62G5hVo4Qv37UoYR?usp=sharing) and save it in directory "saved_model".

Next, run ```python generate_faces.py``` to generate 50 fake faces.


Then run ```python resnet_test.py``` to save reconstructed faces in results folder.

---

![comparison](https://github.com/nikiibayat/Inverse_Mapping_Face_GANs/blob/main/figures/generated_pixel_perceptual.png?raw=true)

*The comparison of our method with optimization-based method (with and without stochastic clipping) is depicted in below figure. Original generated faces are presented in column(a). Column (b) is the generated faces of recovered latent vectors by (a) using gradient-descent method with 200 iterations. Column (c) applies stochastic clipping while updating gradient descent. Column (d) is our method, which utilizes our trained ResNet to map generated images to their corresponding latent vectors. Column (e) is our ResNet trained using both pixel and perceptual loss.*

In order to obtain results for optimization-based alternatives (Lipton and Tripathi 2017), you can run below command. You can comment stochastic clipping lines to replicate column (b).

```python gradient_based_recovery.py```

## Real faces
### Real faces dataset:
Select all VggFace2 (Cao et al. 2018) identities that have more than 100 samples. Then select 100 samples for each identity randomly. Next, select 100 random identities, 10000 Vggface2 images will remain. Align and crop all faces using MTCNN (Zhang et al. 2016). PATH_TRAIN_real is the directory that contains these images. VggFace2 dataset can be downloaded from [This link](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).

Now that you have your real faces dataset, you can train ResNet using ```python resnet.py``` All you have to do is to uncomment real faces section in fit function.
You can also use our pre-trained model from [this link](https://drive.google.com/drive/folders/1nZ7q2roWMXxNc4NPoNeIYNuLICW3EJWR?usp=sharing) to replicate same results as below figure. 

You must download AR dataset (Martinez1998) and align and crop them using MTCNN. AR dataset is used to evaluate the model in style transfer task and will replicate below figure. This dataset can be downloaded from [This link](https://www2.ece.ohio-state.edu/~aleix/ARdatabase.html).

Please make sure to change the model in resnet_test.py to the real faces model you just downloaded. Moreover, PATH_TEST should refer to AR faces now.

After downloading the model (the saved checkpoint) as well as AR dataset, the you can run ```python resnet_test.py``` to achieve below results.

![style](https://github.com/nikiibayat/Inverse_Mapping_Face_GANs/blob/main/figures/Style_transfer_AR.png?raw=true)

*The results for mapping natural faces to latent-space vectors that contain same style and facial features. The faces ineach row represent how recovered latent vectors understand the gender, hair style and emotions of the target image respectively.*

![pose](https://github.com/nikiibayat/Inverse_Mapping_Face_GANs/blob/main/figures/pose.png?raw=true)

*Recovered latent vectors preserve the pose of the target face.*

## References
[Karras et al. 2017] Karras, T.; Aila, T.; Laine, S.; and Lehti- nen, J. 2017. Progressive growing of gans for im- proved quality, stability, and variation. arXiv preprint arXiv:1710.10196.<br/>
[Zhang et al. 2016] Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10), 1499-1503.<br/>
[Lipton and Tripathi 2017] Lipton, Z. C., & Tripathi, S. (2017). Precise recovery of latent vectors from generative adversarial networks. arXiv preprint arXiv:1702.04782.<br/>
[Schroff, Kalenichenko, and Philbin 2015] Schroff, F.; Kalenichenko, D.; and Philbin, J. 2015. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, 815–823.<br/>
[Cao et al. 2018] Cao, Q.; Shen, L.; Xie, W.; Parkhi, O. M.; and Zisserman, A. 2018. Vggface2: A dataset for recog- nising faces across pose and age. In 2018 13th IEEE Inter- national Conference on Automatic Face & Gesture Recog- nition (FG 2018), 67–74. IEEE.<br/>
[Martinez1998] Martinez,A.M.1998.Thearfacedatabase. CVC Technical Report24.<br/>
