<h2>TensorFlow-FlexUNet-Image-Segmentation-Ultrasound-Liver (2026/02/10)</h2>
Sarah T.  Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Ultrasound-Liver</b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass), 
and 
<a href="https://drive.google.com/file/d/1jLu40yhhN3wyke6qW7Y0bOURuLTD339U/view?usp=sharing">
<b>Augmented-Ultrasound-Liver-ImageMask-Dataset.zip</b></a> with colorized masks, which was derived by us from <br><br>
<a href="https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset">
<b>Annotated Ultrasound Liver images Dataset</b> </a> on the kaggle.com.
<br><br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of images and masks of the original <b>Ultrasound-Liver </b> dataset,
we used our offline augmentation tool <a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a> (please see also: 
<a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a>)
 to generate our Augmented Ultrasound-Liver dataset.
<br><br> 
<hr>
<b>Actual Image Segmentation for Ultrasound-Liver Images </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<b>rgb_map = {Benign:green, Malignant:red}</b>
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/images/10151.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/masks/10151.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test_output/10151.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/images/10275.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/masks/10275.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test_output/10275.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/images/10268.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/masks/10268.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test_output/10268.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset">
<b>Annotated Ultrasound Liver images Dataset</b>. </a>
<br><br>
The following explanation was taken from the above kaggle web site.
<br><br>
<b>About Dataset</b><br>
This dataset contains a collection of annotated ultrasound images of the liver, designed to aid in the development of 
computer vision models for liver analysis, segmentation, and disease detection. <br>
The annotations include outlines of the liver and liver mass regions, as well as classifications into benign, malignant, and normal cases.
<br><br>
<b>Creators</b><br>
Xu Yiming, Zheng Bowen, Liu Xiaohong, Wu Tao, Ju Jinxiu, Wang Shijie, Lian Yufan, Zhang Hongjun,<br>
 Liang Tong, Sang Ye, Jiang Rui, Wang Guangyu, Ren Jie, Chen Ting
 <br><br>
Published: November 2, 2022<br>
Version: v1<br>
DOI: 10.5281/zenodo.7272660<br>
<br>
<b>Dataset Overview</b><br>
This dataset provides ultrasound images of the liver with detailed annotations. <br>
The annotations highlight the liver itself and any liver mass regions present. The images are categorized into three classes:
<ul>
<li>Benign: Images showing benign liver conditions.</li>
<li>Malignant: Images showing malignant liver conditions.</li>
<li>Normal: Images of healthy livers.</li>
</ul>
<br>
<b>Annotations</b><br>
The ultrasound images have been annotated to show:<br>
<ul>
<li>Outlines of the liver.</li>
<li>Regions of liver masses (where applicable).</li>
</ul>
<br>
<b>Copyright and Citation</b><br>
This dataset is subject to copyright. Any use of the data must include appropriate acknowledgement and credit.
 Please contact the authors of the published data and cite the publication and the provided URL.
 <br><br>
<b>Citations</b><br>
Xu Yiming, Zheng Bowen, Liu Xiaohong, Wu Tao, Ju Jinxiu, Wang Shijie, Lian Yufan, Zhang Hongjun, <br>
Liang Tong, Sang Ye, Jiang Rui, Wang Guangyu, Ren Jie, & Chen Ting. (2022).<br>
 Annotated Ultrasound Liver images [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7272660<br>
<br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/">
Attribution 4.0 International (CC BY 4.0)</a>
<br>
<br>
<h3>
2 Ultrasound-Liver ImageMask Dataset
</h3>
 If you would like to train this Ultrasound-Liver Segmentation model by yourself,
please down load our dataset <a href="https://drive.google.com/file/d/1jLu40yhhN3wyke6qW7Y0bOURuLTD339U/view?usp=sharing">
<b>Augmented-Ultrasound-Liver-ImageMask-Dataset.zip</b>
</a> on the google drive,
expand the downloaded, and put it under <b>./dataset/</b> to be.
<pre>
./dataset
└─Ultrasound-Liver
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Ultrasound-Liver Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Ultrasound-Liver/Ultrasound-Liver_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Ultrasound-Liver TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Ultrasound-Liver/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Ultrasound-Liver and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 3
base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Ultrasound-Liver 1+2 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Ultrasound-Liver 1+2
;Ultrasound-Liver Benign:green, Malignant:red
rgb_map = {(0,0,0):0, (0,255,0):1, (255,0,0):2}         
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middle-point (23,24,25)</b><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (48,49,50)</b><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was terminated at epoch 50.<br><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/asset/train_console_output_at_epoch50.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Ultrasound-Liver/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Ultrasound-Liver/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Ultrasound-Liver</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Ultrasound-Liver.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/asset/evaluate_console_output_at_epoch50.png" width="880" height="auto">
<br><br>Image-Segmentation-Ultrasound-Liver

<a href="./projects/TensorFlowFlexUNet/Ultrasound-Liver/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Ultrasound-Liver/test was very low, and dice_coef_multiclass very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0059
dice_coef_multiclass,0.9971
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Ultrasound-Liver</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Ultrasound-Liver.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Ultrasound-Liver  Images</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<b>rgb_map = {Benign:green, Malignant:red}</b>
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/images/10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/masks/10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test_output/10013.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/images/10159.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/masks/10159.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test_output/10159.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/images/10151.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/masks/10151.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test_output/10151.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/images/10268.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/masks/10268.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test_output/10268.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/images/10242.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/masks/10242.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test_output/10242.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/images/10338.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test/masks/10338.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Ultrasound-Liver/mini_test_output/10338.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Improving artificial intelligence pipeline for liver malignancy diagnosis using ultrasound images and video frames </b><br>
Yiming Xu , Bowen Zheng , Xiaohong Liu , Tao Wu , Jinxiu Ju , Shijie Wang , Yufan Lian , <br>
Hongjun Zhang , Tong Liang , Ye Sang , Rui Jiang , Guangyu Wang , Jie Ren , Ting Chen<br>
<a href="https://academic.oup.com/bib/article/24/1/bbac569/6961609">
https://academic.oup.com/bib/article/24/1/bbac569/6961609</a>
<br>
<br>
<b>2. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model</a>
<br>
<br>
