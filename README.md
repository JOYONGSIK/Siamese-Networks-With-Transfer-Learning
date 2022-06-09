# Siamese-Networks-With-Transfer-Learning
For Face Recognition on Small Samples Datasets Using Transfer Learning(VGG16)

![image](https://user-images.githubusercontent.com/74164413/172753157-4c7e09ba-d81a-465e-b1b1-af02e4e0ac09.png)<br>
<h4 align="center">- Siamese Network. (image by author)</h1>

- 2개의 Input을 가지고 같은 네트워크를 통해서 Output Vector가 추출되는 구조를 가지고 있습니다.
- It has a structure in which the Output Vector is extracted through the same network with two inputs.
- Vector들은 서로 비교를 통해 두 입력의 유사도를 측정할 수 있습니다.
- Vectors can compare with each other to measure the similarity between the two inputs.

<hr>
<h4>논문을 참고했습니다.<br>
<br>you can read at https://ieeexplore.ieee.org/abstract/document/9116915?casa_token=8ATI0UDbXZMAAAAA:nGX6x0vAS076gA2EK4N_pkMujbwYNyJHRkqJBatJXZwN_cnJSMQAdUpqry57lYieeoMAMVo56Q </h4>

We are implementing face recognition using a “siamese network” architecture which consists of two similar CNN networks- and transfer learning 

* We improve the accuracy up to 95.2% by using transfer learning and VGG-16 Model which is pre-trained on ImageNet dataset.

* We assumed that batch sizes are equal to 32. We used ADAM Optimizer Function with α = 0.000005 as learning rate. 

reference : Heidari, Mohsen, and Kazim Fouladi-Ghaleh. "Using Siamese networks with transfer learning for face recognition on small-samples datasets." 2020 International Conference on Machine Vision and Image Processing (MVIP). IEEE, 2020. 
<hr> 

- Train.py을 실행하면 모델이 저장됩니다. 
- Running Train.py saves the model.
- Test.py을 실행하면 두 입력의 유사도를 측정할 수 있습니다.
- You can measure the similarity between the two inputs by running Test.py

* 미완성 코드입니다.

<hr>
reference : https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch 
