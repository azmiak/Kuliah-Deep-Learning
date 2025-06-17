# Hands‑On Machine Learning with Scikit‑Learn, Keras, and TensorFlow

![Machine Learning](link)

**Concepts, Tools, and Techniques to Build Intelligent Systems**  
_Aurélien Géron, 2nd Edition_

Repository ini berisi contoh kode dan ringkasan setiap bab dari buku **Hands‑On Machine Learning with Scikit‑Learn, Keras, and TensorFlow**.  

## Daftar Isi

### Part I. The Fundamentals of Machine Learning

1. **The Machine Learning Landscape**  
   Pengantar ML: definisi, supervised vs unsupervised, batch vs online, instance‑ vs model‑based, tantangan umum, workflow end‑to‑end.

2. **End‑to‑End Machine Learning Project**  
   Contoh proyek ML lengkap (California Housing): framing, data acquisition, eksplorasi & visualisasi, preprocessing, train/test split, pipeline, model selection, tuning, deployment.

3. **Classification**  
   Klasifikasi biner & multikelas: confusion matrix, precision, recall, F1, ROC/AUC; error analysis; multilabel & multioutput.

4. **Training Models**  
   Regresi linear (normal equation & gradient descent), polynomial regression, learning curves, regularisasi (Ridge, Lasso, Elastic Net), logistic regression.

5. **Support Vector Machines**  
   Linear SVM (hard/soft margin), kernel trick (polynomial, RBF), SVM regression, hyperparameter tuning (C, γ).

6. **Decision Trees**  
   Algoritma CART, impurity (Gini & entropy), overfitting & pruning (max_depth, min_samples), regresi & klasifikasi, feature importance.

7. **Ensemble Learning and Random Forests**  
   Voting, bagging & pasting, random patches/subspaces, Random Forest, Extra‑Trees, boosting (AdaBoost, Gradient Boosting), stacking.

8. **Dimensionality Reduction**  
   Curse of dimensionality, projection (PCA, randomized & incremental PCA), kernel PCA, manifold learning (LLE), choosing components.

9. **Unsupervised Learning Techniques**  
   Clustering (K‑Means, DBSCAN), Gaussian Mixtures, anomaly & novelty detection (One‑Class SVM, Isolation Forest), brief on association rules.

### Part II. Neural Networks and Deep Learning

10. **Introduction to Artificial Neural Networks with Keras**  
    MLP & backpropagation, activation functions, Sequential API, building & training simple classifiers/regressors.

11. **Training Deep Neural Networks**  
    Vanishing/exploding gradients, initialization (Glorot, He), non‑saturating activations, BatchNorm, gradient clipping, optimizers (SGD, Adam), transfer learning basics.

12. **Custom Models and Training with TensorFlow**  
    Tensors, Variables, custom layers/models, custom loss/metric, GradientTape training loops, `@tf.function` for performance.

13. **Loading and Preprocessing Data with TensorFlow**  
    `tf.data` API (from arrays, CSV, TFRecord), map/shuffle/batch/prefetch, Keras preprocessing layers (Normalization, CategoryEncoding, TextVectorization).

14. **Deep Computer Vision Using Convolutional Neural Networks**  
    Convolution & pooling, CNN architectures (LeNet, AlexNet, VGG, ResNet), implement CNN di Keras, transfer learning (ResNet50, MobileNetV2).

15. **Processing Sequences Using RNNs and CNNs**  
    RNN, LSTM & GRU, windowing time series, SimpleRNN forecasting, seq2seq multi‑step prediction, handling long sequences (stateful & bidirectional).

16. **Natural Language Processing with RNNs and Attention**  
    Char‑RNN, sentiment analysis, sequence-to-sequence, attention mechanisms, Transformers (encoder/decoder), beam search.

17. **Representation Learning and Generative Learning Using Autoencoders and GANs**  
    Autoencoders (vanilla, denoising, sparse), Variational Autoencoders, Generative Adversarial Networks (DCGAN, StyleGAN overview).

18. **Reinforcement Learning**  
    MDPs, tabular Q‑Learning, Policy Gradient (REINFORCE), Deep Q‑Network (DQN), replay buffer, target network, extensions (Double DQN, Dueling, Prioritized Replay).

19. **Training and Deploying TensorFlow Models at Scale**  
    SavedModel & TensorFlow Serving, TFLite conversion & quantization, GPU & multi‑GPU training (`MirroredStrategy`), hyperparameter tuning with KerasTuner.

---
